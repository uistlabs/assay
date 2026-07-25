from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from assay.recipes import Calib, Recipe

# Real llm-compressor NVFP4 API (verified against the W4A4-FP4 example):
#   QuantizationModifier(targets="Linear", scheme="NVFP4", ignore=["lm_head"])
#   oneshot(model=, dataset=, recipe=, max_seq_length=, num_calibration_samples=)
#   model.save_pretrained(out, save_compressed=True)


def load_calibration(calib: Calib, tokenizer):
    """Load + tokenize the calibration set with the SAME tokenizer used to load
    the model being quantized (caller passes the tokenizer loaded from
    model_path). Kept separate from quantize_to_nvfp4 so the shape can be
    inspected without loading the model.

    Deliberately takes a tokenizer rather than loading its own from the
    recipe's base_model: base_model is the hub identifier while model_path
    (what's actually being quantized) is a local volume path - if those ever
    diverge, a self-loaded tokenizer here would calibrate with the wrong
    vocab/chat template."""
    from datasets import Dataset, load_dataset

    # streaming=True so we never materialize/cache the whole dataset (ultrachat_200k
    # generates ~515k examples across 4 splits) just to keep num_samples
    # of them: on a bounded container disk that generation is GBs of cache and minutes
    # of runtime for nothing. Shuffle within a bounded buffer, take N, then materialize
    # only that small slice into an in-memory Dataset for the tokenizing map.
    stream = load_dataset(calib.dataset, split=calib.split, streaming=True)
    stream = stream.shuffle(seed=42, buffer_size=10_000)
    ds = Dataset.from_list(list(stream.take(calib.num_samples)))

    def _tokenize(sample):
        text = tokenizer.apply_chat_template(sample["messages"], tokenize=False)
        return tokenizer(text, truncation=True, max_length=calib.max_seq_len)

    return ds.map(_tokenize, remove_columns=ds.column_names)


def quantize_to_nvfp4(recipe: Recipe, model_path: str, out_dir: str, hb=None) -> str:
    """PTQ the model at model_path to NVFP4; write compressed checkpoint to out_dir."""
    import gc

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from llmcompressor import oneshot, reset_session
    from llmcompressor.modifiers.quantization import QuantizationModifier

    if hb:
        hb.emit("quantize", f"loading {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    ds = load_calibration(recipe.calib, tokenizer)
    q = QuantizationModifier(targets="Linear", scheme=recipe.quant_scheme, ignore=["lm_head"])

    if hb:
        hb.emit("quantize", f"oneshot {recipe.quant_scheme} calibration")
    oneshot(
        model=model,
        dataset=ds,
        recipe=q,
        max_seq_length=recipe.calib.max_seq_len,
        num_calibration_samples=recipe.calib.num_samples,
    )

    if hb:
        hb.emit("quantize", f"saving compressed checkpoint to {out_dir}")
    model.save_pretrained(out_dir, save_compressed=True)
    tokenizer.save_pretrained(out_dir)

    # Release the GPU before returning (verified on metal): vLLM v1's
    # EngineCore is a SEPARATE process that checks DEVICE-WIDE free memory at
    # startup (torch.cuda.mem_get_info), so whatever THIS process still holds
    # counts against the eval engine - ~5.7 GiB of quantize residue tripped
    # its "free < desired gpu_memory_utilization" ValueError. Two things pin
    # the weights after oneshot returns: our locals, and llm-compressor's
    # GLOBAL CompressionSession (session_functions._global_session), whose
    # lifecycle.state.model still references the model. reset_session() drops
    # that ref and finalizes any un-finalized modifiers (removing calibration
    # hooks); only then can gc actually collect the tensors (transformers
    # module graphs are cycle-heavy, so an explicit collect, not refcounting),
    # and empty_cache() returns the freed blocks from torch's caching
    # allocator to the driver where the eval subprocess can see them as free.
    reset_session()
    del model, tokenizer, ds, q
    gc.collect()
    torch.cuda.empty_cache()
    if hb:
        free, total = torch.cuda.mem_get_info()
        hb.emit("quantize",
                f"gpu released: free={free / 2**30:.2f}/{total / 2**30:.2f} GiB")
    return out_dir
