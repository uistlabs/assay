import subprocess
import sys
import os


def test_survives_non_utf8_byte(tmp_path):
    """Verify non-UTF-8 bytes in stdin don't crash the tee."""
    raw = tmp_path / "raw.log"
    red = tmp_path / "vol.log"
    env = {"HF_TOKEN": "hf_SECRET", "RUNPOD_API_KEY": "", "PYTHONPATH": "src"}
    # Pipe BYTES containing invalid UTF-8 (0xff), plus a secret and GATE marker
    proc = subprocess.run(
        [sys.executable, "-m", "assay.log_tee", str(raw), str(red)],
        input=b'before \xff after\nline hf_SECRET\nGATE PASSED\n',
        capture_output=True,
        env=env,
    )
    # Must NOT crash (returncode == 0)
    assert proc.returncode == 0, f"log_tee crashed with rc={proc.returncode}\nstderr: {proc.stderr}"
    # Redacted volume must exist and contain the marker but NOT the secret
    assert red.exists(), "redacted volume file not created"
    red_content = red.read_text()
    assert "GATE PASSED" in red_content, "GATE marker not in redacted output"
    assert "hf_SECRET" not in red_content, "secret leaked to redacted volume"
    # Raw file contains unredacted secret (because local ephemeral)
    assert "hf_SECRET" in raw.read_text(), "secret not in raw file"


def test_redacts_secret_to_volume_but_keeps_raw_local(tmp_path):
    raw = tmp_path / "raw.log"
    red = tmp_path / "vol.log"
    env = {"HF_TOKEN": "hf_SECRET", "RUNPOD_API_KEY": "rp_KEY"}
    proc = subprocess.run(
        [sys.executable, "-m", "assay.log_tee", str(raw), str(red)],
        input="line hf_SECRET and rp_KEY\nGATE PASSED\n",
        capture_output=True, text=True, env={**env, "PYTHONPATH": "src"},
    )
    assert proc.returncode == 0
    # console passthrough still shows everything (redacted)
    assert "***" in proc.stdout and "hf_SECRET" not in proc.stdout
    # raw ephemeral = unredacted (local only), redacted volume file = scrubbed
    assert "hf_SECRET" in raw.read_text()
    assert "hf_SECRET" not in red.read_text() and "rp_KEY" not in red.read_text()
    assert "GATE PASSED" in red.read_text()


def test_stalled_volume_does_not_block_and_spills_locally(tmp_path):
    """The volume sink is a NAMED PIPE (FIFO) with no reader -> writes to it block.
    The tee must still consume all of stdin, keep the raw local file complete, and
    spill every redacted line to the local failover file - no loss - rather
    than wedging."""
    raw = tmp_path / "raw.log"
    vol = tmp_path / "vol.fifo"
    spill = tmp_path / "vol.spill"
    os.mkfifo(str(vol))  # nobody reads it -> a real write to it blocks
    lines = "".join(f"line-{i}\n" for i in range(500))
    proc = subprocess.run(
        [sys.executable, "-m", "assay.log_tee", str(raw), str(vol), str(spill)],
        input=lines, capture_output=True, text=True,
        env={"PYTHONPATH": "src", "HF_TOKEN": "", "RUNPOD_API_KEY": ""},
        timeout=30,  # MUST finish; a synchronous tee would hang here forever
    )
    assert proc.returncode == 0
    assert raw.read_text() == lines  # raw local sink is complete + stall-proof
    # The reader-less FIFO ENXIOs on open(2) (O_NONBLOCK), so every redacted
    # line routes straight to the spill - assert genuine no-loss (every line
    # present, exact count), not just that the spill file exists.
    assert spill.exists(), "overflow must spill to the local failover file"
    spill_text = spill.read_text()
    assert "line-0" in spill_text and "line-499" in spill_text
    assert spill_text.count("\n") == 500, "all 500 redacted lines must survive"


def test_two_arg_invocation_spills_local_not_volume(tmp_path):
    """CRITICAL: pod_entry.sh's log_tee call must never let the local-failover spill
    land on the (possibly stalled) volume. This drives log_tee with ONLY the two
    required args (raw, vol) - no explicit spill path - and reuses the FIFO-stall
    mechanism from test_stalled_volume_does_not_block_and_spills_locally to force an
    overflow. The derived spill must be <raw>.spill (local, stall-proof) and
    <vol>.spill (on the stalled mount) must never be created."""
    raw = tmp_path / "raw.log"
    vol = tmp_path / "vol.fifo"
    os.mkfifo(str(vol))  # nobody reads it -> a real write to it blocks
    lines = "".join(f"line-{i}\n" for i in range(500))
    proc = subprocess.run(
        [sys.executable, "-m", "assay.log_tee", str(raw), str(vol)],  # NO 3rd arg
        input=lines, capture_output=True, text=True,
        env={"PYTHONPATH": "src", "HF_TOKEN": "", "RUNPOD_API_KEY": ""},
        timeout=30,  # MUST finish; a volume-path spill would hang on the FIFO open()
    )
    assert proc.returncode == 0
    assert raw.read_text() == lines
    raw_spill = tmp_path / "raw.log.spill"
    vol_spill = tmp_path / "vol.fifo.spill"
    assert raw_spill.exists(), "2-arg default spill must derive from the RAW local path"
    assert not vol_spill.exists(), "spill must NEVER be derived from the volume path"
    assert raw_spill.read_text().count("\n") == 500, "all 500 lines must survive"


def test_redacts_across_async_boundary(tmp_path):
    raw = tmp_path / "raw.log"
    vol = tmp_path / "vol.log"
    proc = subprocess.run(
        [sys.executable, "-m", "assay.log_tee", str(raw), str(vol)],
        input="tok hf_SECRET here\nGATE PASSED\n", capture_output=True, text=True,
        env={"PYTHONPATH": "src", "HF_TOKEN": "hf_SECRET", "RUNPOD_API_KEY": ""},
        timeout=30,
    )
    assert proc.returncode == 0
    assert "hf_SECRET" not in vol.read_text()
    assert "GATE PASSED" in vol.read_text()
    assert "hf_SECRET" in raw.read_text()  # raw stays unredacted + local
