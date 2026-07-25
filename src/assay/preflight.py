"""Launch-side, NO-NETWORK operator reminders (printed by scripts/launch.sh before the
paid pod is created). Deliberately NOT an in-pod base-card fetch: that would put an
HF request on the paid critical path (504-prone) and land in a log nobody reads. This
prints to the operator's terminal at spend-time. ASCII-only (reaches the terminal/log)."""
from __future__ import annotations


def base_model_card_url(base_model: str) -> str:
    """The HF card URL for a base model id, constructed with NO network call."""
    return f"https://huggingface.co/{base_model}"


def launch_reminder(recipe) -> str:
    """One-screen reminder tying this run's recipe to its base model's card + the
    authoring checklist. See docs/recipe-authoring.md for the full list."""
    url = base_model_card_url(recipe.base_model)
    return (
        "  ---------------------------------------------------------------------\n"
        f"  recipe: {recipe.slug}  ->  base model: {recipe.base_model}\n"
        f"  base model card: {url}\n"
        "  Before spending on this run, confirm the recipe encodes the base card's\n"
        "  eval protocol: sampling (temperature/top_p), avg@K, chat template, and\n"
        "  system prompt - or that each deviation is a documented, deliberate choice.\n"
        "  Checklist: docs/recipe-authoring.md\n"
        "  ---------------------------------------------------------------------"
    )
