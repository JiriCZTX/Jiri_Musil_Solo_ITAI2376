"""
Best-available LLM model selection (reviewer correction §9 + follow-up #4).

Per PATH_B5_PLUS_FINAL.md §8, no Anthropic model version is hardcoded inside
extraction / adjudication / pseudo-label code paths. Every call goes through
the helpers below, which:

  1. Resolve the model ID from environment variables first
     (BEST_AVAILABLE_OPUS / BEST_AVAILABLE_SONNET) so the operator can pin a
     specific build for a batch run without code changes.
  2. Fall back to a documented-current default (April 2026) only if the env
     var is unset.
  3. Always return the *exact* model ID string — never a family alias like
     "Opus 4.6" — so the provenance ledger captures the build that actually
     served the response.

Reviewer follow-up #4 — sampling-knob discipline:
  • Do NOT pass temperature, top_p, or top_k to messages.create. The
    Anthropic API treats them as opaque hints; relying on them for
    determinism is unsound.
  • Determinism is enforced INSTEAD by:
       (a) schema validation on every LLM response (strict JSON shape +
           required fields)
       (b) exact substring validation for any extracted span
           (text[start:end] == span_text byte-for-byte)
       (c) provenance-ledger logging of the exact model ID returned by
           the API (response.model), not the requested model.
  • `make_call_kwargs(model_id, **extra)` REJECTS temperature/top_p/top_k
    in `extra` so a hot-fix can't sneak them back in.

Usage:
    from config.llm_models import best_available_opus, make_call_kwargs
    model_id = best_available_opus()
    kwargs = make_call_kwargs(model_id, max_tokens=4096, system=PROMPT)
    response = anthropic_client.messages.create(**kwargs, messages=msgs)
    exact_id = response.model            # log this in the ledger, not model_id
    ledger.append(make_entry(teacher_model_id=exact_id, ...))

The exact ID returned by the API MUST be persisted on every artifact.
"""
from __future__ import annotations
import os
from typing import Any, Dict


# ---------------------------------------------------------------------------
# Environment variable names
# ---------------------------------------------------------------------------
ENV_BEST_OPUS = "BEST_AVAILABLE_OPUS"
ENV_BEST_SONNET = "BEST_AVAILABLE_SONNET"
ENV_BEST_HAIKU = "BEST_AVAILABLE_HAIKU"


# ---------------------------------------------------------------------------
# Documented-current defaults — updated when a new build ships
# ---------------------------------------------------------------------------
# Source: HANDOVER_V9.md §7.12 ("Opus 4.7 launched April 2026").
# These are *defaults* — every adjudication batch logs the actual exact ID
# returned by best_available_*().
_DEFAULT_OPUS = "claude-opus-4-7-20260418"
_DEFAULT_SONNET = "claude-sonnet-4-6-20260201"
_DEFAULT_HAIKU = "claude-haiku-4-5-20251001"


def best_available_opus() -> str:
    """Return the exact Opus model ID to use for adjudication / pseudo-labeling.
    Reads BEST_AVAILABLE_OPUS env var first; falls back to the documented-current
    default if unset. Never returns a family alias like 'Opus 4.6'."""
    return os.environ.get(ENV_BEST_OPUS, _DEFAULT_OPUS)


def best_available_sonnet() -> str:
    """Sibling helper for cross-extraction (e.g. JD LF #4)."""
    return os.environ.get(ENV_BEST_SONNET, _DEFAULT_SONNET)


def best_available_haiku() -> str:
    """Sibling helper for high-volume / low-stakes drafting passes."""
    return os.environ.get(ENV_BEST_HAIKU, _DEFAULT_HAIKU)


# ---------------------------------------------------------------------------
# Sampling-knob discipline (reviewer follow-up #4)
# ---------------------------------------------------------------------------
# Forbid temperature / top_p / top_k from any LLM call kwargs. Determinism is
# enforced via schema + exact-substring validation, not via sampling knobs.
FORBIDDEN_SAMPLING_KEYS: tuple = ("temperature", "top_p", "top_k")


class ForbiddenSamplingKnob(ValueError):
    """Raised when an LLM call kwargs dict tries to set temperature/top_p/top_k."""


def make_call_kwargs(model_id: str, **extra: Any) -> Dict[str, Any]:
    """
    Build the kwargs for anthropic_client.messages.create with discipline:

      • model_id is required and forwarded verbatim.
      • temperature / top_p / top_k are REJECTED — pass them and you get
        ForbiddenSamplingKnob. Reviewer follow-up #4 enforces this so callers
        can't silently rely on sampling-based determinism.
      • Anything else (max_tokens, system, tools, ...) passes through.

    Returns a dict ready to splat into messages.create:
        client.messages.create(**make_call_kwargs(...), messages=msgs)
    """
    bad = [k for k in extra if k in FORBIDDEN_SAMPLING_KEYS]
    if bad:
        raise ForbiddenSamplingKnob(
            f"Sampling knobs are forbidden for adjudication / silver-label calls: "
            f"{bad}. Use schema validation + exact substring validation as the "
            f"determinism control instead. See config/llm_models.py docstring."
        )
    out: Dict[str, Any] = {"model": model_id}
    out.update(extra)
    return out


def log_exact_model_id(response: Any) -> str:
    """
    Return the exact model ID the Anthropic API actually served the response
    on. Anthropic's response object exposes `.model` post-2024 — fall back to
    `request.model` if not available. The caller is expected to persist this
    via data.provenance.make_entry(teacher_model_id=...).

    Why not just log `model_id` from make_call_kwargs?
      The API may map a family alias to a specific dated build server-side.
      The response's `.model` is the ground-truth build that served you;
      logging the request alias would be lossy.
    """
    exact = getattr(response, "model", None)
    if isinstance(exact, str) and exact:
        return exact
    # Fallback: try response.id-prefix or response.usage.model — best-effort
    return getattr(response, "id", "") or "unknown_model_id"


def resolved_models_snapshot() -> Dict[str, str]:
    """Snapshot of every model ID resolved by this module *right now*.
    Persist this dict alongside any final-benchmark / Gate-2 / pilot run so
    auditors can reproduce which builds served the answers."""
    return {
        "opus": best_available_opus(),
        "sonnet": best_available_sonnet(),
        "haiku": best_available_haiku(),
        "resolved_via_env_opus": ENV_BEST_OPUS in os.environ,
        "resolved_via_env_sonnet": ENV_BEST_SONNET in os.environ,
        "resolved_via_env_haiku": ENV_BEST_HAIKU in os.environ,
    }


def _self_test() -> Dict[str, object]:
    snap = resolved_models_snapshot()
    issues: list[str] = []

    # Forbidden-knob check: each forbidden kwarg must raise.
    for bad_key in FORBIDDEN_SAMPLING_KEYS:
        try:
            make_call_kwargs(best_available_opus(), max_tokens=1024, **{bad_key: 0.0})
        except ForbiddenSamplingKnob:
            pass
        else:
            issues.append(f"{bad_key} should have raised ForbiddenSamplingKnob")

    # Allowed kwargs pass through cleanly
    try:
        kw = make_call_kwargs(best_available_opus(), max_tokens=1024,
                              system="EXTRACT", tools=[])
        if kw["model"] != best_available_opus():
            issues.append("make_call_kwargs lost model id")
        if "temperature" in kw:
            issues.append("temperature unexpectedly present in kwargs")
    except Exception as e:
        issues.append(f"make_call_kwargs allowed-kwargs path raised: {e!r}")

    return {
        "snapshot": snap,
        "no_family_aliases": not any(
            v in {"Opus 4.6", "Opus 4.7", "Sonnet 4.5", "Sonnet 4.6"}
            for k, v in snap.items() if k in {"opus", "sonnet", "haiku"}
        ),
        "opus_id_is_exact_form": "-2026" in best_available_opus()
                                  or "-2025" in best_available_opus(),
        "forbidden_sampling_knobs": list(FORBIDDEN_SAMPLING_KEYS),
        "issues": issues,
    }


if __name__ == "__main__":
    import json
    print(json.dumps(_self_test(), indent=2))
