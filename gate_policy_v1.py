"""gate_policy_v1.py

Configurable Step 6 gate policies for simpleapp_v42+.

simpleapp_v40 and v41 use binary gating: a draft fails Step 6 if
rule_verifier_v1.verify_draft() returns pass=False (any violation in
any cap). This module decouples the verifier's FACTS (counts, violations)
from the DECISION (pass / fail) so different policies can be tried
without editing the verifier.

VERIFIER CONTRACT (extracted from a real v43 manifest, 2026-05-22):
  verify_result["violations"] is a list of dicts:
    {'cap': 'Cap 1', 'rule': 'the_way_x', 'count': 3}
    {'cap': 'Cap 4', 'rule': 'em_dash',   'count': 8, 'per_1k': 2.34}
  The `cap` field is a STRING like 'Cap 1' through 'Cap 19'.
  The `rule` field is the rule name (e.g. 'the_way_x', 'em_dash').
  The `count` field is an integer.
  Some entries have extra fields (e.g. `per_1k` for density caps).

  Earlier versions of this module looked for `cap_id` (int) and
  `rule_name`. That was wrong and silently passed every draft as a
  survivor on the 2026-05-22T19-24Z grimaldi run. The parser below
  accepts both shapes for backward-compat but prefers the real one.

Policy schema:

  {
    "name":               str,              # human-readable label
    "default_tolerance":  int,              # max hits per cap if not overridden
    "per_cap_tolerance":  {cap_id: int},    # cap-specific overrides (int keys or str)
    "max_caps_tripped":   int | None,       # max # of caps with > 0 hits; None = unlimited
    "hard_caps":          [cap_id, ...],    # caps that must be 0 regardless of tolerance
  }

Three presets are defined:

  STRICT — reproduces v40 / v41 behavior. All caps must be 0. Equivalent
    to the current binary verify_result["pass"] check.

  PRAGMATIC — splits caps by what's repairable post-draft. Cadence /
    observational fingerprints (Caps 1, 2, 3, 5, 10, 19) are hard-gated
    at 0 because rewriting them requires sentence-level voice judgment.
    Mechanical fingerprints (Caps 4, 16, 18) are tolerated up to the v10H
    cap definition's own budgets, because they can be deterministically
    cleaned post-draft. Caps 8, 9, 14 tolerate 1 hit (rare and easy to
    fix manually).

  USER_PROPOSED — the policy you asked about in chat: each cap may have
    up to 1 hit, and the draft may trip up to 3 caps total. Empirical
    note: on your six-draft batch this admits 1 of 6 drafts (b8dac6e3).

Public API:

  passes_policy(verify_result, policy) -> (bool, str)
    Returns (pass_flag, reason_string).
  build_policy_from_ui(preset, per_cap_overrides_json, max_caps) -> dict
    Helper for the Streamlit sidebar.
  format_policy(policy) -> str
    Human-readable description for the UI.
"""
from __future__ import annotations
import json
from typing import Optional


# Cap classes per the cadence/mechanical split from chat analysis.
_CADENCE_CAPS  = {1, 2, 3, 5, 10, 19}    # voice fingerprints — hard-gate
_MECHANICAL_CAPS = {4, 16, 18}            # repairable post-draft
_LIGHT_CAPS    = {8, 9, 14}               # rare; tolerate 1


STRICT_POLICY = {
    "name":              "STRICT",
    "description":       "Reproduces v40/v41 behavior. All caps must be 0.",
    "default_tolerance": 0,
    "per_cap_tolerance": {},
    "max_caps_tripped":  None,
    "hard_caps":         [],   # default_tolerance=0 makes hard_caps redundant
}


PRAGMATIC_POLICY = {
    "name":              "PRAGMATIC",
    "description":       "Cadence caps (1,2,3,5,10,19) hard at 0. Mechanical caps "
                         "(4,16,18) tolerated within v10H budgets. Light caps "
                         "(8,9,14) tolerate 1.",
    "default_tolerance": 0,
    "per_cap_tolerance": {
        # Mechanical caps — v10H budgets
        4:  6,    # Cap 4 em-dashes: 0-4 under 3000 words, 0-6 over (use the larger)
        16: 0,    # Cap 16: actually 0 per def ("zero narration sentences..."),
                  #   but verifier counts per-sentence; keep at 0
        18: 0,    # Cap 18 tautological loop: 0 per def
        # Light caps
        8:  1,
        9:  1,
        14: 1,
    },
    "max_caps_tripped":  None,
    "hard_caps":         list(_CADENCE_CAPS),
}


USER_PROPOSED_POLICY = {
    "name":              "USER_PROPOSED",
    "description":       "1 hit per cap, max 3 caps tripped per draft. "
                         "On the 6-draft test batch from chat, this admits 1 of 6.",
    "default_tolerance": 1,
    "per_cap_tolerance": {},
    "max_caps_tripped":  3,
    "hard_caps":         [],
}


PRESETS = {
    "STRICT":        STRICT_POLICY,
    "PRAGMATIC":     PRAGMATIC_POLICY,
    "USER_PROPOSED": USER_PROPOSED_POLICY,
}


# ============================================================================
# Evaluation
# ============================================================================
def _extract_cap_id(violation: dict) -> Optional[int]:
    """Extract the integer cap ID from a violation dict.

    The real rule_verifier_v1 contract uses `cap`: 'Cap N' string. Older
    versions of this code expected `cap_id`: int. Handle both."""
    # Preferred: real verifier contract
    cap_str = violation.get("cap", "")
    if isinstance(cap_str, str) and cap_str:
        import re
        m = re.match(r"\s*Cap\s+(\d+)", cap_str, re.IGNORECASE)
        if m:
            return int(m.group(1))
        # Maybe it's just a number as a string
        try:
            return int(cap_str)
        except (TypeError, ValueError):
            pass
    # Fallback: legacy cap_id field (int)
    cap_id = violation.get("cap_id")
    if cap_id is not None:
        try:
            return int(cap_id)
        except (TypeError, ValueError):
            return None
    return None


def passes_policy(verify_result: dict, policy: dict) -> tuple[bool, str]:
    """Evaluate verify_result against policy. Returns (passes, reason_string).

    Uses verify_result['violations'] as ground truth (a list of dicts each
    with `cap` (e.g. 'Cap 1'), `rule`, and `count`). Ignores
    verify_result['pass'] entirely — that field is the verifier's own binary
    read; this function reapplies a configurable policy to the raw counts."""
    violations = verify_result.get("violations", []) or []

    default_tol  = int(policy.get("default_tolerance", 0))
    per_cap_raw  = policy.get("per_cap_tolerance", {}) or {}
    # Accept both int and str keys (JSON deserializes int dict keys as str).
    per_cap = {}
    for k, v in per_cap_raw.items():
        try:
            per_cap[int(k)] = int(v)
        except (TypeError, ValueError):
            continue
    max_caps     = policy.get("max_caps_tripped")
    hard_caps    = set(int(c) for c in (policy.get("hard_caps") or []))

    over_budget = []
    caps_tripped = 0

    for v in violations:
        cap_id = _extract_cap_id(v)
        if cap_id is None:
            continue
        count = int(v.get("count", 0))
        if count <= 0:
            continue

        caps_tripped += 1

        if cap_id in hard_caps:
            over_budget.append(f"Cap {cap_id} hard ({count})")
            continue

        budget = per_cap.get(cap_id, default_tol)
        if count > budget:
            over_budget.append(f"Cap {cap_id}={count}>{budget}")

    # Apply max-caps-tripped check
    if max_caps is not None and caps_tripped > int(max_caps):
        over_budget.append(f"{caps_tripped} caps tripped (max {max_caps})")

    if over_budget:
        return False, "FAIL — " + "; ".join(over_budget)
    return True, f"PASS — {caps_tripped} cap(s) within budget"


# ============================================================================
# UI helpers
# ============================================================================
def build_policy_from_ui(
    preset_name: str,
    per_cap_overrides_json: str = "",
    max_caps_override: Optional[int] = None,
) -> dict:
    """Construct a policy from sidebar inputs. Starts from the named preset,
    then layers per-cap overrides parsed from a JSON string, then a
    max_caps override if provided. Invalid JSON falls back to the preset."""
    base = dict(PRESETS.get(preset_name, STRICT_POLICY))
    # Deep-copy the dict members we'll mutate
    base["per_cap_tolerance"] = dict(base.get("per_cap_tolerance", {}))
    base["hard_caps"] = list(base.get("hard_caps", []))

    if per_cap_overrides_json and per_cap_overrides_json.strip():
        try:
            overrides = json.loads(per_cap_overrides_json)
            if isinstance(overrides, dict):
                for k, v in overrides.items():
                    try:
                        base["per_cap_tolerance"][int(k)] = int(v)
                    except (TypeError, ValueError):
                        continue
        except json.JSONDecodeError:
            # Leave base unmodified; caller may want to surface this.
            pass

    if max_caps_override is not None:
        base["max_caps_tripped"] = (
            int(max_caps_override) if max_caps_override > 0 else None
        )

    return base


def format_policy(policy: dict) -> str:
    """One-line summary for the UI."""
    name = policy.get("name", "?")
    desc = policy.get("description", "")
    return f"{name}: {desc}"
