"""failure_diagnostic_v1.py

Diagnostic module for simpleapp_v40 Step 6 failures. Consumes rejected
drafts (each carrying `verify_result` from rule_verifier_v1.verify_draft)
and produces a structured diagnosis of what is going wrong upstream.

The module operates at two layers:

  Layer 1 — Aggregation (deterministic, no LLM call).
    Counts cap incidence across all rejected drafts in a batch, ranks
    caps by failure rate (how many drafts trip each cap) and by severity
    (total hits across the batch), and pulls every flagged passage so the
    operator can read the actual offending sentences.

  Layer 2 — Characterization (LLM call, optional).
    For each cap that fired, asks Claude to look at the actual flagged
    sentences and characterize what is compelling the failure: a packet
    instruction-form, a structural pressure from the outline, an
    AUTHOR_CONSTRUCTION block trigger, or a freestanding drafter habit.
    Then proposes the upstream lever to pull — packet edit, author block
    edit, or sentence-level repair pass.

Public API:

  diagnose_batch(rejected_drafts, outline_text=None, author_construction=None,
                 client=None, model=None) -> dict
    Returns a diagnosis dict with per-cap aggregation, flagged passages
    grouped by cap, and (if client+model provided) per-cap LLM
    characterizations and fix proposals.

  format_diagnosis_for_log(diagnosis) -> str
    Renders the diagnosis as a readable text block for the Streamlit
    run log.

Wire-up point in simpleapp_v40.py: after filter_compliant_drafts() returns
in the inner loop (around line 340), when len(survivors) < min_survivors,
call diagnose_batch(batch_rejected, ...) and log the result before the
regeneration call. The rejected drafts already carry verify_result from
filter_compliant_drafts, so no pipeline plumbing change is needed.

This module does NOT modify the gate decision. It produces diagnostic
output only. The gate stays as-is; the diagnostic informs upstream
changes the operator makes between runs.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from typing import Optional


# Caps that v10H treats as voice-fingerprint cadences (sentence-level
# rewriting requires voice judgment, not mechanical repair).
CADENCE_CAPS = {1, 2, 3, 5, 8, 9, 10, 14, 18, 19}

# Caps that are mechanical and deterministically repairable post-draft.
MECHANICAL_CAPS = {4, 16}


# ============================================================================
# Layer 1 — Aggregation
# ============================================================================
def _extract_cap_id(violation: dict) -> Optional[int]:
    """Extract integer cap ID from a violation dict.

    Real rule_verifier_v1 contract uses `cap`: 'Cap N' string. Earlier
    versions of this module looked for `cap_id`: int (which silently
    matched nothing on the real data). Handle both."""
    import re
    cap_str = violation.get("cap", "")
    if isinstance(cap_str, str) and cap_str:
        m = re.match(r"\s*Cap\s+(\d+)", cap_str, re.IGNORECASE)
        if m:
            return int(m.group(1))
        try:
            return int(cap_str)
        except (TypeError, ValueError):
            pass
    cap_id = violation.get("cap_id")
    if cap_id is not None:
        try:
            return int(cap_id)
        except (TypeError, ValueError):
            return None
    return None


def _extract_rule_name(violation: dict) -> str:
    """Real verifier uses `rule`; legacy field was `rule_name`."""
    return violation.get("rule") or violation.get("rule_name") or ""


def aggregate_failures(rejected_drafts: list[dict]) -> dict:
    """Tally cap incidence across rejected drafts.

    Returns:
      {
        "n_drafts": int,
        "per_cap": {
          cap_id: {
            "drafts_tripped": int,     # how many drafts hit this cap
            "total_hits": int,         # sum of hits across drafts
            "max_hits_single_draft": int,
            "draft_ids": [run_id, ...],
            "rule_name": str,          # from verify_result
          },
          ...
        },
        "cap_order_by_breadth": [cap_id, ...],   # most drafts tripped first
        "cap_order_by_severity": [cap_id, ...],  # most total hits first
      }
    """
    per_cap = defaultdict(lambda: {
        "drafts_tripped": 0,
        "total_hits": 0,
        "max_hits_single_draft": 0,
        "draft_ids": [],
        "rule_name": "",
    })

    for d in rejected_drafts:
        vr = d.get("verify_result", {})
        violations = vr.get("violations", []) or []

        for v in violations:
            cap_id = _extract_cap_id(v)
            if cap_id is None:
                continue
            hits = int(v.get("count", 0))
            if hits <= 0:
                continue
            entry = per_cap[cap_id]
            entry["drafts_tripped"] += 1
            entry["total_hits"] += hits
            entry["max_hits_single_draft"] = max(entry["max_hits_single_draft"], hits)
            entry["draft_ids"].append(d.get("run_id", "?"))
            rule = _extract_rule_name(v)
            if rule:
                entry["rule_name"] = rule

    cap_order_by_breadth = sorted(
        per_cap.keys(),
        key=lambda c: (-per_cap[c]["drafts_tripped"], -per_cap[c]["total_hits"]),
    )
    cap_order_by_severity = sorted(
        per_cap.keys(),
        key=lambda c: (-per_cap[c]["total_hits"], -per_cap[c]["drafts_tripped"]),
    )

    return {
        "n_drafts": len(rejected_drafts),
        "per_cap": dict(per_cap),
        "cap_order_by_breadth": cap_order_by_breadth,
        "cap_order_by_severity": cap_order_by_severity,
    }


def collect_flagged_passages(rejected_drafts: list[dict]) -> dict:
    """Group flagged passages by cap_id across all rejected drafts.

    Returns:
      {
        cap_id: [
          {"draft_id": run_id, "rule": str, "excerpt": str, "context": str},
          ...
        ],
        ...
      }
    """
    by_cap = defaultdict(list)
    for d in rejected_drafts:
        vr = d.get("verify_result", {})
        for fp in vr.get("flagged_passages", []) or []:
            cap_id = _extract_cap_id(fp)
            if cap_id is None:
                # Try to recover from rule name via violations list.
                rule = fp.get("rule") or fp.get("rule_name") or ""
                if rule:
                    for v in vr.get("violations", []):
                        if _extract_rule_name(v) == rule:
                            cap_id = _extract_cap_id(v)
                            break
            if cap_id is None:
                continue
            by_cap[cap_id].append({
                "draft_id": d.get("run_id", "?"),
                "rule": fp.get("rule") or fp.get("rule_name") or "",
                "excerpt": fp.get("excerpt", "") or fp.get("context", ""),
                "context": fp.get("context", ""),
            })
    return dict(by_cap)


# ============================================================================
# Layer 2 — Characterization (LLM-driven)
# ============================================================================
DIAGNOSIS_PROMPT_TEMPLATE = """\
You are diagnosing why an AI-assisted commercial-fiction drafter is repeatedly producing a specific prose fingerprint. The drafter generates chapters from a chapter outline and an AUTHOR_CONSTRUCTION block (a characterization of a primary bestselling author's habits). A rule-based scanner detected that the drafter produced the following cap-{cap_id} violations across {n_drafts} draft attempts in a single batch.

CAP DEFINITION:
{cap_definition}

FLAGGED PASSAGES FROM THIS BATCH:
{flagged_passages}

{outline_section}{author_section}

Your task is to characterize the root cause and propose an upstream fix. Look at the actual flagged sentences and answer these four questions, briefly and concretely:

1. PATTERN: What is the specific construction the drafter keeps reaching for? (Quote the recurring shape, e.g. "X but Y", "the way she Xs", "And A and B and C and D".)

2. TRIGGER: What is most likely compelling it? Choose one and explain in one sentence:
   (a) An instruction-form in the chapter outline (e.g. "show competence through how she handles X");
   (b) A habit specified or implied by the AUTHOR_CONSTRUCTION block;
   (c) A structural pressure (e.g. the beat requires summary; the drafter defaults to negation-pivot to compress);
   (d) A freestanding drafter habit not traceable to the inputs.

3. LEVER: Where should the fix be applied?
   - PACKET edit: revise outline phrasing to remove the triggering instruction-form;
   - AUTHOR block edit: remove or replace the habit prescription;
   - POST-DRAFT repair: deterministically fix at sentence level after draft generation;
   - DRAFTER-LEVEL: requires a prompt-layer change (e.g. add positive design guidance to draft_chapter_prompt_v1.txt).

4. CONCRETE FIX: One specific, copy-pasteable change. If PACKET or AUTHOR, give the exact replacement text. If POST-DRAFT, name the regex transformation. If DRAFTER-LEVEL, give the exact sentence to add to the drafting prompt.

Respond in this exact format with no preamble:

PATTERN: [one paragraph]
TRIGGER: [letter + one sentence]
LEVER: [one of the four lever types]
CONCRETE FIX: [the specific change]
"""


# Cap definitions lifted from normalize_outline_v10H.txt (the binding source).
# These are pasted verbatim from the HARD CAPS section so the diagnostic has
# the canonical definition in front of it.
CAP_DEFINITIONS = {
    1: """Cap 1 — Characterisation-by-tradecraft (observational construction).
Function: The model defaults to compact characterisation through observational phrasing. In this register, characterisation should come from action, choice, pressure, dialogue, and concrete behavior.
Prohibition: Zero instances of "the way [person/thing] [verbs]," "the way [it] felt," "the way [thing] wanted/wants to be [verbed]," "the way [person] [verbs] when/who," "the way you/one know(s)," and sense-perception variants such as "the noise/sound a [person/thing] makes." State the conclusion flatly or trust the action.""",

    2: """Cap 2 — Observational periphrasis (comparison-as-characterisation).
Function: This is the substitute the model reaches for when direct observational construction is closed. The function is the same: characterisation through ornamental comparison.
Prohibition: Zero instances in narration or interior thought of "as though," "as if," "in the manner of," "like a woman/man/person who," or equivalent observational framing. Characters may use natural comparison in dialogue.""",

    3: """Cap 3 — Negation-pivot (reverse definition).
Function: The model often defines a thing by first saying what it is not. This creates a literary-corrective cadence. In the target register, definitions should move forward.
Prohibition: Zero instances in narration or interior thought of "not X but Y," "not quite X and not quite Y," "not the X but the Y," "Not the X. The Y," "Not the X; the Y," or "Not [adjective]. [Same subject] [different adjective]." Characters may use ordinary negation in dialogue.""",

    4: """Cap 4 — Em dashes (overused interruption signal).
Function: Em dashes carry pause, parenthesis, and interruption, but the model overuses them at detectable density.
Prohibition: Target 0–4 em dashes for chapters under 3,000 words and 0–6 for chapters over 3,000 words. No em dash in the first paragraph. Never two em-dash sentences in a row. Prefer comma, full stop, parentheses, or sentence break.""",

    5: """Cap 5 — Emotion naming (displaced feeling).
Function: The model often turns emotion into narrator metaphor. In first person, the POV character may name feelings plainly. In close third, interiority should stay embodied and specific.
Prohibition: Zero narration such as "she felt a wave of grief," "a flush of anger," "with a sense of dread," or similar displaced feeling-metaphor. First-person POV may use plain self-naming. Do not use meta-naming such as "I named the feeling" or "I named the thing in my chest." """,

    8: """Cap 8 — Aphoristic standalone (abstract-subject verdict).
Function: The model uses short abstract verdicts as literary punctuation.
Prohibition: Zero sentences under 10 words whose subject is an abstraction (morning, silence, air, hour, weather, quiet, room, house, year, dark, wind, rain, cold, dusk, dawn, world) and whose verb evaluates or negates. Do not use abstract-subject verdicts at paragraph ends.""",

    9: """Cap 9 — Explanatory backfill (action plus justification).
Function: The model states an action and then explains the psychology in a "because/since I had…" tail.
Prohibition: Zero clauses where an action is justified by "because/since I/he/she had known/thought/realized/seen/felt/understood/suspected/recognized/guessed/sensed." Put the knowledge before the action or let the action stand.""",

    10: """Cap 10 — Verdict constructions (compact narrator judgment).
Function: The model compresses judgment into elegant but fingerprinted verdict phrasing.
Prohibition: Zero narration or interior thought using "[noun] too [adjective] for [noun]," "too [adjective] to [verb]" as narrator verdict, or "a specific/particular/certain/peculiar/distinct kind of [bare adjective]." Characters may use such phrasing in dialogue if natural.""",

    14: """Cap 14 — Negation-as-action (refusal as structure).
Function: The model turns not-doing into the action of a sentence or paragraph.
Prohibition: Avoid building sentences or paragraphs around what the POV character does not do: "I did not look," "He did not turn," "She did not answer" as the main architecture. State the chosen action instead. Ordinary negation is permitted when needed for meaning.""",

    16: """Cap 16 — Polysyndetic run-on (repeated "and" chaining).
Function: The model imitates Hemingway-like breathless accumulation by chaining clauses with repeated "and."
Prohibition: Zero narration sentences joining four or more clauses with coordinating "and." Dialogue is exempt if character voice requires it.""",

    18: """Cap 18 — Tautological loop (subject doing what subjects do).
Function: The model creates literary-sounding loops whose object merely restates the subject's category.
Prohibition: Zero instances of "[Subject] did/does the thing [subjects] do," including variants such as "her face did the thing faces do" or "wolves did the thing wolves do." """,

    19: """Cap 19 — Anaphoric escalation (repeated sentence openers).
Function: Three-fold repeated openers create a rhetorical-anaphora fingerprint.
Prohibition: Zero instances of three or more sentences within a five-sentence window sharing the same opening one or two words. Two-fold parallel structure is permitted. Three-fold escalation is prohibited.""",
}


def characterize_cap_failure(
    cap_id: int,
    flagged_passages: list[dict],
    n_drafts: int,
    client,
    model: str,
    outline_text: Optional[str] = None,
    author_construction: Optional[str] = None,
) -> dict:
    """Single LLM call: characterize why a cap is failing in this batch.

    Returns:
      {
        "cap_id": int,
        "pattern": str,
        "trigger": str,
        "lever": str,
        "concrete_fix": str,
        "raw_response": str,
        "error": str | None,
      }
    """
    if not flagged_passages:
        return {
            "cap_id": cap_id,
            "pattern": "(no flagged passages available)",
            "trigger": "",
            "lever": "",
            "concrete_fix": "",
            "raw_response": "",
            "error": "no flagged passages",
        }

    cap_def = CAP_DEFINITIONS.get(
        cap_id,
        f"Cap {cap_id} — (definition not available in failure_diagnostic_v1; see normalize_outline_v10H.txt)",
    )

    passages_block = "\n".join(
        f"  [draft {fp['draft_id']}] {fp['excerpt'].strip()!r}"
        for fp in flagged_passages[:20]  # cap at 20 to keep prompt tractable
    )

    outline_section = ""
    if outline_text:
        # Trim outline to first 4000 chars to keep prompt size manageable.
        snip = outline_text[:4000]
        outline_section = f"\n\nCHAPTER OUTLINE (first 4000 chars):\n{snip}\n"

    author_section = ""
    if author_construction:
        author_section = f"\n\nAUTHOR_CONSTRUCTION BLOCK:\n{author_construction}\n"

    prompt = DIAGNOSIS_PROMPT_TEMPLATE.format(
        cap_id=cap_id,
        n_drafts=n_drafts,
        cap_definition=cap_def,
        flagged_passages=passages_block,
        outline_section=outline_section,
        author_section=author_section,
    )

    try:
        resp = client.messages.create(
            model=model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "\n".join(b.text for b in resp.content if getattr(b, "text", None))
    except Exception as e:
        return {
            "cap_id": cap_id,
            "pattern": "",
            "trigger": "",
            "lever": "",
            "concrete_fix": "",
            "raw_response": "",
            "error": f"LLM call failed: {e}",
        }

    parsed = _parse_diagnosis_response(text)
    parsed["cap_id"] = cap_id
    parsed["raw_response"] = text
    parsed["error"] = None
    return parsed


def _parse_diagnosis_response(text: str) -> dict:
    """Pull PATTERN / TRIGGER / LEVER / CONCRETE FIX sections from the LLM reply."""
    out = {"pattern": "", "trigger": "", "lever": "", "concrete_fix": ""}
    current = None
    buf = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("PATTERN:"):
            _flush(out, current, buf)
            current = "pattern"
            buf = [stripped[len("PATTERN:"):].strip()]
        elif stripped.startswith("TRIGGER:"):
            _flush(out, current, buf)
            current = "trigger"
            buf = [stripped[len("TRIGGER:"):].strip()]
        elif stripped.startswith("LEVER:"):
            _flush(out, current, buf)
            current = "lever"
            buf = [stripped[len("LEVER:"):].strip()]
        elif stripped.startswith("CONCRETE FIX:"):
            _flush(out, current, buf)
            current = "concrete_fix"
            buf = [stripped[len("CONCRETE FIX:"):].strip()]
        else:
            buf.append(line)
    _flush(out, current, buf)
    return out


def _flush(out: dict, current: Optional[str], buf: list[str]) -> None:
    if current is None:
        return
    out[current] = "\n".join(buf).strip()


# ============================================================================
# Top-level entry point
# ============================================================================
def diagnose_batch(
    rejected_drafts: list[dict],
    outline_text: Optional[str] = None,
    author_construction: Optional[str] = None,
    client=None,
    model: Optional[str] = None,
    max_caps_to_characterize: int = 5,
) -> dict:
    """Run the full diagnostic on a failed batch.

    If client+model provided, runs Layer 2 LLM characterization on the top
    `max_caps_to_characterize` caps by breadth. Otherwise returns Layer 1
    aggregation only.
    """
    aggregation = aggregate_failures(rejected_drafts)
    passages_by_cap = collect_flagged_passages(rejected_drafts)

    characterizations = {}
    if client is not None and model is not None:
        # Run on top N caps by breadth (how many drafts they trip).
        top_caps = aggregation["cap_order_by_breadth"][:max_caps_to_characterize]
        for cap_id in top_caps:
            characterizations[cap_id] = characterize_cap_failure(
                cap_id=cap_id,
                flagged_passages=passages_by_cap.get(cap_id, []),
                n_drafts=aggregation["n_drafts"],
                client=client,
                model=model,
                outline_text=outline_text,
                author_construction=author_construction,
            )

    return {
        "aggregation":        aggregation,
        "passages_by_cap":    passages_by_cap,
        "characterizations":  characterizations,
    }


# ============================================================================
# Log formatting
# ============================================================================
def format_diagnosis_for_log(diagnosis: dict) -> str:
    """Render the diagnosis as plain text for the Streamlit log."""
    agg = diagnosis["aggregation"]
    lines = []
    lines.append(f"=== Failure diagnosis — {agg['n_drafts']} draft(s) rejected ===")
    lines.append("")

    if not agg["per_cap"]:
        lines.append("(no cap violations recorded)")
        return "\n".join(lines)

    lines.append("Cap incidence (ranked by breadth):")
    for cap_id in agg["cap_order_by_breadth"]:
        e = agg["per_cap"][cap_id]
        lines.append(
            f"  Cap {cap_id} — {e['rule_name']}: tripped in {e['drafts_tripped']}/{agg['n_drafts']} drafts, "
            f"{e['total_hits']} total hits, max {e['max_hits_single_draft']} in a single draft"
        )
    lines.append("")

    chars = diagnosis.get("characterizations") or {}
    if chars:
        lines.append("LLM characterizations:")
        for cap_id, c in chars.items():
            lines.append(f"")
            lines.append(f"--- Cap {cap_id} ---")
            if c.get("error"):
                lines.append(f"  ERROR: {c['error']}")
                continue
            lines.append(f"  PATTERN: {c.get('pattern','')}")
            lines.append(f"  TRIGGER: {c.get('trigger','')}")
            lines.append(f"  LEVER:   {c.get('lever','')}")
            lines.append(f"  FIX:     {c.get('concrete_fix','')}")
    else:
        lines.append("(LLM characterization not run — no client supplied)")

    return "\n".join(lines)
