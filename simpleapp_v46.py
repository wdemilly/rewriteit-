"""simpleapp_v46.py

Production Streamlit app implementing a commercial-fiction drafting
pipeline. v46 restores the v36.32 composition model.

v46 change vs v45 (Step 2-3 removed):

  PROBLEM v44/v45 had: Step 2-3 ran an LLM call that characterized a
  named author across 13 abstract dimensions ("Subject-first dominates",
  "Mid-length sentences of 12-20 words form the default", etc.) and
  injected that characterization into Step 5 as an AUTHOR_CONSTRUCTION
  block. The drafter received the outline plus a printed checklist of
  voice properties and tried to compose sentences against the checklist.
  This produced per-sentence compliance-loop prose: structurally on-list
  but lacking voice, ear, or rhythm. Empirically, v44/v45 drafts scored
  ~59% Human on Originality vs v36.32's 82-99%.

  v46 REMOVES Step 2-3 entirely. The pipeline becomes:

    outline + CSV-selected prompt → drafter → chapter

  No author characterization is generated. No AUTHOR_CONSTRUCTION block
  enters Step 5 input. The drafter sees the outline (which names the
  voice author internally on its 'Selected author:' line) plus the
  operator-selected CSV prompt, and writes from training knowledge of
  that named author. This is the v36.32 mechanism: implicit voice
  anchoring through training memory of the named author, with no
  intermediate explicit-characterization step that the drafter would
  otherwise compliance-loop against.

  Cost reduction: Step 2-3 cost ~$0.56 per chapter on Opus 4.7. v46
  saves that cost and removes the duplicate abstraction (the outline
  itself still carries an internal Author Construction block, which is
  a separate normalize-prompt cleanup if desired).

  Files no longer referenced: author_construction_prompt_v1.txt.
  Files still required: prompts.csv.

v45 changes (carried forward): Step 5 drafting prompt selected by
operator from prompts.csv via sidebar dropdown.

v44 changes (carried forward): last completed run cached in
st.session_state["last_result"] and re-rendered on every script
execution; two-column body layout matching v36.

v43 changes (carried forward): APP_VERSION constant displayed in UI;
APP_VERSION and outline label embedded in run dirs + download filenames.
v42 changes (carried forward): configurable Step 6 gate (STRICT /
PRAGMATIC / USER_PROPOSED); auto failure diagnostic on outer-batch
failures.
v41 changes (carried forward): incremental run persistence; crash
capture; Step 7 evaluator guarded against KeyError.

ARCHITECTURE (per architecture_record_v1.txt):

ARCHITECTURE (v46, simplified):

  Step 1.  Operator uploads the chapter outline.
  Step 2.  (v44/v45: LLM call selected authors and characterized the
           primary author's habits across 13 dimensions, outputting an
           AUTHOR_CONSTRUCTION block. v46 removes this entirely. The
           outline itself names the voice author on its 'Selected
           author:' line; the drafter reads that and writes in the
           named author's voice from training knowledge.)
  Step 3.  (Removed in v46. See Step 2.)
  Step 4.  (v44 and earlier: five architectural design rules baked into
           the drafting prompt. v45 removes those — drafting prompt is
           now whatever the operator selects from prompts.csv.)
  Step 5.  Generate X drafts (default X=3) of the chapter from the
           outline + operator-selected prompt. No AUTHOR_CONSTRUCTION
           block. Same temperature; variation from sampling.
  Step 6.  Rule-compliance gate. Each draft is verified by
           rule_verifier_v1.verify_draft(). Failures dumped. If fewer
           than MIN_SURVIVORS_STEP_6 survive, regenerate the batch
           (up to MAX_BATCHES_STEP_6 attempts).
  Step 7.  Score survivors. Two scores per survivor:
             quality:     0-10 from quality_evaluator_v2
             ai_estimate: 0-100 from local_scorer (predicted turbo)
  Step 8.  Threshold gate. Survivor passes if:
             quality verdict == ACCEPTABLE
             quality score >= QUALITY_MIN
             ai_estimate    >= AI_ESTIMATE_MIN
           If no survivor passes, regenerate the batch (up to
           MAX_BATCHES_STEP_8 outer batches).
  Step 9.  Choose the winner: highest combined score across passing
           survivors. Tie-break by ai_estimate.

DEFAULTS (committed in architecture_record_v2.txt):
  N_DRAFTS              = 3
  QUALITY_MIN           = 6
  AI_ESTIMATE_MIN       = 80
  MIN_SURVIVORS_STEP_6  = 2
  MAX_BATCHES_STEP_6    = 3
  MAX_BATCHES_STEP_8    = 3

DEPENDENCIES (must be co-located in the same directory):
  prompts.csv                 (Step 5 prompts; operator selectable)
  rule_verifier_v1.py
  quality_evaluator_v2.py
  local_scorer.py            (from v32)
  band_classifier.py         (from v32)
  extended_band_features.py  (from v32)
  calibration.json           (from v32)

USAGE:
  streamlit run simpleapp_v46.py
"""
from __future__ import annotations
import os
import re
import sys
import io
import json
import csv
import time
import uuid
import traceback
from pathlib import Path
from typing import Optional

import streamlit as st

# Make co-located modules importable.
APP_DIR = Path(__file__).parent.resolve()
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import anthropic
from rule_verifier_v1 import (
    verify_draft, scan_for_evaluator, format_violation_summary,
)
from quality_evaluator_v2 import evaluate_drafts_with_anthropic
from local_scorer import score_text
import band_classifier
from run_state_v1 import RunState, capture_crash
from gate_policy_v1 import (
    passes_policy, build_policy_from_ui, format_policy, PRESETS,
)
from failure_diagnostic_v1 import diagnose_batch, format_diagnosis_for_log


# ============================================================================
# App version — visible in UI and embedded in run dir + output filenames.
# Pattern matches v36_32.py (APP_VERSION = "v36.32" at line 286).
# ============================================================================
APP_VERSION = "v46"


# ============================================================================
# Defaults (overridable in the sidebar)
# ============================================================================
DEFAULT_N_DRAFTS             = 3
DEFAULT_QUALITY_MIN          = 6
DEFAULT_AI_ESTIMATE_MIN      = 80
DEFAULT_MIN_SURVIVORS_STEP_6 = 2
DEFAULT_MAX_BATCHES_STEP_6   = 3
DEFAULT_MAX_BATCHES_STEP_8   = 3

MAX_DRAFT_TOKENS = 16000

DEFAULT_MODEL = "claude-opus-4-7"
AVAILABLE_MODELS = [
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
]


# ============================================================================
# Filename + outline-label helpers (ported verbatim from simpleapp_v36_32.py
# at lines 3401-3445, with one added function for the uploaded-file case)
# ============================================================================
def sanitize_filename_part(value: str, max_len: int = 80) -> str:
    """Return a compact Windows-safe filename component."""
    value = str(value or "").strip()
    value = re.sub(r"[\r\n\t]+", " ", value)
    value = re.sub(r"[^A-Za-z0-9._() -]+", "", value)
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"_+", "_", value).strip("._- ")
    if not value:
        value = "no_heading"
    return value[:max_len].strip("._- ") or "no_heading"


def extract_outline_label_from_filename(filename: Optional[str]) -> str:
    """Derive a short label from an uploaded outline's filename. Strips
    the extension and sanitizes. e.g. 'Tide_Walkers_Ch1_Outline_v22.txt'
    -> 'Tide_Walkers_Ch1_Outline_v22'. Returns '' if no filename."""
    if not filename:
        return ""
    stem = Path(filename).stem
    return sanitize_filename_part(stem, max_len=36)


def extract_outline_label_from_text(outline_text: str) -> str:
    """Return a short outline label for filenames (v36 pattern).

    Numbered chapter headings collapse to CH_<number>, so 'Chapter 3:
    Departure' and 'CH 3 - Departure' both become 'CH_3' after sanitizing.
    If no chapter number is found, fall back to a compact sanitized
    heading."""
    if not outline_text:
        return "no_outline_heading"

    raw_lines = [ln.strip() for ln in str(outline_text).splitlines()]
    lines = [ln.strip().strip("#*-").strip() for ln in raw_lines if ln.strip()]

    numbered_chapter_patterns = [
        r"\bchapter\s*(\d{1,3})\b",
        r"\bch\.?\s*(\d{1,3})\b",
        r"^\s*(\d{1,3})\s*[.)\-:–—]\s+",
    ]
    for line in lines:
        if len(line) > 120:
            continue
        for pat in numbered_chapter_patterns:
            m = re.search(pat, line, flags=re.IGNORECASE)
            if m:
                return sanitize_filename_part(f"CH {m.group(1)}", 16)

    heading_skip = r"\b(words?|target|global drafting controls|drafting controls|outline)\b"
    for line in lines:
        if 3 <= len(line) <= 90 and not re.search(heading_skip, line, re.IGNORECASE):
            return sanitize_filename_part(line, 36)

    return "outline_heading"


def derive_outline_label(upload_filename: Optional[str], outline_text: str) -> str:
    """Prefer the uploaded filename when available; otherwise extract
    from the outline text. Returns a sanitized, filename-safe label."""
    label = extract_outline_label_from_filename(upload_filename)
    if label:
        return label
    return extract_outline_label_from_text(outline_text)


# ============================================================================
# Prompt loading
# ============================================================================
def load_prompt(filename: str) -> str:
    """Read a prompt file from the app directory; strip leading comments."""
    path = APP_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Prompt file not found: {path}")
    raw = path.read_text(encoding="utf-8")
    # Strip leading lines that start with "#"; keep blank lines and content.
    lines = raw.splitlines()
    start = 0
    while start < len(lines) and (lines[start].startswith("#") or not lines[start].strip()):
        start += 1
    return "\n".join(lines[start:]).strip()


# ============================================================================
# v45: prompts.csv selection
#
# Restores the pre-v40 mechanism where the operator chooses the Step 5
# drafting instruction from a versioned CSV at run time, instead of the
# instruction being hardcoded in draft_chapter_prompt_v1.txt.
#
# CSV schema: id,category,text
#   id        — integer string, unique
#   category  — free-form bucket label (identity_lock, sentence_construction,
#               sentence_check, voice_source, etc.)
#   text      — the prompt body. Plain prose. May be a single sentence.
#               Should NOT contain {outline} placeholder — the assembler
#               appends the outline section below.
#
# To add a new prompt, edit prompts.csv. No code change required.
# ============================================================================
PROMPTS_CSV_FILENAME = "prompts.csv"


def load_prompts_csv() -> list[dict]:
    """Read prompts.csv from APP_DIR and return a list of
    {id, category, text} dicts in file order. Empty rows are skipped.
    Raises FileNotFoundError if the file is missing."""
    path = APP_DIR / PROMPTS_CSV_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Prompts CSV not found: {path}")
    rows: list[dict] = []
    # utf-8-sig handles the BOM that Excel often writes.
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            pid = (r.get("id") or "").strip()
            text = (r.get("text") or "").strip()
            if not pid or not text:
                continue
            rows.append({
                "id":       pid,
                "category": (r.get("category") or "").strip(),
                "text":     text,
            })
    return rows


def lookup_prompt(prompts: list[dict], prompt_id: str) -> dict:
    """Return the row whose id == prompt_id. Raises KeyError if absent."""
    for r in prompts:
        if r["id"] == str(prompt_id):
            return r
    raise KeyError(f"Prompt id {prompt_id!r} not found in {PROMPTS_CSV_FILENAME}")


def build_draft_prompt(
    csv_prompt_text: str,
    outline_text: str,
) -> str:
    """Assemble the full Step 5 user-message body.

    Structure (v46 — outline only, no AUTHOR_CONSTRUCTION section):

        <csv prompt text>

        === OUTLINE ===

        <outline text>

    The CSV prompt body is the only operator-controlled drafting
    instruction. v46 deliberately adds no other framing, design rules,
    compliance language, self-suppression directives, or author
    characterization. The outline carries the voice anchor on its
    'Selected author:' line; the drafter writes in that author's voice
    from training memory.
    """
    return (
        f"{csv_prompt_text.strip()}\n\n"
        f"=== OUTLINE ===\n\n{outline_text.strip()}\n"
    )


# ============================================================================
# Step 2-3 — REMOVED in v46.
#
# v44/v45 ran an LLM call here that characterized a named author across
# 13 abstract dimensions and emitted an AUTHOR_CONSTRUCTION block, which
# was then injected into Step 5's input. That step is gone in v46. The
# outline already names the voice author on its 'Selected author:' line;
# the drafter writes in that author's voice from training memory. No
# explicit characterization step. No checklist for the drafter to
# compliance-loop against.
# ============================================================================


# ============================================================================
# Step 5 — Generate one draft
# ============================================================================
def generate_draft(
    client: anthropic.Anthropic,
    model: str,
    outline_text: str,
    csv_prompt_text: str,
    run_id: str,
    log,
) -> dict:
    """Single drafting call. Returns dict with run_id, text, usage.

    v46: drafter receives outline + csv_prompt_text only. No
    AUTHOR_CONSTRUCTION block.
    """
    prompt = build_draft_prompt(csv_prompt_text, outline_text)
    log(f"Step 5 — generating draft {run_id} (model: {model})")
    resp = client.messages.create(
        model=model,
        max_tokens=MAX_DRAFT_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "\n".join(b.text for b in resp.content if getattr(b, "text", None))
    return {
        "run_id": run_id,
        "text": text.strip(),
        "input_tokens": resp.usage.input_tokens,
        "output_tokens": resp.usage.output_tokens,
    }


def generate_draft_batch(
    client, model, outline_text, csv_prompt_text,
    n_drafts, log,
    state: Optional[RunState] = None,
) -> list[dict]:
    """Generate n_drafts drafts and return them as a list. If state is
    provided, each draft is persisted to disk the moment it's produced
    so a downstream crash doesn't lose work already paid for."""
    drafts = []
    for i in range(n_drafts):
        run_id = uuid.uuid4().hex[:8]
        try:
            d = generate_draft(
                client, model, outline_text,
                csv_prompt_text, run_id, log,
            )
            drafts.append(d)
            if state is not None:
                state.write_draft(d)
            log(f"  Draft {run_id}: {len(d['text'].split())} words")
        except Exception as e:
            log(f"  ERROR generating draft {run_id}: {e}")
            if state is not None:
                state.write_draft_error(run_id, f"{type(e).__name__}: {e}")
    return drafts


# ============================================================================
# Step 6 — Rule-compliance gate
# ============================================================================
def filter_compliant_drafts(
    drafts: list[dict],
    log,
    state: Optional[RunState] = None,
    gate_policy: Optional[dict] = None,
) -> tuple[list[dict], list[dict]]:
    """Run verify_draft on each. Return (survivors, rejected).

    If gate_policy is None, falls back to the verifier's own binary
    verify_result['pass'] field (v40/v41 behavior). If provided, the
    policy is applied to verify_result['violations'] via passes_policy.
    The verifier itself is unchanged — only the pass/fail DECISION
    layered on top of the FACTS changes."""
    survivors = []
    rejected = []
    for d in drafts:
        result = verify_draft(d["text"])
        d["verify_result"] = result
        if state is not None:
            state.write_verify_result(d["run_id"], result)

        if gate_policy is None:
            passed = bool(result.get("pass"))
            reason = format_violation_summary(result)
        else:
            passed, reason = passes_policy(result, gate_policy)
            # Also store the verifier's own raw reason for diagnostic context.
            verifier_reason = format_violation_summary(result)
            reason = f"{reason}  |  verifier: {verifier_reason}"

        if passed:
            survivors.append(d)
            log(f"  {d['run_id']}: {reason}")
        else:
            rejected.append(d)
            log(f"  {d['run_id']}: {reason}")
    return survivors, rejected


# ============================================================================
# Step 7 — Score survivors (quality + AI estimate)
# ============================================================================
def score_survivors(
    client, model, survivors, outline_text, log,
    state: Optional[RunState] = None,
) -> dict:
    """Compute quality scores (one shared LLM call) and per-draft AI estimates.

    Robust against:
      - quality evaluator LLM call raising (network, model error, parse error);
      - quality evaluator returning a dict missing some run_ids;
      - local scorer raising per-draft.

    On any of these, the survivor is given safe default scores (verdict=ERROR,
    score=0, ai_estimate=0) so downstream code (filter_passing, pick_winner)
    runs without KeyError. v40 crashed the entire run on any of these paths,
    losing every paid-for draft.
    """
    # Build the scan_by_run_id dict the evaluator expects.
    scan_by_run_id = {}
    for d in survivors:
        scan_by_run_id[d["run_id"]] = scan_for_evaluator(d["text"])

    log(f"Step 7 — quality evaluator on {len(survivors)} survivor(s)")
    eval_result = {"quality_by_run_id": {}, "quality_score_by_run_id": {}}
    try:
        eval_result = evaluate_drafts_with_anthropic(
            client=client,
            model=model,
            drafts=survivors,
            outline_text=outline_text,
            scan_by_run_id=scan_by_run_id,
        )
    except Exception as e:
        log(f"  Quality evaluator FAILED: {type(e).__name__}: {e}")
        log(f"  Continuing with score=0 for all survivors; AI estimate still computed.")

    log(f"Step 7 — AI estimate via local_scorer")
    # The band classifier inside local_scorer needs a Claude client. Build
    # one with the operator's API key (same as the drafting client) and pass
    # it in so score_text doesn't try to create an anonymous default.
    classifier_config = band_classifier.BandClassifierConfig()
    classifier = band_classifier.BandClassifier(
        config=classifier_config,
        client=client,
    )
    for d in survivors:
        rid = d["run_id"]
        try:
            predicted, details = score_text(d["text"], classifier=classifier)
            d["ai_estimate"] = float(predicted)
            d["ai_estimate_details"] = details
        except Exception as e:
            d["ai_estimate"] = 0.0
            d["ai_estimate_details"] = {"error": str(e)}
            log(f"  AI estimate ERROR for {rid}: {e}")

        # Defensive lookup — eval_result may be missing this run_id.
        qbyrid = eval_result.get("quality_by_run_id", {})
        qrec = qbyrid.get(rid, {"verdict": "ERROR", "reason": "evaluator missing rid"})
        d["quality_verdict"] = qrec.get("verdict", "ERROR")
        d["quality_reason"]  = qrec.get("reason", "")
        d["quality_score"]   = eval_result.get("quality_score_by_run_id", {}).get(rid, 0)

        if state is not None:
            state.write_scores(
                run_id=rid,
                quality_verdict=d["quality_verdict"],
                quality_reason=d["quality_reason"],
                quality_score=d["quality_score"],
                ai_estimate=d["ai_estimate"],
                ai_estimate_details=d["ai_estimate_details"],
            )

        log(f"  {rid}: quality={d['quality_verdict']} "
            f"score={d['quality_score']}, ai_estimate={d['ai_estimate']:.1f}")
    return eval_result


# ============================================================================
# Step 8 — Threshold gate
# ============================================================================
def filter_passing(
    survivors: list[dict],
    quality_min: int,
    ai_estimate_min: float,
) -> list[dict]:
    """Drafts passing both thresholds."""
    return [
        d for d in survivors
        if d.get("quality_verdict") == "ACCEPTABLE"
        and d.get("quality_score", 0) >= quality_min
        and d.get("ai_estimate", 0) >= ai_estimate_min
    ]


# ============================================================================
# Step 9 — Choose the winner
# ============================================================================
def pick_winner(passing: list[dict]) -> Optional[dict]:
    """Highest (quality + ai_estimate/10). Tie-break by ai_estimate."""
    if not passing:
        return None
    return max(
        passing,
        key=lambda d: (d["quality_score"] + d["ai_estimate"] / 10.0, d["ai_estimate"]),
    )


# ============================================================================
# Orchestration
# ============================================================================
def run_pipeline(
    client: anthropic.Anthropic,
    model: str,
    outline_text: str,
    csv_prompt_id: str,
    csv_prompt_text: str,
    n_drafts: int,
    quality_min: int,
    ai_estimate_min: float,
    min_survivors_step_6: int,
    max_batches_step_6: int,
    max_batches_step_8: int,
    log,
    gate_policy: Optional[dict] = None,
    run_diagnostic: bool = True,
    diagnostic_max_caps: int = 5,
    outline_label: Optional[str] = None,
) -> dict:
    """Run all 9 steps. Returns a result dict. Persists every draft to
    ./runs/<APP_VERSION>_<outline_label>_<timestamp>/ as it is generated;
    a crash leaves work on disk.

    gate_policy: dict per gate_policy_v1 schema; None = v40/v41 binary gate.
    run_diagnostic: when True, after an outer batch fails to produce enough
       survivors, runs failure_diagnostic_v1.diagnose_batch on the rejected
       drafts. Layer 2 LLM characterization runs on the top diagnostic_max_caps
       caps by breadth. Diagnoses persist to ./runs/<run>/diagnoses/.
    outline_label: filename-safe label embedded in the run dir. Caller
       computes via derive_outline_label() so the same label appears in
       sidebar logs and downstream file names."""
    # Initialize persistent run state BEFORE any LLM call so even a
    # failure inside step 2-3 leaves a manifest with the outline saved.
    state = RunState.create(
        outline_text=outline_text,
        config={
            "n_drafts":              n_drafts,
            "quality_min":           quality_min,
            "ai_estimate_min":       ai_estimate_min,
            "min_survivors_step_6":  min_survivors_step_6,
            "max_batches_step_6":    max_batches_step_6,
            "max_batches_step_8":    max_batches_step_8,
            "gate_policy":           gate_policy,
            "run_diagnostic":        run_diagnostic,
            "outline_label":         outline_label,
            "csv_prompt_id":         csv_prompt_id,
            "csv_prompt_text":       csv_prompt_text,
        },
        model=model,
        label=outline_label,
        app_version=APP_VERSION,
    )
    log(f"Run dir: {state.run_dir}")
    log(f"Step 5 prompt: id={csv_prompt_id} ({len(csv_prompt_text)} chars)")
    if gate_policy is not None:
        log(f"Gate policy: {format_policy(gate_policy)}")
    else:
        log("Gate policy: STRICT (binary verify_result[pass])")

    try:
        # Step 2-3 — REMOVED in v46. See module docstring.

        all_batches = []
        final_winner = None
        final_eval = None
        final_passing = []
        final_survivors = []

        for outer in range(max_batches_step_8):
            log(f"--- Outer batch {outer + 1}/{max_batches_step_8} ---")
            # Step 5-6 inner loop: keep generating until we have enough survivors.
            survivors = []
            outer_rejected = []   # all rejected drafts across inner attempts
            inner_batches = []
            for inner in range(max_batches_step_6):
                log(f"Step 5 — generating draft batch ({n_drafts} drafts), "
                    f"inner attempt {inner + 1}/{max_batches_step_6}")
                drafts = generate_draft_batch(
                    client, model, outline_text,
                    csv_prompt_text, n_drafts, log,
                    state=state,
                )
                log(f"Step 6 — rule-compliance gate on {len(drafts)} draft(s)")
                batch_survivors, batch_rejected = filter_compliant_drafts(
                    drafts, log, state=state, gate_policy=gate_policy,
                )
                inner_batches.append({
                    "inner_attempt":  inner + 1,
                    "drafts_generated": len(drafts),
                    "survivors":     len(batch_survivors),
                    "rejected":      len(batch_rejected),
                    "drafts":        drafts,
                })
                state.record_inner_batch(
                    outer_attempt=outer + 1,
                    inner_attempt=inner + 1,
                    n_generated=len(drafts),
                    n_survivors=len(batch_survivors),
                    n_rejected=len(batch_rejected),
                    draft_run_ids=[d["run_id"] for d in drafts],
                )
                survivors.extend(batch_survivors)
                outer_rejected.extend(batch_rejected)
                if len(survivors) >= min_survivors_step_6:
                    log(f"  Step 6 satisfied with {len(survivors)} survivor(s)")
                    break
                else:
                    log(f"  Only {len(survivors)} survivor(s); need {min_survivors_step_6}, "
                        f"regenerating")

            all_batches.append({"outer_attempt": outer + 1, "inner_batches": inner_batches})

            if len(survivors) < min_survivors_step_6:
                log(f"Outer batch {outer + 1}: step 6 exhausted "
                    f"({max_batches_step_6} attempts), got {len(survivors)} survivor(s). "
                    f"Skipping to next outer batch.")
                # Run failure diagnostic on accumulated rejected drafts.
                if run_diagnostic and outer_rejected:
                    log(f"  Running failure diagnostic on {len(outer_rejected)} rejected draft(s)...")
                    try:
                        diagnosis = diagnose_batch(
                            rejected_drafts=outer_rejected,
                            outline_text=outline_text,
                            author_construction="",  # v46: no AUTHOR_CONSTRUCTION generated
                            client=client,
                            model=model,
                            max_caps_to_characterize=diagnostic_max_caps,
                        )
                        state.write_diagnosis(outer + 1, diagnosis)
                        log(format_diagnosis_for_log(diagnosis))
                    except Exception as diag_exc:
                        log(f"  Diagnostic failed: {type(diag_exc).__name__}: {diag_exc}")
                continue

            # Step 7
            eval_result = score_survivors(client, model, survivors, outline_text, log, state=state)

            # Step 8
            passing = filter_passing(survivors, quality_min, ai_estimate_min)
            log(f"Step 8 — {len(passing)} draft(s) passed thresholds "
                f"(quality >= {quality_min}, ai_estimate >= {ai_estimate_min})")

            if passing:
                # Step 9
                winner = pick_winner(passing)
                log(f"Step 9 — WINNER: {winner['run_id']} "
                    f"(quality {winner['quality_score']}, "
                    f"ai_estimate {winner['ai_estimate']:.1f})")
                state.write_winner(winner)
                final_winner = winner
                final_eval = eval_result
                final_passing = passing
                final_survivors = survivors
                break
            else:
                log(f"Outer batch {outer + 1}: no draft passed thresholds. "
                    f"Regenerating.")
                final_eval = eval_result
                final_survivors = survivors

        state.finalize()
        return {
            "winner":              final_winner,
            "all_batches":         all_batches,
            "final_eval":          final_eval,
            "final_passing":       final_passing,
            "final_survivors":     final_survivors,
            "run_state":           state,
        }
    except BaseException as exc:
        # Capture the traceback to disk before the exception bubbles up to
        # the Streamlit error handler. This is what makes a v40-style crash
        # recoverable: even if the UI resets, the manifest + crash.txt
        # remain in state.run_dir.
        capture_crash(state, exc)
        log(f"PIPELINE CRASHED — traceback saved to {state.run_dir}/crash.txt")
        log(f"Drafts already generated remain in {state.run_dir}/drafts/")
        raise


# ============================================================================
# Streamlit UI
# ============================================================================
def _render_recent_runs_panel() -> None:
    """Show a panel listing recent runs with download links. This is the
    primary recovery mechanism if the pipeline crashes mid-run: even with a
    fresh-looking UI, prior runs (and their drafts) remain accessible here."""
    recent = RunState.list_recent_runs(app_dir=APP_DIR, n=10)
    if not recent:
        return
    with st.expander(f"Recent runs ({len(recent)}) — download prior or crashed runs"):
        for m in recent:
            run_dir = Path(m.get("run_dir", ""))
            status = m.get("status", "?")
            started = m.get("started_at", "?")
            n_drafts_on_disk = len(m.get("drafts", {}))
            winner_rid = m.get("winner_run_id")
            label = f"{started}  ·  status: {status}  ·  drafts on disk: {n_drafts_on_disk}"
            if winner_rid:
                label += f"  ·  winner: {winner_rid}"
            if status == "crashed":
                label = "⚠ " + label
            st.markdown(f"**{label}**")
            st.code(str(run_dir), language="text")

            # Download buttons
            cols = st.columns(3)
            # Zip archive of the whole run directory
            if run_dir.exists():
                try:
                    state = RunState.load(run_dir)
                    zip_path = state.zip_archive()
                    with open(zip_path, "rb") as fh:
                        cols[0].download_button(
                            "Download full run (.zip)",
                            data=fh.read(),
                            file_name=zip_path.name,
                            mime="application/zip",
                            key=f"zip_{run_dir.name}",
                        )
                    winner_path = state.winner_path()
                    if winner_path and winner_path.exists():
                        cols[1].download_button(
                            "Download winning chapter (.txt)",
                            data=winner_path.read_text(encoding="utf-8"),
                            file_name=winner_path.name,
                            mime="text/plain",
                            key=f"win_{run_dir.name}",
                        )
                    crash_path = run_dir / "crash.txt"
                    if crash_path.exists():
                        cols[2].download_button(
                            "Download crash log (.txt)",
                            data=crash_path.read_text(encoding="utf-8"),
                            file_name=f"crash_{run_dir.name}.txt",
                            mime="text/plain",
                            key=f"crash_{run_dir.name}",
                        )
                except Exception as e:
                    st.warning(f"Could not load run {run_dir.name}: {e}")
            st.write("")


def _render_last_result_panel() -> None:
    """Render the most recently completed pipeline run.

    Reads from st.session_state["last_result"] first (set when a pipeline
    finishes in THIS streamlit session). If empty, falls back to the most
    recent terminal-state run on disk — so a script kill (e.g., Streamlit
    Cloud reaping the process when its websocket idles) doesn't hide the
    prior work from the user when a new script execution starts.

    This is the v44 fix for the "fresh dashboard after run" problem:
    Streamlit reruns main() from the top on every interaction, AND if
    the hosting environment kills the script entirely, a new fresh
    process starts. Without this, both cases produce a blank result
    column. With this, the last completed run stays visible across
    both kinds of reset."""
    result = st.session_state.get("last_result")
    if not result:
        # Disk fallback — last terminal-state run from ./runs/
        result = _load_last_terminal_result_from_disk()

    if not result:
        st.subheader("Last completed run")
        st.caption("No run completed yet. After you click **Run pipeline** "
                   "on the left, results will appear here and persist across "
                   "page refreshes — including across process restarts.")
        return

    st.subheader("Last completed run")
    elapsed = result.get("elapsed", 0)
    source = result.get("_source", "this session")
    st.caption(
        f"Outline label: `{result.get('outline_label', '?')}` "
        f"· source: {source}"
        + (f" · {elapsed:.1f}s" if elapsed else "")
    )

    winner = result.get("winner")
    if winner is None:
        st.error(
            f"No draft passed the thresholds within "
            f"{result.get('max_batches_step_8', '?')} outer batch(es). "
            f"See diagnostics below for what was tried."
        )
    else:
        st.success(
            f"Winner: {winner['run_id']} — "
            f"quality {winner.get('quality_score', '?')}, "
            f"ai_estimate {winner.get('ai_estimate', 0):.1f}"
        )
        outline_label = result.get("outline_label", "")
        winner_filename_stem = "_".join(
            p for p in [APP_VERSION, outline_label, "chapter", winner["run_id"]] if p
        )
        st.download_button(
            "Download winning chapter (.txt)",
            data=winner["text"],
            file_name=f"{winner_filename_stem}.txt",
            mime="text/plain",
            key="last_winner_txt",
        )
        run_dir = result.get("run_dir")
        if run_dir and Path(run_dir).exists():
            try:
                state = RunState.load(Path(run_dir))
                zip_path = state.zip_archive()
                with open(zip_path, "rb") as fh:
                    st.download_button(
                        "Download full run (.zip)",
                        data=fh.read(),
                        file_name=zip_path.name,
                        mime="application/zip",
                        key="last_winner_zip",
                    )
                st.caption(f"Run directory: `{run_dir}`")
            except Exception as e:
                st.warning(f"Could not build run archive: {e}")

        with st.expander("Winning chapter prose"):
            st.text_area(
                "Chapter prose",
                value=winner["text"],
                height=400,
                key="last_winner_text",
            )

    if result.get("all_batches"):
        with st.expander("Batch history"):
            for batch in result.get("all_batches", []):
                st.markdown(f"**Outer batch {batch['outer_attempt']}**")
                for inner in batch["inner_batches"]:
                    st.write(
                        f"- Inner attempt {inner['inner_attempt']}: "
                        f"{inner['drafts_generated']} drafts, "
                        f"{inner['survivors']} survivors, "
                        f"{inner['rejected']} rejected"
                    )

    survivors = result.get("final_survivors") or []
    if survivors:
        with st.expander(f"All survivors ({len(survivors)})"):
            for d in survivors:
                st.markdown(f"**Draft {d['run_id']}**")
                cols = st.columns(3)
                cols[0].metric("Quality", f"{d.get('quality_score', '?')}")
                cols[1].metric("AI estimate", f"{d.get('ai_estimate', 0):.1f}")
                cols[2].metric("Verdict", f"{d.get('quality_verdict', '?')}")
                if d.get("quality_reason"):
                    st.caption(d["quality_reason"])
                st.text_area(
                    f"Draft {d['run_id']} text",
                    value=d["text"],
                    height=200,
                    key=f"last_survivor_{d['run_id']}",
                )

    log_lines = result.get("log_lines") or []
    if log_lines:
        with st.expander("Run log (full)"):
            st.text("\n".join(log_lines))


def _load_last_terminal_result_from_disk() -> Optional[dict]:
    """Find the most recent run on disk whose status is a terminal state
    (complete / no_winner / terminated / crashed) and rehydrate it into
    the result-dict shape that _render_last_result_panel expects.

    Returns None if no terminal-state run exists yet."""
    recent = RunState.list_recent_runs(app_dir=APP_DIR, n=20)
    terminal = ["complete", "no_winner", "terminated", "crashed"]
    for m in recent:
        if m.get("status") not in terminal:
            continue
        run_dir = Path(m.get("run_dir", ""))
        if not run_dir.exists():
            continue
        try:
            state = RunState.load(run_dir)
        except Exception:
            continue

        # Rehydrate winner if there was one
        winner = None
        winner_rid = state.manifest.get("winner_run_id")
        if winner_rid:
            wp = state.winner_path()
            if wp and wp.exists():
                # Pull scores from drafts manifest summary
                draft_summary = state.manifest.get("drafts", {}).get(winner_rid, {})
                winner = {
                    "run_id":          winner_rid,
                    "text":            wp.read_text(encoding="utf-8"),
                    "quality_score":   draft_summary.get("quality_score", "?"),
                    "quality_verdict": draft_summary.get("quality_verdict", "?"),
                    "ai_estimate":     draft_summary.get("ai_estimate", 0),
                }

        # Rehydrate batches from manifest (compact form, no draft text)
        all_batches = []
        for ob in state.manifest.get("outer_batches", []):
            all_batches.append({
                "outer_attempt": ob["outer_attempt"],
                "inner_batches": [{
                    "inner_attempt":     ib["inner_attempt"],
                    "drafts_generated":  ib.get("n_generated", 0),
                    "survivors":         ib.get("n_survivors", 0),
                    "rejected":          ib.get("n_rejected", 0),
                } for ib in ob.get("inner_batches", [])],
            })

        return {
            "winner":              winner,
            "all_batches":         all_batches,
            "final_survivors":     [],   # not persisted in full text form
            "outline_label":       state.manifest.get("label", ""),
            "log_lines":           [],
            "elapsed":             0,
            "run_dir":             str(run_dir),
            "max_batches_step_8":  state.manifest.get("config", {}).get("max_batches_step_8", "?"),
            "_source":             f"disk ({m.get('status')})",
        }
    return None


def main():
    st.set_page_config(
        page_title=f"simpleapp {APP_VERSION} — commercial fiction drafter",
        layout="wide",
    )
    st.title(f"simpleapp {APP_VERSION} — commercial fiction drafter")
    st.caption(f"App version: `{APP_VERSION}` · Author Construction · Step 6 Gate · Diagnostic · Winner")
    st.caption(
        "9-step architecture per architecture_record_v1.txt. "
        "Author Construction step + 5 architectural design rules at prompt layer; "
        "15 pattern-prohibition caps verified at step 6 (with configurable gate policy); "
        "quality + AI estimate scored at step 7; winner picked at step 9. "
        f"{APP_VERSION} stamps version and outline label into run dirs + output filenames "
        "(matching v36 convention). v42 gate-policy + diagnostic carried forward; "
        "v41 persistence carried forward."
    )

    # --------------------------------------------------------------
    # Sidebar — configuration
    # --------------------------------------------------------------
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input(
            "Anthropic API key",
            value=os.environ.get("ANTHROPIC_API_KEY", ""),
            type="password",
        )
        model = st.selectbox("Model", AVAILABLE_MODELS, index=0)
        st.divider()
        st.subheader("Step 5 drafting prompt")
        try:
            _prompts = load_prompts_csv()
        except FileNotFoundError as _exc:
            st.error(f"prompts.csv missing: {_exc}")
            _prompts = []
        if _prompts:
            # Show id [category] first-60-chars-of-text, default to last row.
            _options = [
                f"{p['id']} [{p['category']}] {p['text'][:60]}"
                + ("…" if len(p['text']) > 60 else "")
                for p in _prompts
            ]
            _default_index = len(_prompts) - 1
            _selected_label = st.selectbox(
                f"Pick a prompt from prompts.csv ({len(_prompts)} available)",
                options=_options,
                index=_default_index,
                help="Drafting instruction sent to the model at Step 5. "
                     "Add a new row to prompts.csv to add a new option.",
            )
            _selected_idx = _options.index(_selected_label)
            csv_prompt_row = _prompts[_selected_idx]
            csv_prompt_id = csv_prompt_row["id"]
            csv_prompt_text = csv_prompt_row["text"]
            with st.expander("Selected prompt (full text)", expanded=False):
                st.text(csv_prompt_text)
        else:
            csv_prompt_id = ""
            csv_prompt_text = ""
        st.divider()
        st.subheader("Pipeline parameters")
        n_drafts = st.number_input(
            "Drafts per batch (X)", min_value=1, max_value=10, value=DEFAULT_N_DRAFTS,
        )
        quality_min = st.number_input(
            "Quality threshold (0-10)", min_value=0, max_value=10, value=DEFAULT_QUALITY_MIN,
        )
        ai_estimate_min = st.number_input(
            "AI-estimate threshold (0-100)", min_value=0, max_value=100,
            value=DEFAULT_AI_ESTIMATE_MIN,
        )
        min_survivors = st.number_input(
            "Min survivors after step 6", min_value=1, max_value=10,
            value=DEFAULT_MIN_SURVIVORS_STEP_6,
        )
        max_batches_step_6 = st.number_input(
            "Max batches at step 6", min_value=1, max_value=10,
            value=DEFAULT_MAX_BATCHES_STEP_6,
        )
        max_batches_step_8 = st.number_input(
            "Max batches at step 8", min_value=1, max_value=10,
            value=DEFAULT_MAX_BATCHES_STEP_8,
        )

        st.divider()
        st.subheader("Step 6 gate policy")
        gate_preset_name = st.selectbox(
            "Preset",
            options=list(PRESETS.keys()),
            index=0,
            help="STRICT reproduces v40/v41 behavior. PRAGMATIC tolerates "
                 "mechanical caps (4,16,18) within v10H budgets and light "
                 "caps (8,9,14) at 1, but keeps cadence caps (1,2,3,5,10,19) "
                 "hard at 0. USER_PROPOSED is the policy from chat: 1 hit "
                 "per cap, max 3 caps tripped per draft.",
        )
        st.caption(format_policy(PRESETS[gate_preset_name]))
        per_cap_overrides_json = st.text_area(
            "Per-cap overrides (JSON, optional)",
            value="",
            height=60,
            placeholder='{"4": 6, "16": 2}',
            help="Override the preset's per-cap tolerance budgets. Keys are "
                 "cap IDs (as strings), values are max-hits-per-cap. Applied "
                 "on top of the selected preset.",
        )
        max_caps_override = st.number_input(
            "Override max caps tripped (0 = use preset)",
            min_value=0, max_value=15, value=0,
            help="0 keeps the preset's value. Positive = max caps with any "
                 "violation. Applies on top of the selected preset.",
        )

        st.divider()
        st.subheader("Failure diagnostic")
        run_diagnostic = st.checkbox(
            "Run diagnostic when an outer batch fails",
            value=True,
            help="When an outer batch can't produce min_survivors at step 6, "
                 "aggregate the rejected drafts' cap incidence and use Claude "
                 "to characterize the top failing caps with concrete upstream "
                 "fix proposals. Persists to ./runs/<run>/diagnoses/.",
        )
        diagnostic_max_caps = st.number_input(
            "Caps to characterize per failed batch",
            min_value=1, max_value=12, value=5,
            help="Number of top caps (by breadth of failure) for which to "
                 "request an LLM characterization. Higher = more signal, "
                 "but each cap is one API call.",
        )

    # ==============================================================
    # Main body — v36-style two-column layout
    # Left column: outline input, Run button, live log (during a run)
    # Right column: recent runs, last-completed-run results (persistent
    # across reruns via st.session_state)
    # ==============================================================
    left_col, right_col = st.columns([1, 1])

    # --------------------------------------------------------------
    # LEFT COLUMN — inputs
    # --------------------------------------------------------------
    with left_col:
        st.subheader("Step 1 — Chapter outline")
        upload = st.file_uploader("Upload outline (.txt or .md)", type=["txt", "md"])
        pasted = st.text_area(
            "Or paste the outline below",
            height=200,
            placeholder="Paste your chapter outline here...",
        )

        outline_text = ""
        upload_filename: Optional[str] = None
        if upload is not None:
            upload_filename = upload.name
            outline_text = upload.read().decode("utf-8", errors="replace")
            st.success(f"Loaded outline from `{upload_filename}` ({len(outline_text)} chars)")
        elif pasted.strip():
            outline_text = pasted.strip()

        outline_label = derive_outline_label(upload_filename, outline_text) if outline_text else ""
        if outline_label:
            st.caption(f"Outline label for filenames: `{outline_label}`")

        run_button = st.button(
            "Run pipeline",
            type="primary",
            disabled=not (outline_text and api_key and csv_prompt_text),
        )

        # Live log area — only populated during an actual run
        log_placeholder = st.empty()

    # --------------------------------------------------------------
    # RIGHT COLUMN — recent runs + last completed run results
    # This column is rendered on every script execution. Because the
    # last result is cached in st.session_state, a browser refresh,
    # tab switch, or websocket reconnect after a completed run still
    # shows the winner and download buttons here.
    # --------------------------------------------------------------
    with right_col:
        _render_recent_runs_panel()
        st.divider()
        _render_last_result_panel()

    # --------------------------------------------------------------
    # Run the pipeline only when the button was just clicked.
    # All result-rendering happens via st.session_state in
    # _render_last_result_panel above — so a rerun after the click
    # still displays the winner.
    # --------------------------------------------------------------
    if not run_button:
        return

    client = anthropic.Anthropic(api_key=api_key)
    log_lines = []

    def log(msg: str):
        log_lines.append(msg)
        with left_col:
            with log_placeholder.container():
                st.text("\n".join(log_lines[-30:]))

    start_time = time.time()

    gate_policy = build_policy_from_ui(
        preset_name=gate_preset_name,
        per_cap_overrides_json=per_cap_overrides_json,
        max_caps_override=max_caps_override if max_caps_override > 0 else None,
    )

    try:
        result = run_pipeline(
            client=client,
            model=model,
            outline_text=outline_text,
            csv_prompt_id=csv_prompt_id,
            csv_prompt_text=csv_prompt_text,
            n_drafts=n_drafts,
            quality_min=quality_min,
            ai_estimate_min=ai_estimate_min,
            min_survivors_step_6=min_survivors,
            max_batches_step_6=max_batches_step_6,
            max_batches_step_8=max_batches_step_8,
            log=log,
            gate_policy=gate_policy,
            run_diagnostic=run_diagnostic,
            diagnostic_max_caps=diagnostic_max_caps,
            outline_label=outline_label,
        )
    except Exception as e:
        with right_col:
            st.error(f"Pipeline failed: {e}")
            st.text(traceback.format_exc())
            st.warning(
                "Drafts generated before the crash are persisted to disk "
                "under ./runs/. See the **Recent runs** panel on the right "
                "to download them. A crash log (crash.txt) is in the most "
                "recent run directory."
            )
        return

    elapsed = time.time() - start_time
    log(f"Pipeline complete in {elapsed:.1f}s")

    # Cache the result in session_state so a subsequent rerun
    # (browser refresh, button click elsewhere, websocket reconnect)
    # re-displays it without re-running the pipeline.
    st.session_state["last_result"] = {
        "winner":              result["winner"],
        "all_batches":         result["all_batches"],
        "final_survivors":     result["final_survivors"],
        "outline_label":       outline_label,
        "log_lines":           log_lines,
        "elapsed":             elapsed,
        "run_dir":             str(result["run_state"].run_dir) if result.get("run_state") else None,
        "max_batches_step_8":  max_batches_step_8,
        "_source":             "this session",
    }

    # Trigger one rerun so the right column re-renders from session_state
    # immediately. (Without this, the result wouldn't appear until the next
    # user interaction.)
    st.rerun()


if __name__ == "__main__":
    main()
