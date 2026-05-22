"""simpleapp_v37.py

Production Streamlit app implementing the 9-step commercial-fiction
drafting architecture from architecture_record_v1.txt.

ARCHITECTURE (per architecture_record_v1.txt):

  Step 1.  Operator uploads the chapter outline.
  Step 2.  LLM call: select 2-3 bestselling commercial fiction authors
           matching the chapter's genre/period/tone/pacing.
  Step 3.  Same LLM call: characterize the primary author's habits
           across 13 dimensions. Outputs the AUTHOR_CONSTRUCTION block.
  Step 4.  Five architectural design rules (Caps 6, 7, 12, 15, 20
           repurposed as positive design directives) are included in
           the drafting prompt. The other 15 caps are NOT in the prompt.
  Step 5.  Generate X drafts (default X=3) of the chapter from the
           outline + AUTHOR_CONSTRUCTION + design rules. Same temperature;
           variation from sampling.
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
  author_construction_prompt_v1.txt
  draft_chapter_prompt_v1.txt
  rule_verifier_v1.py
  quality_evaluator_v2.py
  local_scorer.py            (from v32)
  band_classifier.py         (from v32)
  extended_band_features.py  (from v32)
  calibration.json           (from v32)

USAGE:
  streamlit run simpleapp_v37.py
"""
from __future__ import annotations
import os
import sys
import io
import json
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
MAX_AUTHOR_TOKENS = 4000

DEFAULT_MODEL = "claude-opus-4-7"
AVAILABLE_MODELS = [
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-sonnet-4-6",
    "claude-haiku-4-5-20251001",
]


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
# Step 2-3 — Author selection and construction characterization
# ============================================================================
def select_authors_and_habits(
    client: anthropic.Anthropic,
    model: str,
    outline_text: str,
    log,
) -> tuple[str, dict]:
    """Run the author_construction prompt against the outline. Returns
    (author_construction_block, raw_response_meta)."""
    prompt = load_prompt("author_construction_prompt_v1.txt")
    full = f"{prompt}\n\n=== CHAPTER OUTLINE ===\n\n{outline_text}"
    log(f"Step 2-3 — generating author selection and habit characterization "
        f"(model: {model})")
    resp = client.messages.create(
        model=model,
        max_tokens=MAX_AUTHOR_TOKENS,
        messages=[{"role": "user", "content": full}],
    )
    text = "\n".join(b.text for b in resp.content if getattr(b, "text", None))
    return text, {"input_tokens": resp.usage.input_tokens,
                  "output_tokens": resp.usage.output_tokens}


def extract_author_construction_block(raw: str) -> str:
    """Pull the AUTHOR_CONSTRUCTION block out of the raw response.
    Falls back to the full response if markers aren't found."""
    start_marker = "=== AUTHOR_CONSTRUCTION ==="
    end_marker = "=== END AUTHOR_CONSTRUCTION ==="
    if start_marker in raw and end_marker in raw:
        i = raw.index(start_marker)
        j = raw.index(end_marker) + len(end_marker)
        return raw[i:j]
    return raw.strip()


# ============================================================================
# Step 5 — Generate one draft
# ============================================================================
def generate_draft(
    client: anthropic.Anthropic,
    model: str,
    outline_text: str,
    author_construction: str,
    run_id: str,
    log,
) -> dict:
    """Single drafting call. Returns dict with run_id, text, usage."""
    template = load_prompt("draft_chapter_prompt_v1.txt")
    prompt = template.replace("{outline}", outline_text).replace(
        "{author_construction}", author_construction
    )
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
    client, model, outline_text, author_construction, n_drafts, log
) -> list[dict]:
    """Generate n_drafts drafts and return them as a list."""
    drafts = []
    for i in range(n_drafts):
        run_id = uuid.uuid4().hex[:8]
        try:
            d = generate_draft(client, model, outline_text, author_construction, run_id, log)
            drafts.append(d)
            log(f"  Draft {run_id}: {len(d['text'].split())} words")
        except Exception as e:
            log(f"  ERROR generating draft {run_id}: {e}")
    return drafts


# ============================================================================
# Step 6 — Rule-compliance gate
# ============================================================================
def filter_compliant_drafts(drafts: list[dict], log) -> tuple[list[dict], list[dict]]:
    """Run verify_draft on each. Return (survivors, rejected)."""
    survivors = []
    rejected = []
    for d in drafts:
        result = verify_draft(d["text"])
        d["verify_result"] = result
        summary = format_violation_summary(result)
        if result["pass"]:
            survivors.append(d)
            log(f"  {d['run_id']}: {summary}")
        else:
            rejected.append(d)
            log(f"  {d['run_id']}: {summary}")
    return survivors, rejected


# ============================================================================
# Step 7 — Score survivors (quality + AI estimate)
# ============================================================================
def score_survivors(
    client, model, survivors, outline_text, log
) -> dict:
    """Compute quality scores (one shared LLM call) and per-draft AI estimates."""
    # Build the scan_by_run_id dict the evaluator expects.
    scan_by_run_id = {}
    for d in survivors:
        scan_by_run_id[d["run_id"]] = scan_for_evaluator(d["text"])
    log(f"Step 7 — quality evaluator on {len(survivors)} survivor(s)")
    eval_result = evaluate_drafts_with_anthropic(
        client=client,
        model=model,
        drafts=survivors,
        outline_text=outline_text,
        scan_by_run_id=scan_by_run_id,
    )
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
        try:
            predicted, details = score_text(d["text"], classifier=classifier)
            d["ai_estimate"] = float(predicted)
            d["ai_estimate_details"] = details
        except Exception as e:
            d["ai_estimate"] = 0.0
            d["ai_estimate_details"] = {"error": str(e)}
            log(f"  AI estimate ERROR for {d['run_id']}: {e}")
        d["quality_verdict"] = eval_result["quality_by_run_id"][d["run_id"]]["verdict"]
        d["quality_reason"]  = eval_result["quality_by_run_id"][d["run_id"]]["reason"]
        d["quality_score"]   = eval_result["quality_score_by_run_id"][d["run_id"]]
        log(f"  {d['run_id']}: quality={d['quality_verdict']} "
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
    n_drafts: int,
    quality_min: int,
    ai_estimate_min: float,
    min_survivors_step_6: int,
    max_batches_step_6: int,
    max_batches_step_8: int,
    log,
) -> dict:
    """Run all 9 steps. Returns a result dict."""
    # Step 2-3
    raw_author, author_usage = select_authors_and_habits(client, model, outline_text, log)
    author_construction = extract_author_construction_block(raw_author)
    log(f"Step 2-3 complete. Author block: {len(author_construction)} chars.")

    all_batches = []
    final_winner = None
    final_eval = None
    final_passing = []
    final_survivors = []

    for outer in range(max_batches_step_8):
        log(f"--- Outer batch {outer + 1}/{max_batches_step_8} ---")
        # Step 5-6 inner loop: keep generating until we have enough survivors.
        survivors = []
        inner_batches = []
        for inner in range(max_batches_step_6):
            log(f"Step 5 — generating draft batch ({n_drafts} drafts), "
                f"inner attempt {inner + 1}/{max_batches_step_6}")
            drafts = generate_draft_batch(
                client, model, outline_text, author_construction, n_drafts, log
            )
            log(f"Step 6 — rule-compliance gate on {len(drafts)} draft(s)")
            batch_survivors, batch_rejected = filter_compliant_drafts(drafts, log)
            inner_batches.append({
                "inner_attempt":  inner + 1,
                "drafts_generated": len(drafts),
                "survivors":     len(batch_survivors),
                "rejected":      len(batch_rejected),
                "drafts":        drafts,
            })
            survivors.extend(batch_survivors)
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
            continue

        # Step 7
        eval_result = score_survivors(client, model, survivors, outline_text, log)

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

    return {
        "winner":              final_winner,
        "author_construction": author_construction,
        "author_construction_raw": raw_author,
        "author_usage":        author_usage,
        "all_batches":         all_batches,
        "final_eval":          final_eval,
        "final_passing":       final_passing,
        "final_survivors":     final_survivors,
    }


# ============================================================================
# Streamlit UI
# ============================================================================
def main():
    st.set_page_config(
        page_title="simpleapp v37 — commercial fiction drafter",
        layout="wide",
    )
    st.title("simpleapp v37 — commercial fiction drafter")
    st.caption(
        "9-step architecture per architecture_record_v1.txt. "
        "Author Construction step + 5 architectural design rules at prompt layer; "
        "15 pattern-prohibition caps verified at step 6; "
        "quality + AI estimate scored at step 7; winner picked at step 9."
    )

    # Sidebar — configuration
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input(
            "Anthropic API key",
            value=os.environ.get("ANTHROPIC_API_KEY", ""),
            type="password",
        )
        model = st.selectbox("Model", AVAILABLE_MODELS, index=0)
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

    # Main — outline input
    st.subheader("Step 1 — Chapter outline")
    upload = st.file_uploader("Upload outline (.txt or .md)", type=["txt", "md"])
    pasted = st.text_area(
        "Or paste the outline below",
        height=300,
        placeholder="Paste your chapter outline here...",
    )

    outline_text = ""
    if upload is not None:
        outline_text = upload.read().decode("utf-8", errors="replace")
        st.success(f"Loaded outline from upload ({len(outline_text)} chars)")
    elif pasted.strip():
        outline_text = pasted.strip()

    run_button = st.button("Run pipeline", type="primary", disabled=not (outline_text and api_key))

    if not run_button:
        st.stop()

    # Run pipeline
    client = anthropic.Anthropic(api_key=api_key)
    log_lines = []
    log_placeholder = st.empty()

    def log(msg: str):
        log_lines.append(msg)
        with log_placeholder.container():
            st.text("\n".join(log_lines[-30:]))

    start_time = time.time()
    try:
        result = run_pipeline(
            client=client,
            model=model,
            outline_text=outline_text,
            n_drafts=n_drafts,
            quality_min=quality_min,
            ai_estimate_min=ai_estimate_min,
            min_survivors_step_6=min_survivors,
            max_batches_step_6=max_batches_step_6,
            max_batches_step_8=max_batches_step_8,
            log=log,
        )
    except Exception as e:
        st.error(f"Pipeline failed: {e}")
        st.text(traceback.format_exc())
        st.stop()

    elapsed = time.time() - start_time
    log(f"Pipeline complete in {elapsed:.1f}s")

    # Results
    st.divider()
    winner = result["winner"]
    if winner is None:
        st.error(
            f"No draft passed the thresholds within "
            f"{max_batches_step_8} outer batch(es). "
            f"See diagnostics below for what was tried."
        )
    else:
        st.success(
            f"Winner: {winner['run_id']} — "
            f"quality {winner['quality_score']}, "
            f"ai_estimate {winner['ai_estimate']:.1f}"
        )
        st.download_button(
            "Download winning chapter (.txt)",
            data=winner["text"],
            file_name=f"chapter_{winner['run_id']}.txt",
            mime="text/plain",
        )
        st.subheader("Winning chapter")
        st.text_area("Chapter prose", value=winner["text"], height=500)

    # Diagnostics
    st.divider()
    st.subheader("Diagnostics")

    with st.expander("Author construction block (step 2-3)"):
        st.code(result["author_construction"], language="text")

    with st.expander("Batch history"):
        for batch in result["all_batches"]:
            st.markdown(f"**Outer batch {batch['outer_attempt']}**")
            for inner in batch["inner_batches"]:
                st.write(
                    f"- Inner attempt {inner['inner_attempt']}: "
                    f"{inner['drafts_generated']} drafts, "
                    f"{inner['survivors']} survivors, "
                    f"{inner['rejected']} rejected"
                )

    if result.get("final_survivors"):
        with st.expander(f"All survivors ({len(result['final_survivors'])})"):
            for d in result["final_survivors"]:
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
                    key=f"survivor_{d['run_id']}",
                )

    with st.expander("Run log (full)"):
        st.text("\n".join(log_lines))


if __name__ == "__main__":
    main()
