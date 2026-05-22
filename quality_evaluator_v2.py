# quality_evaluator_v2.py
# Adapted from quality_evaluator_v1.py for use in simpleapp_v40.
#
# v2 changes from v1:
#   1. WORD COUNTS line in EVALUATOR_PROMPT step 1 — dropped the
#      "outline's target range" language. The new pipeline has stripped
#      word-count targets at every layer. The evaluator now notes word
#      counts only as a fact, with no target comparison.
#   2. GRAFT CANDIDATES — removed METHOD step 6 and the corresponding
#      OUTPUT FORMAT paragraph. There is no graft stage in simpleapp_v40.
#      The evaluator returns winner and ranking only.
#   3. Scanner fields — added new field names to match rule_verifier_v1's
#      scan_for_evaluator() output: polysyndetic_count, tautological_count,
#      negation_cluster_count, anaphora_count. Field names the simpleapp
#      scanner used (the_way, periphrastic, not_but, em_dash, emotion_naming,
#      aphoristic, backfill, verdict) carried forward unchanged.
#
# Source-extracted-from: simpleapp_v36_32.py (via quality_evaluator_v1.py).
#
# Lineage. This module contains the Q1 literary-quality evaluator originally
# embedded in the simpleapp drafting pipeline (v36 series). The simpleapp
# pipeline is being abandoned (per architecture_record_v1.txt, step 7 decision).
# The Q1 evaluator is the only "quality" scoring mechanism the prior stack had,
# and the v32 local_scorer infrastructure (band_classifier, extended_band_features,
# local_scorer) only produces AI-detection score estimates, not literary quality.
# This module is preserved for use in the new pipeline's step 7 (quality scoring
# on rule-compliant survivors).
#
# Source: simpleapp_v36_32.py lines 414-455 (EVALUATOR_PROMPT and
# EVALUATOR_SCANNER_BLOCK) and lines 2610-2773 (evaluate_drafts_with_anthropic).
#
# What this module provides.
#   - EVALUATOR_PROMPT: the instruction block for Claude evaluating N drafts.
#   - EVALUATOR_SCANNER_BLOCK: the scanner + outline framing block.
#   - evaluate_drafts_with_anthropic(): the call + parse function. Returns
#     a dict with quality verdicts (ACCEPTABLE/UNACCEPTABLE), quality scores
#     (0-10), a ranking, and a winner.
#
# What this module DOES NOT do.
#   - It does not score AI detection. Use local_scorer.py for that.
#   - It does not handle grafting, Stage F/G routing, line edits, or any
#     other simpleapp downstream logic. All of that was the part being
#     abandoned. This is purely the literary-quality evaluator.
#
# Dependencies the caller must supply.
#   - An Anthropic client (anthropic.Anthropic instance).
#   - A model identifier string.
#   - drafts: a list of dicts each with keys "run_id" (str) and "text" (str).
#   - outline_text: the chapter outline as a string (the same outline used
#     to generate the drafts).
#   - scan_by_run_id: optional dict mapping run_id -> dict of scanner counts.
#     If the new pipeline's step 6 produces rule-violation counts in a
#     compatible shape, plug them in here. If not, pass None and the prompt
#     will note that scanner data is not available per draft.
#
# Adaptation notes for the new pipeline.
#   - The scanner fields the prompt references (the_way, periphrastic, not_but,
#     em_dash, etc.) match the 20-cap pattern-prohibitions. The new pipeline's
#     step 6 verifier is expected to produce counts on a similar set; the dict
#     keys may need renaming to match what step 6 emits. The prompt asks the
#     evaluator to "reference them when assessing prose, but do not let them
#     drive your quality verdict" — so missing fields degrade gracefully.
#   - The prompt asks for word counts against an "outline's target range."
#     The new pipeline has stripped word-count targets at every layer. Either
#     edit the prompt's METHOD step 1 to drop the target-range language, or
#     accept that the evaluator will note word counts without a range to
#     compare against. Recommend editing the prompt; one-line change.
#   - The GRAFT CANDIDATES section (METHOD step 6 and the "graft paragraph"
#     in OUTPUT FORMAT) is dead weight in the new architecture — there is no
#     graft stage. Remove that section from the prompt before deployment.
#     One paragraph removal, no other changes needed.
#   - "WINNER: N" output is still useful as a literary-grounds tie-break
#     before the AI-score gate. Keep it.
#
# Status. UNADAPTED. As-extracted from simpleapp_v36_32. Before using in
# the new pipeline, do the three edits above (word-count line, GRAFT
# CANDIDATES removal, scanner-field renaming if applicable). After those
# edits, this becomes quality_evaluator_v2.py.
import re
# The new pipeline will set this; the original value in simpleapp was 16000.
MAX_EVAL_TOKENS = 16000
EVALUATOR_PROMPT = """You are evaluating {N} drafts of the same chapter against its outline. You have three inputs: the chapter outline, the mechanical scanner results for each draft, and the drafts themselves.
Read every draft in full. Do not skim.
Your job is to do three things in order:
(1) apply a lenient quality floor so the pipeline knows which drafts are fit to ship at all,
(2) assign each draft a prose-quality score,
and (3) rank only the top-scoring drafts.
The pipeline will keep ONLY the drafts that tie for the highest QUALITY_SCORE among ACCEPTABLE drafts. Every acceptable draft below that top score is discarded before the downstream AI ranking. So be willing to use ties when the writing quality is genuinely equal, but do not collapse distinct quality levels into a tie out of caution.
YOUR METHOD — in this order:
1. WORD COUNTS. Note each draft's word count. The new pipeline has no chapter-level target; record the count as a fact for comparison across drafts.
2. MECHANICAL COMPLIANCE. The scanner results are provided below. For each draft, note the violation counts. Do not re-scan — use the provided numbers. Reference them when assessing prose, but do not let them drive your quality verdict. Violations affect downstream diagnostics; your job here is writing quality.
3. QUALITY FLOOR — one verdict per draft. For each draft, decide ACCEPTABLE or UNACCEPTABLE. Apply a LENIENT standard: mark a draft ACCEPTABLE unless you would be embarrassed to ship it. UNACCEPTABLE means one or more of:
   - Voice collapse: the POV character's interior voice is absent, generic, or wrong register for long stretches.
   - Beats missing or compressed to the point of incoherence: a scene the outline requires is not on the page or is a throwaway line.
   - Dialogue that doesn't land: exchanges without subtext, without weapons, without stakes; turns that read like exposition dumps.
   - Structural failure: the chapter doesn't arrive where the outline says it arrives, or the ending doesn't close what was opened.
   - Prose-level damage: runs of flat summary where the outline asks for scene, long stretches of interpretive narration where the outline asks for observation and judgment, abandoned subplots, characters acting out of their profiles.
   Merely being less elegant than another draft is NOT grounds for UNACCEPTABLE. Stylistic difference is NOT grounds for UNACCEPTABLE. A draft can be ACCEPTABLE even if another draft is better at the same beats.
4. QUALITY SCORE — one integer score per draft, on a 1–10 scale, where 10 is the strongest prose in this batch and 1 is the weakest prose that still functions at all. Score on prose quality only: voice fidelity, dialogue craft, interior sharpness, beat execution, specificity, texture, rhythm, wit. Use the scale comparatively across THIS batch. If two drafts are genuinely equal in prose quality, give them the same score. UNACCEPTABLE drafts should get a score of 0.
5. TIE-ONLY RANKING. Rank ONLY the ACCEPTABLE drafts that received the highest QUALITY_SCORE. Omit every other draft from the ranking line, even if acceptable. The downstream AI ranking will break ties among these top-scoring drafts.
OUTPUT FORMAT
For each draft, write a paragraph (3-5 sentences) covering voice quality, best moment, notable weaknesses, and a one-sentence justification for your quality verdict. Reference the scanner numbers.
Then on a line by itself for each draft (one line per draft):
QUALITY: Draft N — ACCEPTABLE
or
QUALITY: Draft N — UNACCEPTABLE — [one-sentence reason]
Then on a line by itself for each draft:
QUALITY_SCORE: Draft N — S
(where S is an integer from 0 to 10. Use 0 only for UNACCEPTABLE drafts.)
Then on a line by itself:
RANKING: N, N, N, ...
(ONLY the ACCEPTABLE drafts tied at the highest QUALITY_SCORE, from strongest to weakest if there is still a distinction. Separated by commas. If only one draft has the top score, the line should contain only that draft number.)
Then on the final line:
WINNER: N
(the one draft from the RANKING line you would ship on literary grounds, before the downstream AI ranking breaks ties)
Nothing after that line."""
EVALUATOR_SCANNER_BLOCK = """=== MECHANICAL SCANNER RESULTS ===
{scanner_text}
=== CHAPTER OUTLINE ===
{outline_text}
"""
def evaluate_drafts_with_anthropic(
    client, model: str, drafts: list,
    outline_text: str = "", scan_by_run_id: dict = None,
) -> dict:
    n = len(drafts)
    scanner_lines = []
    for i, d in enumerate(drafts, 1):
        scan = (scan_by_run_id or {}).get(d["run_id"], {})
        if scan:
            scanner_lines.append(
                f"Draft {i} (run_id: {d['run_id']}): "
                f"word_count={len(d['text'].split())}, "
                f"the_way={scan.get('scan_the_way_count', '?')}, "
                f"periphrastic={scan.get('scan_periphrastic_count', '?')}, "
                f"not_but={scan.get('scan_not_but_count', '?')}, "
                f"em_dash={scan.get('scan_em_dash_count', '?')} "
                f"({scan.get('scan_em_dash_per_1k', '?')}/1k), "
                f"emotion_naming={scan.get('scan_emotion_naming_count', '?')}, "
                f"aphoristic={scan.get('scan_aphoristic_count', '?')}, "
                f"backfill={scan.get('scan_backfill_count', '?')}, "
                f"verdict={scan.get('scan_verdict_count', '?')}, "
                f"polysyndetic={scan.get('scan_polysyndetic_count', '?')}, "
                f"tautological={scan.get('scan_tautological_count', '?')}, "
                f"negation_cluster={scan.get('scan_negation_cluster_count', '?')}, "
                f"anaphora={scan.get('scan_anaphora_count', '?')}, "
                f"hard_cap_pass={scan.get('scan_hard_cap_pass', '?')}"
            )
        else:
            scanner_lines.append(
                f"Draft {i} (run_id: {d['run_id']}): "
                f"word_count={len(d['text'].split())}, scanner data not available"
            )
    scanner_text = "\n".join(scanner_lines)
    parts = [
        EVALUATOR_PROMPT.format(N=n),
        "\n\n",
        EVALUATOR_SCANNER_BLOCK.format(
            scanner_text=scanner_text,
            outline_text=outline_text.strip() if outline_text else "(no outline provided)",
        ),
    ]
    for i, d in enumerate(drafts, 1):
        parts.append(f"=== DRAFT {i} (run_id: {d['run_id']}) ===\n\n{d['text']}\n\n")
    resp = client.messages.create(
        model=model,
        max_tokens=MAX_EVAL_TOKENS,
        messages=[{"role": "user", "content": "".join(parts)}],
    )
    raw = "\n".join(b.text for b in resp.content if getattr(b, "text", None))
    quality_by_index = {}
    quality_pattern = re.compile(
        r"QUALITY:\s*Draft\s*(\d+)\s*[—-]+\s*(ACCEPTABLE|UNACCEPTABLE)"
        r"(?:\s*[—-]+\s*(.+?))?(?=\n|$)",
        re.IGNORECASE,
    )
    for m in quality_pattern.finditer(raw):
        idx = int(m.group(1))
        verdict = m.group(2).upper()
        reason = (m.group(3) or "").strip()
        if 1 <= idx <= n:
            quality_by_index[idx] = {"verdict": verdict, "reason": reason}
    for i in range(1, n + 1):
        if i not in quality_by_index:
            quality_by_index[i] = {
                "verdict": "ACCEPTABLE",
                "reason": "(no explicit verdict in evaluator output; defaulted to ACCEPTABLE)",
            }
    quality_score_by_index = {}
    score_pattern = re.compile(
        r"QUALITY_SCORE:\s*Draft\s*(\d+)\s*[—-]+\s*(-?\d+)(?=\n|$)",
        re.IGNORECASE,
    )
    for m in score_pattern.finditer(raw):
        idx = int(m.group(1))
        score = int(m.group(2))
        if 1 <= idx <= n:
            score = max(0, min(score, 10))
            if quality_by_index[idx]["verdict"] == "UNACCEPTABLE":
                score = 0
            quality_score_by_index[idx] = score
    parse_status = "clean"
    ranking = []
    rank_match = re.search(r"RANKING:\s*([0-9,\s]+)", raw)
    if rank_match:
        nums = [int(x.strip()) for x in rank_match.group(1).split(",") if x.strip().isdigit()]
        seen = set()
        deduped = []
        for x in nums:
            if 1 <= x <= n and x not in seen:
                seen.add(x)
                deduped.append(x)
        ranking = [x for x in deduped if quality_by_index[x]["verdict"] == "ACCEPTABLE"]
    else:
        parse_status = "no_ranking_line"
    if ranking:
        fallback_map = {}
        current = 10
        for idx in ranking:
            if idx not in fallback_map:
                fallback_map[idx] = current
                current = max(1, current - 1)
    else:
        acceptable_idxs = [
            i for i in range(1, n + 1)
            if quality_by_index[i]["verdict"] == "ACCEPTABLE"
        ]
        fallback_map = {idx: 10 for idx in acceptable_idxs}
    for i in range(1, n + 1):
        if i in quality_score_by_index:
            continue
        if quality_by_index[i]["verdict"] == "UNACCEPTABLE":
            quality_score_by_index[i] = 0
        else:
            quality_score_by_index[i] = fallback_map.get(i, 10)
            if parse_status == "clean":
                parse_status = "partial_missing_quality_score"
    acceptable_idxs = [
        i for i in range(1, n + 1)
        if quality_by_index[i]["verdict"] == "ACCEPTABLE"
    ]
    top_quality_score = max((quality_score_by_index[i] for i in acceptable_idxs), default=0)
    top_quality_idxs = [
        i for i in acceptable_idxs
        if quality_score_by_index[i] == top_quality_score
    ]
    if ranking:
        ranking = [x for x in ranking if x in acceptable_idxs]
        missing_acceptable = [i for i in acceptable_idxs if i not in ranking]
        ranking += missing_acceptable
        if missing_acceptable and parse_status == "clean":
            parse_status = "partial"
    else:
        ranking = acceptable_idxs[:]
    if not ranking:
        ranking = acceptable_idxs[:]
    winner_match = re.search(r"WINNER:\s*(\d+)", raw)
    if winner_match:
        winner_idx = int(winner_match.group(1))
    else:
        winner_idx = ranking[0] if ranking else 1
        if parse_status == "clean":
            parse_status = "no_winner_line"
    if winner_idx not in ranking:
        winner_idx = ranking[0] if ranking else winner_idx
    winner_idx = max(1, min(winner_idx, n))
    winner_run_id = drafts[winner_idx - 1]["run_id"]
    quality_by_run_id = {}
    quality_score_by_run_id = {}
    for i, d in enumerate(drafts, 1):
        quality_by_run_id[d["run_id"]] = quality_by_index[i]
        quality_score_by_run_id[d["run_id"]] = int(quality_score_by_index[i])
    return {
        "winner_run_id": winner_run_id,
        "winner_index": winner_idx,
        "ranking": ranking,
        "quality_by_run_id": quality_by_run_id,
        "quality_by_index": quality_by_index,
        "quality_score_by_run_id": quality_score_by_run_id,
        "quality_score_by_index": quality_score_by_index,
        "top_quality_score": int(top_quality_score),
        "top_quality_indexes": top_quality_idxs,
        "raw_text": raw,
        "parse_status": parse_status,
        "model": model,
    }
