"""rule_verifier_v1.py

Rule-compliance verifier for step 6 of the simpleapp_v37 pipeline.

Lineage. Pattern detectors lifted from simpleapp_v36_32.py (scan_draft and
its helpers) and extended with new detectors for the caps the simpleapp
scanner did not previously cover. This module is the step 6 gate: it takes
a chapter draft and returns whether the draft passes the 15 pattern-
prohibition caps from option (b) of step 4 of the architecture record.

Caps covered (regex/detector-based):
  Cap 1  — the-way constructions             [lifted from simpleapp]
  Cap 2  — periphrastic observational         [lifted from simpleapp]
  Cap 3  — not-X-but-Y negation pivots        [lifted from simpleapp]
  Cap 4  — em-dash density                    [lifted from simpleapp]
  Cap 5  — emotion-naming / meta-naming       [lifted from simpleapp]
  Cap 8  — aphoristic standalone              [lifted from simpleapp]
  Cap 9  — explanatory backfill               [lifted from simpleapp]
  Cap 10 — verdict constructions              [lifted from simpleapp]
  Cap 14 — negation-as-action clusters        [lifted from simpleapp]
  Cap 16 — polysyndetic and-chains            [NEW in v1]
  Cap 18 — tautological loops                 [NEW in v1]
  Cap 19 — anaphoric escalation               [lifted from simpleapp]

Caps not covered (LIMITATION — to be added in v2 when reliable detectors
are written; until then, the quality evaluator at step 7 catches these by
human-style read):
  Cap 11 — triple noun-phrase escalation
  Cap 13 — echoed-clipped dialogue
  Cap 17 — aphoristic generalisation clusters

Why these three are deferred. Cap 11 needs an evaluative-modifier classifier
to distinguish "three comma-separated noun phrases" (permitted) from "three
comma-separated noun phrases with evaluative modifiers" (prohibited). Cap 13
needs robust dialogue-pair extraction and token-overlap analysis. Cap 17
needs an aphoristic-statement classifier. All three are LLM-tractable but
not regex-tractable at production reliability. The quality evaluator at
step 7 reads the full chapter and will catch these by literary judgment;
they do not need to gate step 6.

PUBLIC API.
  verify_draft(text: str) -> dict
    Returns:
      {
        "pass": bool,
        "counts": dict (per-cap violation counts),
        "violations": list of dicts (each: cap_id, rule_name, count),
        "flagged_passages": list of dicts (each: rule, context, ~80-char excerpt),
      }
  scan_for_evaluator(text: str) -> dict
    Returns a flat dict of count fields with the same key names the quality
    evaluator's prompt references (scan_the_way_count, scan_periphrastic_count,
    etc.). Used to populate the scanner_text block in the quality evaluator's
    input.
"""
import re
import json
from collections import Counter


# ============================================================================
# Pattern constants — lifted verbatim from simpleapp_v36_32.py
# ============================================================================

# Cap 1 — "The way X" family.
THE_WAY_PATTERN = re.compile(r"\bthe\s+way\s+\w+", re.IGNORECASE)

# Cap 2 — Periphrastic observational ("as though he were", "in the manner of").
PERIPHRASTIC_PATTERN = re.compile(
    r"\b(?:as\s+though\s+(?:he|she|it|they)\s+were|in\s+the\s+manner\s+of)\b",
    re.IGNORECASE,
)

# Cap 3 — "Not X but Y" negation pivots. Filtered against dialogue at scan time.
NOT_BUT_PATTERN = re.compile(
    r"\bnot\s+(?:[a-z\s,']{1,50}?)\s+but\s+(?:[a-z]+)",
    re.IGNORECASE,
)

# Cap 5 — Emotion-naming in narration.
EMOTION_WORDS = (
    "anger|anxiety|anxious|bitter|calm|contempt|despair|disgust|dread|"
    "embarrassment|envy|fear|fearful|frustration|grief|guilt|happiness|"
    "happy|hope|hopeless|joy|joyful|loneliness|love|melancholy|nostalgia|"
    "panic|peace|pity|pride|proud|rage|regret|relief|remorse|resentment|"
    "sad|sadness|satisfaction|shame|shock|sorrow|surprise|tenderness|"
    "terror|tired|tiredness|weariness|weary|worry|yearning"
)
EMOTION_NAMING_PATTERN = re.compile(
    rf"\b(?:she\s+felt|he\s+felt|a\s+wave\s+of\s+(?:{EMOTION_WORDS})|"
    rf"a\s+flush\s+of\s+(?:{EMOTION_WORDS})|a\s+pang\s+of\s+(?:{EMOTION_WORDS})|"
    rf"with\s+a\s+sense\s+of\s+(?:{EMOTION_WORDS})|"
    rf"a\s+(?:{EMOTION_WORDS})\s+(?:settled|rose|came|washed|filled|took))\b",
    re.IGNORECASE,
)

# Cap 5 variant — meta-naming of feelings.
META_NAMING_PATTERN = re.compile(
    r"\bI\s+(?:name|named)\s+(?:it|the|this|that|my)\b[^.!?\n]{0,120}[.!?]",
    re.IGNORECASE,
)

# Cap 8 — Aphoristic standalone — abstract subject + verdict verb.
APHORISTIC_STANDALONE_PATTERN = re.compile(
    r"(?:^|(?<=[.!?\u201d\"])\s+)"
    r"(?:The\s+)?"
    r"(?:morning|evening|afternoon|night|dawn|dusk|day|silence|air|hour|"
    r"room|quiet|weather|house|year|dark|stillness|world|wind|rain|cold|"
    r"heat|light)"
    r"(?:\s+(?:light|air|wind|rain|cold|heat|quiet|silence|stillness))?"
    r"\s+"
    r"(?:did\s+not|offered\s+(?:no|nothing)|gave\s+(?:nothing|no|back)|"
    r"held\s+its|helped\s+nothing|was\s+no\s+(?:better|help|comfort|use|improvement)|"
    r"made\s+no\s+(?:difference|improvement))"
    r"\b",
    re.IGNORECASE,
)

# Cap 9 — Explanatory backfill.
EXPLANATORY_BACKFILL_PATTERN = re.compile(
    r",\s*(?:because|since)\s+(?:I|she|he)\s+(?:had|'d)\s+"
    r"(?:known|thought|realised|realized|seen|felt|understood|"
    r"suspected|recognised|recognized|guessed|sensed)\b",
    re.IGNORECASE,
)

# Cap 10 — "X too Y for Z" verdict construction.
VERDICT_TOO_FOR_PATTERN = re.compile(
    r"\btoo\s+\w+\s+for\s+(?:the|a|an|his|her|my|its|this|that|their|our)\s+\w+",
    re.IGNORECASE,
)

# Cap 14 — Negation-as-action single occurrence (clusters detected separately).
NEGATION_ACTION_PATTERN = re.compile(
    r"\b(?:I|he|she)\s+(?:do|does|did|will)\s+not\s+\w+",
    re.IGNORECASE,
)

# Cap 16 — Polysyndetic and-chains: sentences joining 4+ clauses with "and".
# NEW in rule_verifier_v1.
# Catches "X and Y and Z and W" within a single sentence. Three or fewer is
# fine; the cap binds at four or more.
POLYSYNDETIC_AND_PATTERN = re.compile(
    r"\band\b(?:[^.!?\n\u201c\u201d\"]{1,200}?\band\b){3,}",
    re.IGNORECASE,
)

# Cap 18 — Tautological loops: "[Subject] did the thing [subjects] do".
# NEW in rule_verifier_v1.
# Catches "her face did the thing faces do", "wolves did the thing wolves do",
# "his hands did the thing hands do", etc.
TAUTOLOGICAL_LOOP_PATTERN = re.compile(
    r"\b(?:her|his|its|my|their|our|the)\s+(\w+)\s+did\s+the\s+thing\s+"
    r"(?:\w+\s+)?\1s?\s+do(?:es)?\b",
    re.IGNORECASE,
)

# Sentence segmentation for the structural scanners.
_SCAN_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=["\u201c]?[A-Z])')


# ============================================================================
# Helpers — lifted from simpleapp_v36_32.py
# ============================================================================

def _scan_split_sentences(text: str) -> list:
    """Split a draft into sentences for structural scanners."""
    sents = _SCAN_SENT_SPLIT.split(text)
    return [s.strip() for s in sents if s.strip()]


def _scan_two_word_opener(s: str) -> str:
    """Lowercase first one or two words of a sentence, stripped of leading quote."""
    s = re.sub(r'^["\u201c]', '', s)
    m = re.match(r"(\w+\s+\w+)", s)
    return m.group(1).lower() if m else ""


def _scan_anaphora_hits(sents: list) -> list:
    """Cap 19 — 3+ sentences within any 5-sentence window sharing 2-word opener."""
    hits = []
    seen_triples = set()
    n = len(sents)
    for i in range(n - 2):
        window = sents[i:i + 5]
        openers = [(_scan_two_word_opener(s), s) for s in window]
        counts = Counter(o for o, _ in openers if o)
        for opener, cnt in counts.items():
            if cnt < 3:
                continue
            matching = tuple(idx for idx, (op, _) in enumerate(openers) if op == opener)
            triple_key = (i + matching[0], i + matching[1], i + matching[2])
            if triple_key in seen_triples:
                continue
            seen_triples.add(triple_key)
            hits.append({
                "rule": "anaphoric_escalation",
                "opener": opener,
                "context": " || ".join(sents[idx][:140] for idx in triple_key),
            })
            break
    return hits


def _scan_negation_action_clusters(sents: list, text: str) -> list:
    """Cap 14 — 3+ negation-as-action instances within any 6-sentence window.
    Single instances are permitted; clusters fire the cap."""
    sent_has_neg = []
    for s in sents:
        has = False
        for m in NEGATION_ACTION_PATTERN.finditer(s):
            before = s[: m.start()]
            normalized_before = before.replace("\u201c", '"').replace("\u201d", '"')
            if normalized_before.count('"') % 2 == 0:
                has = True
                break
        sent_has_neg.append(has)
    hits = []
    seen_clusters = set()
    n = len(sents)
    for i in range(n - 2):
        window = sent_has_neg[i:i + 6]
        cnt = sum(1 for v in window if v)
        if cnt >= 3:
            indices = tuple(i + k for k, v in enumerate(window) if v)[:3]
            if indices in seen_clusters:
                continue
            seen_clusters.add(indices)
            hits.append({
                "rule": "negation_as_action_cluster",
                "context": " || ".join(sents[idx][:140] for idx in indices),
            })
    return hits


def _scan_meta_naming(text: str) -> list:
    """Cap 5 variant — meta-naming feelings outside dialogue."""
    hits = []
    for m in META_NAMING_PATTERN.finditer(text):
        before = text[: m.start()]
        normalized = before.replace("\u201c", '"').replace("\u201d", '"')
        if normalized.count('"') % 2 != 0:
            continue
        start = max(0, m.start() - 30)
        end = min(len(text), m.end() + 30)
        hits.append({
            "rule": "meta_naming",
            "context": text[start:end].replace("\n", " ").strip(),
        })
    return hits


def _scan_polysyndetic(text: str, sents: list) -> list:
    """Cap 16 — polysyndetic and-chains in narration. NEW.
    A single sentence joining 4+ clauses with 'and'. Dialogue exempt.
    We work at the sentence level so we can apply the dialogue filter:
    a sentence wholly inside quotes is dialogue and is exempt."""
    hits = []
    for s in sents:
        stripped = s.strip()
        if stripped.startswith('"') or stripped.startswith("\u201c"):
            continue
        # Count "and" occurrences as standalone tokens (case-insensitive).
        and_count = len(re.findall(r"\band\b", s, re.IGNORECASE))
        if and_count >= 4:
            hits.append({
                "rule": "polysyndetic_and",
                "and_count": and_count,
                "context": s[:200].replace("\n", " ").strip(),
            })
    return hits


def _scan_tautological(text: str) -> list:
    """Cap 18 — tautological loops. NEW.
    Catches '[Subject] did the thing [subjects] do' and its variants."""
    hits = []
    for m in TAUTOLOGICAL_LOOP_PATTERN.finditer(text):
        before = text[: m.start()]
        normalized = before.replace("\u201c", '"').replace("\u201d", '"')
        if normalized.count('"') % 2 != 0:
            continue
        start = max(0, m.start() - 30)
        end = min(len(text), m.end() + 30)
        hits.append({
            "rule": "tautological_loop",
            "context": text[start:end].replace("\n", " ").strip(),
        })
    return hits


# ============================================================================
# Configuration — what triggers a fail
# ============================================================================

# Caps that fail on ANY occurrence. Zero tolerance.
# These are the "zero instances of" caps from the architecture record.
ZERO_TOLERANCE_RULES = {
    "the_way_x":                "Cap 1 (the-way constructions)",
    "periphrastic_observational": "Cap 2 (as-though periphrasis)",
    "not_x_but_y":              "Cap 3 (not-X-but-Y negation pivot)",
    "emotion_naming":           "Cap 5 (displaced feeling-metaphor)",
    "meta_naming":              "Cap 5 (meta-naming feelings)",
    "aphoristic_standalone":    "Cap 8 (aphoristic standalone verdict)",
    "explanatory_backfill":     "Cap 9 (because/since explanatory backfill)",
    "verdict_too_for":          "Cap 10 (verdict constructions)",
    "negation_as_action_cluster": "Cap 14 (negation-as-action clusters)",
    "polysyndetic_and":         "Cap 16 (polysyndetic and-chains)",
    "tautological_loop":        "Cap 18 (tautological loops)",
    "anaphoric_escalation":     "Cap 19 (anaphoric escalation)",
}

# Cap 4 — em-dashes are not zero-tolerance. The packet says target 0-4 for
# chapters under 3000 words and 0-6 for over 3000. We apply the over-3000
# ceiling (6) and a density ceiling (3.0/1k) as the verifier defaults.
EM_DASH_CEILING = 6
EM_DASH_PER_1K_CEILING = 3.0


# ============================================================================
# Public API
# ============================================================================

def verify_draft(text: str) -> dict:
    """Run all detectors and return pass/fail with violation details.

    Returns:
        {
            "pass":             bool,
            "counts":           dict — per-rule counts,
            "violations":       list — per-rule violation summaries that fail,
            "flagged_passages": list — flagged-context dicts for diagnostics,
            "em_dash_per_1k":   float — em-dash density per 1000 words,
        }
    """
    words = re.findall(r"\b[\w']+\b", text)
    wc = max(len(words), 1)
    sents = _scan_split_sentences(text)
    flagged = []

    # Cap 1
    the_way_matches = list(THE_WAY_PATTERN.finditer(text))
    for m in the_way_matches[:30]:
        flagged.append({
            "rule": "the_way_x",
            "context": text[max(0, m.start()-50):min(len(text), m.end()+50)].replace("\n", " ").strip(),
        })

    # Cap 2
    periphrastic_matches = list(PERIPHRASTIC_PATTERN.finditer(text))
    for m in periphrastic_matches[:15]:
        flagged.append({
            "rule": "periphrastic_observational",
            "context": text[max(0, m.start()-50):min(len(text), m.end()+50)].replace("\n", " ").strip(),
        })

    # Cap 3 — filter dialogue
    not_but_matches = []
    for m in NOT_BUT_PATTERN.finditer(text):
        before = text[: m.start()]
        normalized = before.replace("\u201c", '"').replace("\u201d", '"')
        if normalized.count('"') % 2 == 0:
            not_but_matches.append(m)
    for m in not_but_matches[:15]:
        flagged.append({
            "rule": "not_x_but_y",
            "context": text[max(0, m.start()-40):min(len(text), m.end()+40)].replace("\n", " ").strip(),
        })

    # Cap 4 — em-dashes are quantitative
    em_dash_count = text.count("\u2014")
    em_per_1k = round(em_dash_count / wc * 1000, 2)

    # Cap 5 emotion
    emotion_matches = list(EMOTION_NAMING_PATTERN.finditer(text))
    for m in emotion_matches[:15]:
        flagged.append({
            "rule": "emotion_naming",
            "context": text[max(0, m.start()-40):min(len(text), m.end()+40)].replace("\n", " ").strip(),
        })

    # Cap 5 meta-naming
    meta_naming_hits = _scan_meta_naming(text)
    for h in meta_naming_hits[:15]:
        flagged.append(h)

    # Cap 8
    aphoristic_matches = list(APHORISTIC_STANDALONE_PATTERN.finditer(text))
    for m in aphoristic_matches[:15]:
        flagged.append({
            "rule": "aphoristic_standalone",
            "context": text[max(0, m.start()-20):min(len(text), m.end()+80)].replace("\n", " ").strip(),
        })

    # Cap 9
    backfill_matches = list(EXPLANATORY_BACKFILL_PATTERN.finditer(text))
    for m in backfill_matches[:15]:
        flagged.append({
            "rule": "explanatory_backfill",
            "context": text[max(0, m.start()-60):min(len(text), m.end()+60)].replace("\n", " ").strip(),
        })

    # Cap 10 — filter dialogue
    verdict_matches = []
    for m in VERDICT_TOO_FOR_PATTERN.finditer(text):
        before = text[: m.start()]
        normalized = before.replace("\u201c", '"').replace("\u201d", '"')
        if normalized.count('"') % 2 == 0:
            verdict_matches.append(m)
    for m in verdict_matches[:15]:
        flagged.append({
            "rule": "verdict_too_for",
            "context": text[max(0, m.start()-40):min(len(text), m.end()+40)].replace("\n", " ").strip(),
        })

    # Cap 14
    negation_cluster_hits = _scan_negation_action_clusters(sents, text)
    for h in negation_cluster_hits[:15]:
        flagged.append(h)

    # Cap 16 — NEW
    polysyndetic_hits = _scan_polysyndetic(text, sents)
    for h in polysyndetic_hits[:15]:
        flagged.append(h)

    # Cap 18 — NEW
    tautological_hits = _scan_tautological(text)
    for h in tautological_hits[:15]:
        flagged.append(h)

    # Cap 19
    anaphora_hits = _scan_anaphora_hits(sents)
    for h in anaphora_hits[:15]:
        flagged.append(h)

    counts = {
        "the_way_x":              len(the_way_matches),
        "periphrastic":           len(periphrastic_matches),
        "not_x_but_y":            len(not_but_matches),
        "em_dash":                em_dash_count,
        "em_dash_per_1k":         em_per_1k,
        "emotion_naming":         len(emotion_matches),
        "meta_naming":            len(meta_naming_hits),
        "aphoristic_standalone":  len(aphoristic_matches),
        "explanatory_backfill":   len(backfill_matches),
        "verdict_too_for":        len(verdict_matches),
        "negation_cluster":       len(negation_cluster_hits),
        "polysyndetic_and":       len(polysyndetic_hits),
        "tautological_loop":      len(tautological_hits),
        "anaphoric_escalation":   len(anaphora_hits),
    }

    violations = []
    if counts["the_way_x"] > 0:
        violations.append({"cap": "Cap 1", "rule": "the_way_x", "count": counts["the_way_x"]})
    if counts["periphrastic"] > 0:
        violations.append({"cap": "Cap 2", "rule": "periphrastic", "count": counts["periphrastic"]})
    if counts["not_x_but_y"] > 0:
        violations.append({"cap": "Cap 3", "rule": "not_x_but_y", "count": counts["not_x_but_y"]})
    if counts["em_dash"] > EM_DASH_CEILING or counts["em_dash_per_1k"] > EM_DASH_PER_1K_CEILING:
        violations.append({"cap": "Cap 4", "rule": "em_dash",
                          "count": counts["em_dash"], "per_1k": counts["em_dash_per_1k"]})
    if counts["emotion_naming"] > 0:
        violations.append({"cap": "Cap 5", "rule": "emotion_naming", "count": counts["emotion_naming"]})
    if counts["meta_naming"] > 0:
        violations.append({"cap": "Cap 5", "rule": "meta_naming", "count": counts["meta_naming"]})
    if counts["aphoristic_standalone"] > 0:
        violations.append({"cap": "Cap 8", "rule": "aphoristic_standalone",
                          "count": counts["aphoristic_standalone"]})
    if counts["explanatory_backfill"] > 0:
        violations.append({"cap": "Cap 9", "rule": "explanatory_backfill",
                          "count": counts["explanatory_backfill"]})
    if counts["verdict_too_for"] > 0:
        violations.append({"cap": "Cap 10", "rule": "verdict_too_for", "count": counts["verdict_too_for"]})
    if counts["negation_cluster"] > 0:
        violations.append({"cap": "Cap 14", "rule": "negation_cluster", "count": counts["negation_cluster"]})
    if counts["polysyndetic_and"] > 0:
        violations.append({"cap": "Cap 16", "rule": "polysyndetic_and", "count": counts["polysyndetic_and"]})
    if counts["tautological_loop"] > 0:
        violations.append({"cap": "Cap 18", "rule": "tautological_loop", "count": counts["tautological_loop"]})
    if counts["anaphoric_escalation"] > 0:
        violations.append({"cap": "Cap 19", "rule": "anaphoric_escalation",
                          "count": counts["anaphoric_escalation"]})

    passed = len(violations) == 0

    return {
        "pass":             passed,
        "counts":           counts,
        "violations":       violations,
        "flagged_passages": flagged[:40],
        "em_dash_per_1k":   em_per_1k,
    }


def scan_for_evaluator(text: str) -> dict:
    """Return a flat dict of count fields keyed for the quality evaluator's
    scanner_text block. Field names match what quality_evaluator_v2.py expects."""
    result = verify_draft(text)
    c = result["counts"]
    return {
        "scan_the_way_count":          c["the_way_x"],
        "scan_periphrastic_count":     c["periphrastic"],
        "scan_not_but_count":          c["not_x_but_y"],
        "scan_em_dash_count":          c["em_dash"],
        "scan_em_dash_per_1k":         c["em_dash_per_1k"],
        "scan_emotion_naming_count":   c["emotion_naming"] + c["meta_naming"],
        "scan_aphoristic_count":       c["aphoristic_standalone"],
        "scan_backfill_count":         c["explanatory_backfill"],
        "scan_verdict_count":          c["verdict_too_for"],
        "scan_negation_cluster_count": c["negation_cluster"],
        "scan_polysyndetic_count":     c["polysyndetic_and"],
        "scan_tautological_count":     c["tautological_loop"],
        "scan_anaphora_count":         c["anaphoric_escalation"],
        "scan_hard_cap_pass":          result["pass"],
    }


def format_violation_summary(result: dict) -> str:
    """Human-readable one-line summary of verify_draft() output."""
    if result["pass"]:
        return "PASS — all 15 pattern-prohibition caps clean"
    parts = []
    for v in result["violations"]:
        if "per_1k" in v:
            parts.append(f"{v['cap']}: {v['count']} ({v['per_1k']}/1k)")
        else:
            parts.append(f"{v['cap']}: {v['count']}")
    return "FAIL — " + "; ".join(parts)
