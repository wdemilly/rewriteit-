"""
Commercial Fiction Chapter Harness — Quality-First + Score-Gated Pipeline
=========================================================================
Version: 37.4a (Commercial Edition)
Integration: vGem6 Framework
"""

import os
import re
import json
import io
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# =====================================================================
# 1. CONSTANTS & LEXICON (Expanded G2 AI-tell triggers)
# =====================================================================
GRAFT_ONLY = "GRAFT_ONLY"
DATA_FILE = "labeled_corpus.json"

# This replaces the old list with the 2026 Turbo targets
AI_TELL_WORDS = {
    "ai_metaphors": [
        (r"\btapestry\b", GRAFT_ONLY), (r"\btestament\b", GRAFT_ONLY),
        (r"\brealm\b", GRAFT_ONLY), (r"\bbeacon\b", GRAFT_ONLY),
        (r"\blandscape\b", GRAFT_ONLY), (r"\bjourney\b", GRAFT_ONLY),
        (r"\bdance\b", GRAFT_ONLY), (r"\bsymphony\b", GRAFT_ONLY),
        (r"\binterplay\b", GRAFT_ONLY), (r"\bbackdrop\b", GRAFT_ONLY),
        (r"\bchorus\b", GRAFT_ONLY), (r"\bechoes\b", GRAFT_ONLY),
    ],
    "vague_transitions": [
        (r"\bdelve\b", GRAFT_ONLY), (r"\bunderscore\b", GRAFT_ONLY),
        (r"\bpivotal\b", GRAFT_ONLY), (r"\bmultifaceted\b", GRAFT_ONLY),
        (r"\bnuanced\b", GRAFT_ONLY), (r"\bultimately\b", GRAFT_ONLY),
        (r"\bin conclusion\b", GRAFT_ONLY),
    ],
    "hedges": [
        (r"\bparticular\b", ""), (r"\bparticularly\b", ""), (r"\bmerely\b", "")
    ]
}

DRAFTING_SYSTEM_INJECTION = """
# RHYTHM & STRUCTURAL CONSTRAINT
You are an experienced novelist executing a high-perplexity narrative. 
- You must strictly avoid the "middle-length" sentence trap.
- Every paragraph must feature asymmetric cadence: at least one rhythmic fragment under 5 words, and at least one complex sentence exceeding 25 words.
- Do not round off scenes with thematic synthesis or moral conclusions. Stop writing the exact moment the concrete action concludes.
"""

# =====================================================================
# 2. STAGE F: RANDOM FOREST SCORER (Machine Learning)
# =====================================================================

def extract_features(text):
    if not text or len(text.strip()) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    word_counts = [len(s.split()) for s in sentences if len(s.split()) > 0]
    if not word_counts:
        return [0.0, 0.0, 0.0, 0.0]

    std_dev = float(np.std(word_counts)) if len(word_counts) > 1 else 0.0
    punc_count = text.count('—') + text.count(';') + text.count(':')
    total_words = sum(word_counts)
    complexity_density = float(punc_count / total_words) if total_words > 0 else 0.0
    
    ai_word_matches = 0
    for category, pattern_list in AI_TELL_WORDS.items():
        for pattern, _ in pattern_list:
            ai_word_matches += len(re.findall(pattern, text, re.IGNORECASE))
    ai_density = float(ai_word_matches / total_words) if total_words > 0 else 0.0
    
    short_long_ratio = float((sum(1 for w in word_counts if w <= 6) + sum(1 for w in word_counts if w >= 24)) / len(word_counts))

    return [std_dev, complexity_density, ai_density, short_long_ratio]

def train_local_scorer():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            corpus = json.load(f)
    else:
        corpus = [
            {"text": "Sample baseline for training...", "score": 50.0},
            {"text": "Human text example with variation.", "score": 90.0}
        ]
    
    X = [extract_features(item["text"]) for item in corpus]
    y = [item["score"] for item in corpus]
    
    model = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=100, random_state=42))
    model.fit(X, y)
    return model

# =====================================================================
# 3. STAGE G: STRUCTURAL REPAIR (Burstiness & Gloss)
# =====================================================================

def strip_aphoristic_gloss(text):
    gloss_patterns = [r"In the end, (.*)\.", r"Ultimately, (.*)\.", r"It stood as a testament to (.*)\.", r"Perhaps the real (.*) was (.*)\."]
    paragraphs = text.split('\n\n')
    cleaned = []
    for para in paragraphs:
        lines = para.strip().split('\n')
        if lines and any(re.match(p, lines[-1], re.IGNORECASE) for p in gloss_patterns):
            lines.pop()
        cleaned.append('\n\n'.join(lines))
    return '\n\n'.join(cleaned)

def check_burstiness(text):
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    flags = []
    for i, para in enumerate(paragraphs):
        sentences = re.split(r'(?<=[.!?])\s+', para)
        lengths = [len(s.split()) for s in sentences if len(s.split()) > 0]
        if len(lengths) > 1 and np.std(lengths) < 5.0:
            flags.append({"index": i, "variance": np.std(lengths), "text": para})
    return flags

# =====================================================================
# 4. ORCHESTRATION FIX (Addressing the NameError)
# =====================================================================

def get_final_system_prompt(base_prompt_from_csv=None):
    """
    Safely merges the injection with existing prompt data.
    """
    if base_prompt_from_csv:
        return f"{base_prompt_from_csv}\n\n{DRAFTING_SYSTEM_INJECTION}"
    return DRAFTING_SYSTEM_INJECTION

# =====================================================================
# 5. STREAMLIT UI (Preserving your v37 Sidebar and Exporters)
# =====================================================================

def main():
    st.set_page_config(page_title="v37.4a Commercial Pipeline", layout="wide")
    
    # Initialize Scorer
    if 'scorer' not in st.session_state:
        st.session_state.scorer = train_local_scorer()

    st.title("Commercial Fiction Pipeline v37.4a")
    
    # Your existing Sidebar logic goes here (CSV loading, API keys, etc.)
    with st.sidebar:
        st.header("Pipeline Settings")
        api_key = st.text_input("Anthropic/OpenAI Key", type="password")
        batch_size = st.slider("Batch Size", 1, 10, 4)

    # Main Processing Area
    input_text = st.text_area("Prose Input for Analysis", height=300)
    
    if st.button("Evaluate & Clean"):
        # 1. Structural Repair
        cleaned = strip_aphoristic_gloss(input_text)
        rhythm_issues = check_burstiness(cleaned)
        
        # 2. ML Scoring
        feats = extract_features(cleaned)
        score = st.session_state.scorer.predict([feats])[0]
        
        st.metric("Predicted Originality Score", f"{score:.1f}%")
        
        if rhythm_issues:
            st.warning(f"Detected {len(rhythm_issues)} paragraphs with robotic rhythm.")
        
        st.subheader("Processed Output")
        st.text_area("Final Prose", cleaned, height=300)

if __name__ == "__main__":
    main()