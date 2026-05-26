"""
Commercial Fiction Chapter Harness — Quality-First + Score-Gated Pipeline
=========================================================================
Version: 37.4a (Commercial Edition) | vGem6 Synchronized
"""

import os
import re
import json
import io
import time
import zipfile
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# =====================================================================
# 1. GLOBAL CONFIGURATION & LEXICON (2026 TURBO TARGETS)
# =====================================================================
GRAFT_ONLY = "GRAFT_ONLY"
DATA_FILE = "labeled_corpus.json"

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

def load_or_initialize_corpus():
    """Ensures training database persistence with hard historical data profiles."""
    if os.path.exists(DATA_FILE):
        try:
            with open(DATA_FILE, "r") as f:
                data = json.load(f)
                if len(data) >= 5:
                    return data
        except Exception:
            pass
            
    seed_data = [
        {"text": "The wind howled. Inside the ancient room, a tapestry hung over the fireplace, serving as a timeless testament to history. Ultimately, he delved into the multifaceted landscape.", "score": 12.0},
        {"text": "Cold rain hit the glass. Hard. He didn't blink; his fingers traced the deep gouge in the walnut table—a parting gift from the blade last November. Silence filled the halls.", "score": 98.0},
        {"text": "She was particularly anxious. The journey ahead was a symphony of dangers, showcasing an interplay of dark forces. In conclusion, it was an underscore of survival.", "score": 5.0},
        {"text": "The engine dead-ended. Metal scraped frozen mud. If the radiator blew now, they were walking back to Alpine—assuming the wolves left enough to bury.", "score": 95.0},
        {"text": "A multifaceted approach to the problem was selected. It was a beacon of hope in a changing world.", "score": 10.0}
    ]
    with open(DATA_FILE, "w") as f:
        json.dump(seed_data, f, indent=4)
    return seed_data
# =====================================================================
# 2. STAGE F: RANDOM FOREST ENGINE (MACHINE LEARNING SCORER)
# =====================================================================

def extract_features(text):
    """
    Extracts multi-dimensional structural features from raw text to feed
    the non-linear Random Forest Model.
    """
    if not text or len(text.strip()) == 0:
        return [0.0, 0.0, 0.0, 0.0]
        
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    word_counts = [len(s.split()) for s in sentences if len(s.split()) > 0]
    
    if not word_counts:
        return [0.0, 0.0, 0.0, 0.0]

    # Feature 1: Syntactic Burstiness (Sentence Length Standard Deviation)
    std_dev = float(np.std(word_counts)) if len(word_counts) > 1 else 0.0
    
    # Feature 2: Punctuation Complexity Density (Dashes, Semicolons, Colons)
    punc_count = text.count('—') + text.count(';') + text.count(':')
    total_words = sum(word_counts)
    complexity_density = float(punc_count / total_words) if total_words > 0 else 0.0
    
    # Feature 3: Density of Blacklisted AI Tell Expressions
    ai_word_matches = 0
    for category, pattern_list in AI_TELL_WORDS.items():
        for pattern, _ in pattern_list:
            ai_word_matches += len(re.findall(pattern, text, re.IGNORECASE))
    ai_density = float(ai_word_matches / total_words) if total_words > 0 else 0.0
    
    # Feature 4: Short-to-Long Sentence Asymmetry Ratio
    short_sentences = sum(1 for w in word_counts if w <= 6)
    long_sentences = sum(1 for w in word_counts if w >= 24)
    asymmetry_ratio = float((short_sentences + long_sentences) / len(word_counts))

    return [std_dev, complexity_density, ai_density, asymmetry_ratio]


def train_local_scorer():
    """Fits an advanced Random Forest Regressor to capture feature thresholds."""
    corpus = load_or_initialize_corpus()
    
    X = []
    y = []
    for item in corpus:
        X.append(extract_features(item["text"]))
        y.append(item["score"])
        
    model = make_pipeline(
        StandardScaler(),
        RandomForestRegressor(n_estimators=150, max_depth=6, random_state=42)
    )
    model.fit(X, y)
    return model


# =====================================================================
# 3. STAGE G: STRUCTURAL ANALYSIS & REPAIR ENGINE
# =====================================================================

def strip_aphoristic_gloss(text):
    """
    G4 Pass: Strips artificial moralizing and summary codas 
    traditionally appended by base LLM drafting engines.
    """
    gloss_patterns = [
        r"In the end, (.*)\.",
        r"It was (.*) not (.*) but (.*)\.",
        r"Ultimately, (.*)\.",
        r"Perhaps the real (.*) was (.*)\.",
        r"It stood as a testament to (.*)\."
    ]
    
    paragraphs = text.split('\n\n')
    cleaned_paragraphs = []
    
    for para in paragraphs:
        lines = para.strip().split('\n')
        if not lines:
            continue
            
        last_line = lines[-1].strip()
        is_gloss = False
        
        for pattern in gloss_patterns:
            if re.match(pattern, last_line, re.IGNORECASE):
                is_gloss = True
                break
                
        if is_gloss:
            lines.pop()  # Drop the trailing narrative synthesis
            
        cleaned_paragraphs.append('\n'.join(lines))
        
    return '\n\n'.join(cleaned_paragraphs)


def check_burstiness(text):
    """Calculates variance indicators to identify rhythmic monotony."""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    restructure_flags = []
    
    for i, para in enumerate(paragraphs):
        sentences = re.split(r'(?<=[.!?])\s+', para)
        lengths = [len(s.split()) for s in sentences if len(s.split()) > 0]
        
        if len(lengths) > 1:
            variance = np.std(lengths)
            if variance < 5.0:  # Rigid uniformity flag threshold
                restructure_flags.append({
                    "paragraph_index": i,
                    "variance": float(variance),
                    "text": para
                })
    return restructure_flags

def run_line_edit_pass(raw_draft_text):
    """Executes the sequence of linguistic and rhythmic post-corrections."""
    # Step 1: Strip thematic codas
    processed_text = strip_aphoristic_gloss(raw_draft_text)
    
    # Step 2: Lexicon Scan and Standard Replacements
    for category, pairs in AI_TELL_WORDS.items():
        for pattern, replacement in pairs:
            if replacement == "":  # Deletable structural hedges
                processed_text = re.sub(pattern, replacement, processed_text, flags=re.IGNORECASE)
            elif replacement == GRAFT_ONLY:
                # GRAFT_ONLY tokens flagged for alternative draft substitution cycles
                processed_text = re.sub(pattern, " [REWRITE TARGET] ", processed_text, flags=re.IGNORECASE)
                
    # Step 3: Run Rhythmic Analysis
    rhythm_issues = check_burstiness(processed_text)
    
    return processed_text, rhythm_issues
# =====================================================================
# 4. GENERATION ORCHESTRATION, LLM CONNECTOR & EXPORTERS
# =====================================================================

def get_final_system_prompt(base_prompt_from_csv=None):
    """
    Safely merges your loaded template with the high-perplexity injection.
    Prevents NameErrors by checking for template state.
    """
    if base_prompt_from_csv:
        return f"{base_prompt_from_csv.strip()}\n\n{DRAFTING_SYSTEM_INJECTION.strip()}"
    return DRAFTING_SYSTEM_INJECTION.strip()


def mock_llm_generation(prompt, system_prompt):
    """
    Harness stub representing your active Anthropic / OpenAI client API hook.
    Injects a small wait time to replicate production text assembly.
    """
    time.sleep(1.0)
    # Falling back on a highly textured prose generation mock baseline
    return (
        "The engine dead-ended. Metal scraped frozen mud. "
        "If the radiator blew now, they were walking back to Alpine—assuming the wolves left enough to bury.\n\n"
        "Cold rain hit the glass. Hard. He didn't blink; his fingers traced the deep gouge in the walnut table—"
        "a parting gift from the blade last November. Silence filled the halls."
    )


def build_batch_manifest(generated_samples):
    """
    Assembles structural metrics across generations into a comprehensive DataFrame.
    Tracks features across parallel generation variants for rapid analysis.
    """
    rows = []
    for idx, sample in enumerate(generated_samples):
        feats = extract_features(sample["text"])
        rows.append({
            "Variant ID": f"Var_{idx + 1}",
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Word Count": len(sample["text"].split()),
            "Sentence StdDev": round(feats[0], 2),
            "Punctuation Density": round(feats[1], 4),
            "AI Tell Density": round(feats[2], 4),
            "Asymmetry Score": round(feats[3], 2),
            "Model Assigned Score": round(sample["predicted_score"], 1)
        })
    return pd.DataFrame(rows)


def package_production_zip(manifest_df, text_variants):
    """
    Compiles raw drafts, line-edited variations, and the execution manifest 
    into a production-grade in-memory ZIP package ready for disk write.
    """
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        # Write the run manifest
        csv_data = manifest_df.to_csv(index=False)
        archive.writestr("batch_run_manifest.csv", csv_data)
        
        # Write individual production variants
        for idx, variant in enumerate(text_variants):
            filename = f"production_variant_{idx + 1}.txt"
            archive.writestr(filename, variant["text"])
            
            # Write out an explicit analysis card alongside each file
            analysis_card = (
                f"=== METRIC REPORT CARD FOR VARIANT {idx + 1} ===\n"
                f"Calculated Originality: {variant['predicted_score']:.1f}%\n"
                f"Rhythm Invariances Flagged: {len(variant['rhythm_flags'])}\n"
            )
            archive.writestr(f"metrics_variant_{idx + 1}.log", analysis_card)
            
    return zip_buffer.getvalue()
# =====================================================================
# 5. STREAMLIT DASHBOARD & UI INTERFACE
# =====================================================================

def main():
    st.set_page_config(
        page_title="v37.4a Commercial Fiction Harness",
        page_icon="✍️",
        layout="wide"
    )

    # State Initialization: Persistent Scorer
    if 'scorer_engine' not in st.session_state:
        with st.spinner("Initializing Random Forest Neural Calibration..."):
            st.session_state.scorer_engine = train_local_scorer()
    
    if 'batch_history' not in st.session_state:
        st.session_state.batch_history = []

    # --- SIDEBAR CONTROLS ---
    with st.sidebar:
        st.title("🎛️ Pipeline Settings")
        st.caption("Active Config: v37.4a Commercial + vGem6")
        
        provider_key = st.text_input("API Access Key", type="password")
        model_choice = st.selectbox("Drafting Engine", ["claude-3-5-sonnet", "gpt-4o", "gemini-1.5-pro"])
        batch_count = st.slider("Samples to Generate", 1, 8, 4)
        
        st.divider()
        st.subheader("Model Status")
        st.success("Stage F: Random Forest ACTIVE")
        st.success("Stage G: Burstiness Filter ACTIVE")
        
        if st.button("Clear Batch Cache"):
            st.session_state.batch_history = []
            st.rerun()

    # --- MAIN UI TABS ---
    tab_draft, tab_eval, tab_export = st.tabs([
        "🚀 Production Drafting", 
        "🔬 Forensic Analysis", 
        "📦 Export & Manifest"
    ])

    # TAB 1: PRODUCTION DRAFTING
    with tab_draft:
        st.header("Drafting Orchestration")
        col_in, col_out = st.columns([1, 1])
        
        with col_in:
            outline_packet = st.text_area("Paste vGem6 Outline Packet", height=450, 
                                        placeholder="Paste the output from your vGem6 extraction here...")
            
            if st.button("Execute Batch Cycle", type="primary"):
                if not outline_packet:
                    st.error("Outline packet is required.")
                else:
                    progress_bar = st.progress(0)
                    batch_results = []
                    
                    for i in range(batch_count):
                        # 1. Generate (Mocking LLM call for this harness)
                        raw_prose = mock_llm_generation(outline_packet, DRAFTING_SYSTEM_INJECTION)
                        
                        # 2. Stage G: Repair Pass
                        cleaned_prose, rhythm_flags = run_line_edit_pass(raw_prose)
                        
                        # 3. Stage F: Score Pass
                        feats = extract_features(cleaned_prose)
                        score = st.session_state.scorer_engine.predict([feats])[0]
                        
                        batch_results.append({
                            "text": cleaned_prose,
                            "predicted_score": score,
                            "rhythm_flags": rhythm_flags
                        })
                        progress_bar.progress((i + 1) / batch_count)
                    
                    st.session_state.batch_history = batch_results
                    st.success(f"Batch complete. {batch_count} variants generated.")

        with col_out:
            if st.session_state.batch_history:
                st.subheader("Best Performing Variant")
                # Sort by score and get the top one
                best = max(st.session_state.batch_history, key=lambda x: x['predicted_score'])
                st.metric("Top Quality Score", f"{best['predicted_score']:.1f}%")
                st.text_area("Live Preview", best['text'], height=350)
            else:
                st.info("Awaiting batch execution...")

    # TAB 2: FORENSIC ANALYSIS (STAGE F/G MANUAL CHECK)
    with tab_eval:
        st.header("Manual Prose Scrub")
        eval_text = st.text_area("Paste External Prose for Scoring", height=300)
        
        if st.button("Run Forensic Analysis"):
            if eval_text:
                cleaned, issues = run_line_edit_pass(eval_text)
                feats = extract_features(cleaned)
                manual_score = st.session_state.scorer_engine.predict([feats])[0]
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Predicted Originality", f"{manual_score:.1f}%")
                c2.metric("Burstiness (StdDev)", f"{feats[0]:.2f}")
                c3.metric("Rhythm Violations", len(issues))
                
                if issues:
                    with st.expander("🚨 Flagged Paragraphs", expanded=True):
                        for issue in issues:
                            st.warning(f"Para {issue['paragraph_index']} (Low Var: {issue['variance']:.2f})")
                            st.write(issue['text'])
                
                st.subheader("Corrected Text")
                st.text_area("Sanitized Output", cleaned, height=250)

    # TAB 3: EXPORT & MANIFEST
    with tab_export:
        st.header("Batch Management")
        if st.session_state.batch_history:
            # Generate the manifest
            manifest_df = build_batch_manifest(st.session_state.batch_history)
            st.dataframe(manifest_df, use_container_width=True)
            
            # Prepare ZIP
            zip_data = package_production_zip(manifest_df, st.session_state.batch_history)
            
            st.download_button(
                label="📥 Download Batch Production ZIP",
                data=zip_data,
                file_name=f"batch_export_{datetime.now().strftime('%Y%m%d_%H%M')}.zip",
                mime="application/zip"
            )
        else:
            st.info("No active batch in memory. Go to Drafting to generate samples.")

if __name__ == "__main__":
    main()
