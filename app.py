import streamlit as st
import datetime
from engine import ContextoEngine, get_daily_word

# Page config
st.set_page_config(page_title="Contexto Română", page_icon="🇷🇴", layout="centered")

# Custom CSS for Premium Look
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        color: #ffffff;
    }
    
    .stTextInput > div > div > input {
        background-color: rgba(10, 10, 20, 0.6);
        color: #e0e0e0;
        border: 1px solid rgba(0, 210, 255, 0.3);
        border-radius: 12px;
        padding: 12px 20px;
        font-size: 1.2rem;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #00d2ff;
        background-color: rgba(20, 20, 40, 0.8);
        box-shadow: 0 0 15px rgba(0, 210, 255, 0.2);
    }
    
    .guess-card {
        background: rgba(255, 255, 255, 0.08);
        padding: 18px;
        border-radius: 14px;
        margin-bottom: 12px;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    
    .guess-info {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
        z-index: 1;
        position: relative;
    }
    
    .guess-word {
        font-weight: 700;
        font-size: 1.1rem;
        letter-spacing: 1px;
    }
    
    .guess-rank {
        font-size: 0.9rem;
        opacity: 0.9;
        font-weight: 600;
    }
    
    .similarity-track {
        width: 100%;
        height: 12px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 6px;
        overflow: hidden;
    }
    
    .similarity-fill {
        height: 100%;
        border-radius: 6px;
        transition: width 0.5s ease-out;
    }
    
    h1 {
        text-align: center;
        font-weight: 700;
        font-size: 3rem;
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px !important;
    }
    
    .subtitle {
        text-align: center;
        opacity: 0.7;
        margin-bottom: 40px;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize Engine
@st.cache_resource
def load_engine():
    return ContextoEngine()

engine = load_engine()

# Session State
if "guesses" not in st.session_state:
    st.session_state.guesses = []
if "secret_word" not in st.session_state:
    st.session_state.secret_word = get_daily_word()
if "all_ranks" not in st.session_state:
    st.session_state.all_ranks = engine.get_sorted_vocab_ranks(st.session_state.secret_word)
    # The secret word itself is Rank 1. most_similar results start from Rank 2.
    st.session_state.rank_map = {word: i+2 for i, (word, sim) in enumerate(st.session_state.all_ranks)}

# UI Layout
st.markdown("<h1>Contexto Română</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Ghicește cuvântul secret. Cu cât ești mai aproape semantic, cu atât rangul este mai mic.</p>", unsafe_allow_html=True)

# Input
guess_input = st.text_input("Introdu un cuvânt:", key="input_field", placeholder="ex: casă, speranță, munte...")

if guess_input:
    guess = guess_input.strip().lower()
    if guess and guess not in [g['word'] for g in st.session_state.guesses]:
        if guess in engine.model or guess == st.session_state.secret_word:
            if guess == st.session_state.secret_word:
                rank = 1
                similarity = 1.0
            else:
                rank = st.session_state.rank_map.get(guess, 99999)
                similarity = engine.get_similarity(guess, st.session_state.secret_word)
            
            st.session_state.guesses.append({
                "word": guess,
                "rank": rank,
                "similarity": similarity
            })
            st.session_state.guesses.sort(key=lambda x: x['rank'])
        else:
            st.error(f"Cuvântul '{guess}' nu este în dicționarul nostru.")

# Display Guesses
if st.session_state.guesses:
    st.write("### Istoric Ghiciri")
    for g in st.session_state.guesses:
        rank = g['rank']
        word = g['word']
        sim = g['similarity']
        
        # Determine color and visual feedback
        if rank == 1:
            color = "#4caf50" # Green
            label = "FELICITĂRI!"
            bar_width = 100
        elif rank <= 500:
            color = "#4caf50" # Green for close
            label = f"Rang: {rank}"
            bar_width = max(70, 100 - (rank / 10)) # Heuristic for bar width
        elif rank <= 3000:
            color = "#ff9800" # Orange for medium
            label = f"Rang: {rank}"
            bar_width = max(40, 70 - (rank / 100))
        else:
            color = "#f44336" # Red for far
            label = f"Rang: {rank}" if rank < 99999 else "Ești departe..."
            bar_width = max(10, 30 - (rank / 2000))
            
        # Draw Card
        st.markdown(f"""
            <div class="guess-card">
                <div class="guess-info">
                    <div class="guess-word">{word.upper()}</div>
                    <div class="guess-rank">{label}</div>
                </div>
                <div class="similarity-track">
                    <div class="similarity-fill" style="width: {bar_width}%; background-color: {color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

if any(g['rank'] == 1 for g in st.session_state.guesses):
    st.balloons()
    st.success(f"Ai găsit cuvântul secret: **{st.session_state.secret_word}**!")
    if st.button("JOACĂ DIN NOU", use_container_width=True):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.rerun()

# Sidebar Cleanup
st.sidebar.markdown("### Contexto 🇷🇴")
st.sidebar.info("Găsește cuvântul secret folosind indicii de context.")

if st.sidebar.button("RESET JOC (Cuvânt Nou)"):
    for key in st.session_state.keys():
        del st.session_state[key]
    st.rerun()

st.sidebar.write(f"Dicționar: {len(engine.model)} cuvinte")
