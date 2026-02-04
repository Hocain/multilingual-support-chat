import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from langdetect import detect

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Global Customer Support Chat",
    page_icon="🌍",
    layout="centered"
)

st.title("🌍 Global Customer Support Chat")
st.caption("Users can chat in any language. Support replies in English only.")

# =========================
# MODEL CONFIG
# =========================
MODEL_NAME = "facebook/nllb-200-distilled-600M"
device = "cuda" if torch.cuda.is_available() else "cpu"

LANG_MAP = {
    "en": "eng_Latn",
    "fr": "fra_Latn",
    "es": "spa_Latn",
    "de": "deu_Latn",
    "hi": "hin_Deva",
    "ar": "arb_Arab",
    "zh": "zho_Hans",
    "ja": "jpn_Jpan",
    "pt": "por_Latn",
    "ru": "rus_Cyrl"
}

# =========================
# LOAD MODEL (CACHED)
# =========================
@st.cache_resource
def load_model():
    with st.spinner("Loading translation model (first time may take a minute)..."):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        model = model.to(device)
    return tokenizer, model

tokenizer, model = load_model()

# =========================
# TRANSLATION FUNCTIONS
# =========================
def translate(text, src_lang, tgt_lang, max_length=256):
    tokenizer.src_lang = LANG_MAP[src_lang]

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length
    ).to(device)

    outputs = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(LANG_MAP[tgt_lang]),
        max_length=max_length
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def user_to_english(text):
    try:
        detected = detect(text)
    except Exception:
        detected = "en"

    if detected not in LANG_MAP:
        detected = "en"

    english = translate(text, detected, "en")
    return english, detected

def english_to_user(text, user_lang):
    return translate(text, "en", user_lang)

# =========================
# SESSION STATE
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.user_lang = None

# =========================
# SHOW DETECTED LANGUAGE
# =========================
if st.session_state.user_lang:
    st.info(f"Detected language: {st.session_state.user_lang.upper()}")

# =========================
# CHAT HISTORY
# =========================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# =========================
# USER INPUT (ANY LANGUAGE)
# =========================
user_input = st.chat_input("Type your message in any language...")

if user_input:
    if st.session_state.user_lang is None:
        english_text, lang = user_to_english(user_input)
        st.session_state.user_lang = lang
    else:
        english_text = translate(
            user_input,
            st.session_state.user_lang,
            "en"
        )

    # User message
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # English view for agent
    st.session_state.messages.append({
        "role": "assistant",
        "content": f"**To Agent (English):** {english_text}"
    })

    st.rerun()

# =========================
# AGENT REPLY (ENGLISH ONLY)
# =========================
st.divider()
st.subheader("Agent Reply (English only)")

if st.session_state.user_lang is None:
    st.info("Waiting for user message...")
else:
    agent_reply = st.text_input("Type agent reply in English")

    if st.button("Send Reply") and agent_reply:
        translated_reply = english_to_user(
            agent_reply,
            st.session_state.user_lang
        )

        st.session_state.messages.append({
            "role": "assistant",
            "content": translated_reply
        })

        st.rerun()
