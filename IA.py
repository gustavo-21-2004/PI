import streamlit as st
import whisper
import chromadb
import torch
import os
import json
import uuid

from typing import List
from transformers import AutoTokenizer, AutoModel
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage
from gtts import gTTS
from audio_recorder_streamlit import audio_recorder

# ==========================
# CONFIG
# ==========================

st.set_page_config(page_title="Triagem IA", page_icon="💬", layout="wide")
st.title("💬 Assistente de Triagem Inteligente")

# ==========================
# WHISPER
# ==========================


@st.cache_resource
def load_whisper():
    return whisper.load_model("base")


whisper_model = load_whisper()

# ==========================
# LLM
# ==========================

llm = Ollama(
    model="mistral",
    request_timeout=300.0  # 🔥 aumenta tempo (5 min)
)

# ==========================
# BANCO (CHROMADB)
# ==========================


@st.cache_resource
def get_collection():
    os.makedirs("db", exist_ok=True)
    client = chromadb.PersistentClient(path="./db")
    return client.get_or_create_collection("triagem")


collection = get_collection()

# ==========================
# STATUS DO BANCO
# ==========================

st.sidebar.title("📊 Status")

try:
    total_cases = len(collection.get()["ids"])
except:
    total_cases = 0

st.sidebar.write(f"Casos no banco: {total_cases}")

# ==========================
# EMBEDDING
# ==========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("pucpr/biobertpt-clin")
    model = AutoModel.from_pretrained("pucpr/biobertpt-clin").to(device)
    return tokenizer, model


tokenizer, model = load_model()


def embed_text(text: str) -> List[float]:
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, padding=True).to(device)

    with torch.no_grad():
        emb = model(**inputs).last_hidden_state.mean(dim=1).squeeze()

    return emb.cpu().tolist()

# ==========================
# CARREGAR CASOS
# ==========================


def load_cases():
    if not os.path.exists("casos.txt"):
        return []
    with open("casos.txt", "r", encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]


@st.cache_resource
def populate_db():
    cases = load_cases()

    if not cases:
        return

    try:
        existing = set(collection.get()["ids"])
    except:
        existing = set()

    for i, case in enumerate(cases):
        cid = f"case_{i}"

        if cid not in existing:
            emb = embed_text(case)
            collection.add(
                ids=[cid],
                embeddings=[emb],
                metadatas=[{"content": case}]
            )


populate_db()

# ==========================
# BUSCA SEMÂNTICA
# ==========================


def buscar_similares(texto):
    try:
        total = len(collection.get()["ids"])
    except:
        total = 0

    if total == 0:
        return ["Nenhum caso no banco"]

    emb = embed_text(texto)

    results = collection.query(
        query_embeddings=[emb],
        n_results=min(3, total)
    )

    return [m["content"] for m in results["metadatas"][0]]

# ==========================
# TTS
# ==========================


def gerar_audio(texto):
    texto = texto[:800]
    filename = f"audio_{uuid.uuid4().hex}.mp3"
    gTTS(text=texto, lang="pt-br").save(filename)
    return filename

# ==========================
# IA
# ==========================


def responder(texto):

    similares = buscar_similares(texto)

    prompt = f"""
Sintomas:
{texto}

Casos similares:
{' '.join(similares)}

Classifique pelo protocolo de Manchester:
- Cor
- Justificativa
- Conduta
"""

    messages = [
        ChatMessage(role="system",
                    content="Você é especialista em triagem clínica."),
        ChatMessage(role="user", content=prompt)
    ]

    resp = llm.chat(messages)
    return resp.message.content

# ==========================
# HISTÓRICO
# ==========================


def salvar():
    with open("chat.json", "w", encoding="utf-8") as f:
        json.dump(st.session_state.messages, f, ensure_ascii=False)


def carregar():
    if os.path.exists("chat.json"):
        return json.load(open("chat.json", encoding="utf-8"))
    return []


if "messages" not in st.session_state:
    st.session_state.messages = carregar()

# ==========================
# EXIBIR CHAT
# ==========================

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("audio"):
            st.audio(msg["audio"])

# ==========================
# INPUT + BOTÃO DE ÁUDIO
# ==========================

col1, col2 = st.columns([5, 1])

with col1:
    user_input = st.chat_input("Digite sua mensagem...")

with col2:
    audio_bytes = audio_recorder(
        text="",
        icon_size="3x"
    )


if audio_bytes:
    with open("audio.m4a", "wb") as f:
        f.write(audio_bytes)

    result = whisper_model.transcribe("audio.m4a")
    user_input = result["text"]

# ==========================
# PROCESSAMENTO
# ==========================

if user_input:

    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    salvar()

    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Analisando..."):
            resposta = responder(user_input)

        st.write(resposta)

        audio_file = gerar_audio(resposta)
        st.audio(audio_file)

    st.session_state.messages.append({
        "role": "assistant",
        "content": resposta,
        "audio": audio_file
    })
    salvar()
