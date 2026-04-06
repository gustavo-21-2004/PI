from flask import Flask, request, jsonify, render_template
import whisper
import chromadb
import torch
import os
import uuid
from transformers import AutoTokenizer, AutoModel
from llama_index.llms.ollama import Ollama
from llama_index.core.llms import ChatMessage
from gtts import gTTS

app = Flask(__name__)

# ==========================
# WHISPER
# ==========================
whisper_model = whisper.load_model("small")

# ==========================
# LLM
# ==========================
llm = Ollama(model="mistral", request_timeout=300.0)

# ==========================
# BANCO
# ==========================
os.makedirs("db", exist_ok=True)
client = chromadb.PersistentClient(path="./db")
collection = client.get_or_create_collection("triagem")

# ==========================
# EMBEDDING
# ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
model = AutoModel.from_pretrained("neuralmind/bert-base-portuguese-cased").to(device)


def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt",
                       truncation=True, padding=True).to(device)
    with torch.no_grad():
        emb = model(**inputs).last_hidden_state.mean(dim=1).squeeze()
    return emb.cpu().tolist()

# ==========================
# BUSCA
# ==========================
def buscar_similares(texto):
    try:
        data = collection.get()
        total = len(data["ids"]) if data and "ids" in data else 0

        if total == 0:
            return ["Nenhum caso encontrado"]

        emb = embed_text(texto)

        results = collection.query(
            query_embeddings=[emb],
            n_results=min(3, total)
        )

        return [m["content"] for m in results["metadatas"][0]]

    except:
        return ["Erro na busca"]

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
        ChatMessage(role="system", content="Especialista em triagem."),
        ChatMessage(role="user", content=prompt)
    ]

    resp = llm.chat(messages)
    return resp.message.content

# ==========================
# ROTAS
# ==========================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    texto = data.get("mensagem")

    resposta = responder(texto)

    # gerar áudio
    filename = f"static/audio_{uuid.uuid4().hex}.mp3"
    gTTS(text=resposta[:800], lang="pt-br").save(filename)

    return jsonify({
        "resposta": resposta,
        "audio": filename
    })


# ==========================
# RODAR
# ==========================
if __name__ == "__main__":
    app.run(debug=True)