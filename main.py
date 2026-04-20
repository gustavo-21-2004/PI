import os
import chromadb
import ollama
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModel

# 1. Inicializar ChromaDB

chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection("piBigData")

# 2. Carregar BioBERTpt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Carregando BioBERTpt...")
tokenizer = AutoTokenizer.from_pretrained("pucpr/biobertpt-clin")
model = AutoModel.from_pretrained("pucpr/biobertpt-clin").to(device)
print("Modelo carregado.")

# 3. Embedding (COM NORMALIZAÇÃO)

def embed_text(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()

    embedding = embedding / np.linalg.norm(embedding)

    return embedding.tolist()

# 4. Limpeza

def clean_text(text):
    text = str(text)
    text = " ".join(text.split())
    text = " ".join([w for w in text.split() if not w.isdigit()])
    return text.lower()

# 5. Indexação (AJUSTE MÍNIMO)

def upload_directory(path):

    doc_id = 0
    print("\nIndexando diretório:", path)

    for dirname, _, filenames in os.walk(path):

        for filename in filenames:

            file_path = os.path.join(dirname, filename)

            try:

                if filename.endswith(".csv"):

                    df = pd.read_csv(file_path)

                    for _, row in df.iterrows():

                        doenca = clean_text(row["doenca"])
                        sintomas = clean_text(row["sintomas"])

                        if not sintomas.strip():
                            continue

                        texto = f"sintomas: {sintomas}"

                        embedding = embed_text(texto)

                        collection.add(
                            embeddings=[embedding],
                            documents=[texto],
                            metadatas=[{"doenca": doenca}],
                            ids=[f"{file_path}_{doc_id}"]
                        )

                        doc_id += 1

            except Exception as e:
                print("Erro:", e)

    print("\nTotal indexado:", collection.count())

# 6. Indexar

upload_directory("./datasets")

print("\nIndexação concluída.\n")

# 7. Score híbrido

def score_sintomas(query, doc):
    q = set(query.split())
    d = set(doc.split())
    return len(q & d)

# 8. Loop de consulta

while True:

    query = input(">>> ")

    if query.lower() in ["sair", "exit", "quit"]:
        break

    query = clean_text(query)
    query_embedding = embed_text(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        include=["documents", "distances", "metadatas"]
    )

    docs = results["documents"][0]
    dists = results["distances"][0]
    metas = results["metadatas"][0]

    candidatos = []

    for doc, dist in zip(docs, dists):
        print(dist, doc[:80])

    for doc, dist, meta in zip(docs, dists, metas):
        if dist < 1.2:
            score = score_sintomas(query, doc)
            candidatos.append((doc, meta["doenca"], dist, score))

    candidatos = sorted(candidatos, key=lambda x: (x[2], -x[3]))

    if not candidatos:
        print("\nNenhuma doença relevante encontrada.\n")
        continue

    # Contexto

    context = "CONTEXTO CLÍNICO:\n"

    for i, (doc, doenca, dist, score) in enumerate(candidatos[:3]):

        context += f"""
[DOENÇA {i+1}]
Nome: {doenca}
Sintomas: {doc}
"""

    # PROMPT

    prompt = f"""
Você é um profissional de saúde realizando triagem clínica.

REGRAS CRÍTICAS:

- Considere APENAS sintomas explicitamente mencionados pelo paciente
- IGNORE quaisquer sintomas adicionais presentes no contexto
- NÃO adicione sintomas implícitos ou típicos da doença
- Se houver sintomas no contexto que não estão na pergunta, desconsidere-os completamente
- Se não houver correspondência suficiente, diga: "Informações insuficientes"

Classifique usando Manchester:
Vermelho, Laranja, Amarelo, Verde, Azul

Formato obrigatório:

Sintomas considerados:
Possível condição:
Justificativa:
Classificação de risco (Manchester):
Conduta inicial:

{context}

Pergunta do paciente:
{query}
"""

    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}]
    )

    print("\nResposta:\n")
    print(response["message"]["content"])
    print("\n" + "="*50 + "\n")