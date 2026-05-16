# ============================================================
# RECOMENDADOR LITERÁRIO — Flask + Backend ML completo
# pip install flask pymongo sentence-transformers scikit-learn
#             numpy pandas anthropic
#
# Rodar: python app.py
# Acesso: http://localhost:5000
# ============================================================

import os, re, json, pickle, textwrap
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template_string
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

app = Flask(__name__)
CACHE_PATH = "embeddings_cache.pkl"
FEEDBACK_PATH = "feedback_usuario.json"

# ============================================================
# 1. CONEXÃO MONGODB + CARGA DE DADOS
# ============================================================
def carregar_dados():
    client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=5000)
    try:
        client.admin.command("ping")
    except ConnectionFailure as e:
        raise RuntimeError(f"MongoDB não está rodando: {e}")

    collection = client["admin"]["Book_Dataset_V2"]
    docs = list(collection.find())
    df = pd.DataFrame(docs).drop(columns=["_id"], errors="ignore")
    df = df.reset_index(drop=True)

    # Normaliza rating
    scaler = MinMaxScaler()
    if "rating" in df.columns:
        df["rating_norm"] = scaler.fit_transform(
            df["rating"].fillna(0).values.reshape(-1, 1)
        ).flatten()
    else:
        df["rating_norm"] = 0.5

    return df

# ============================================================
# 2. PRÉ-PROCESSAMENTO
# ============================================================
SINONIMOS = {
    "triste":     "melancólico saudade tristeza luto",
    "feliz":      "alegre otimista esperança eufórico",
    "tenso":      "ansioso suspense medo angústia",
    "épico":      "grandioso heróico batalha guerra",
    "romântico":  "amor romance paixão afeto",
    "intenso":    "emocionante forte dramático",
    "misterioso": "suspense thriller enigma crime",
}

def limpar_texto(texto: str) -> str:
    texto = str(texto).lower().strip()
    texto = re.sub(r'[^\w\s]', ' ', texto, flags=re.UNICODE)
    texto = re.sub(r'\b\d+\b', ' ', texto)
    texto = re.sub(r'\s+', ' ', texto).strip()
    return texto

def expandir_query(texto: str) -> str:
    texto_limpo = limpar_texto(texto)
    extras = [v for k, v in SINONIMOS.items() if k in texto_limpo]
    return (texto_limpo + ' ' + ' '.join(extras)).strip() if extras else texto_limpo

def construir_perfil(row) -> str:
    campos = {
        "Sentimento I":  3,
        "Emoção I":      3,
        "Sentimento II": 2,
        "Emoção II":     2,
        "Gênero":        1,
    }
    partes = []
    for campo, peso in campos.items():
        val = limpar_texto(str(row.get(campo, "")))
        if val:
            partes.extend([val] * peso)
    return ' '.join(partes)

# ============================================================
# 3. EMBEDDINGS (com cache)
# ============================================================
MODELO_NOME = "paraphrase-multilingual-MiniLM-L12-v2"

def carregar_modelo_embeddings(df):
    modelo = SentenceTransformer(MODELO_NOME)
    df["perfil_emocional"] = df.apply(construir_perfil, axis=1)
    try:
        with open(CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
        if cache.get("modelo") == MODELO_NOME and cache["embeddings"].shape[0] == len(df):
            embeddings = cache["embeddings"]
            print(f"Embeddings carregados do cache. Shape: {embeddings.shape}")
            return modelo, embeddings
        raise ValueError("Cache desatualizado")
    except Exception:
        print("Computando embeddings (primeira vez — pode demorar ~30s)...")
        embeddings = modelo.encode(
            df["perfil_emocional"].tolist(),
            show_progress_bar=True,
            batch_size=32,
            normalize_embeddings=True,
        )
        with open(CACHE_PATH, "wb") as f:
            pickle.dump({"modelo": MODELO_NOME, "embeddings": embeddings}, f)
        print(f"Embeddings salvos. Shape: {embeddings.shape}")
        return modelo, embeddings

# ============================================================
# 4. CLASSIFICADOR (sem data leakage)
# ============================================================
def treinar_classificador(df):
    colunas_input = ["Gênero", "Emoção II"]
    coluna_target = "Sentimento I"
    df_clf = df[colunas_input + [coluna_target]].dropna()
    contagem = df_clf[coluna_target].value_counts()
    classes_validas = contagem[contagem >= 2].index
    df_clf = df_clf[df_clf[coluna_target].isin(classes_validas)]

    X_text = df_clf[colunas_input].apply(
        lambda row: limpar_texto(' '.join(row.astype(str))), axis=1
    )
    y_clf = df_clf[coluna_target]

    pipeline_clf = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=5000)),
        ("clf",   LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")),
    ])

    if len(df_clf) >= 10:
        X_train, X_test, y_train, _ = train_test_split(
            X_text, y_clf, test_size=0.4, random_state=42, stratify=y_clf
        )
        pipeline_clf.fit(X_train, y_train)
    else:
        pipeline_clf.fit(X_text, y_clf)

    print("Classificador treinado.")
    return pipeline_clf

# ============================================================
# 5. MMR
# ============================================================
def mmr_rerank(query_emb, candidatos_idx, embeddings, top_k=10, lambda_param=0.7):
    selecionados = []
    candidatos = list(candidatos_idx)
    for _ in range(min(top_k, len(candidatos))):
        if not candidatos:
            break
        sims_query = cosine_similarity(query_emb.reshape(1, -1), embeddings[candidatos])[0]
        if selecionados:
            sims_sel = cosine_similarity(embeddings[candidatos], embeddings[selecionados]).max(axis=1)
        else:
            sims_sel = np.zeros(len(candidatos))
        mmr_scores = lambda_param * sims_query - (1 - lambda_param) * sims_sel
        melhor = int(np.argmax(mmr_scores))
        selecionados.append(candidatos[melhor])
        candidatos.pop(melhor)
    return selecionados

# ============================================================
# 6. RECOMENDAÇÃO PRINCIPAL
# ============================================================
def recomendar_livros(query, top_k=10, peso_semantico=0.7, peso_rating=0.3,
                      usar_mmr=True, lambda_mmr=0.7, pre_filtro=50):
    query_expandida = expandir_query(query)
    emb_query = modelo.encode([query_expandida], normalize_embeddings=True)
    sims = cosine_similarity(emb_query, embeddings_livros)[0]
    score_hibrido = peso_semantico * sims + peso_rating * df["rating_norm"].values
    df["_score_sem"]   = sims
    df["_score_final"] = score_hibrido

    top_candidatos_idx = np.argsort(score_hibrido)[::-1][:pre_filtro].tolist()

    if usar_mmr and len(top_candidatos_idx) > top_k:
        idx_final = mmr_rerank(emb_query[0], top_candidatos_idx, embeddings_livros,
                               top_k=top_k, lambda_param=lambda_mmr)
    else:
        idx_final = top_candidatos_idx[:top_k]

    resultado = df.iloc[idx_final].copy()

    def gerar_explicacao(row):
        sentimento = row.get("Sentimento I", "") or row.get("Emoção I", "")
        genero = row.get("Gênero", "")
        sim_pct = round(row["_score_sem"] * 100, 1)
        return f"Match {sim_pct}% — {sentimento}, {genero}"

    resultado["explicacao"] = resultado.apply(gerar_explicacao, axis=1)

    # Fallback
    if resultado["_score_sem"].max() < 0.3:
        emocao_pred = pipeline_clf.predict([limpar_texto(query)])[0]
        fallback = df[df["Sentimento I"].str.lower() == emocao_pred.lower()].nlargest(top_k, "rating_norm")
        if not fallback.empty:
            return fallback

    colunas_saida = [c for c in [
        "Título do Livro", "Gênero", "Sentimento I", "Emoção I",
        "Sinopse", "Público-Alvo", "Páginas", "Tamanho do Livro",
        "rating", "_score_sem", "_score_final", "explicacao"
    ] if c in resultado.columns]

    return resultado[colunas_saida].reset_index(drop=True)

# ============================================================
# 7. ANÁLISE DE IA (Claude API)
# ============================================================
def analisar_top_livro(query: str, livro: dict) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return ""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        campos = {
            "Título":       livro.get("Título do Livro", "N/A"),
            "Gênero":       livro.get("Gênero", "N/A"),
            "Sentimento I": livro.get("Sentimento I", "N/A"),
            "Sentimento II":livro.get("Sentimento II", "N/A"),
            "Emoção I":     livro.get("Emoção I", "N/A"),
            "Emoção II":    livro.get("Emoção II", "N/A"),
            "Sinopse":      livro.get("Sinopse", "N/A"),
            "Público-Alvo": livro.get("Público-Alvo", "N/A"),
            "Páginas":      livro.get("Páginas", "N/A"),
            "Score Match":  f"{round(livro.get('_score_sem', 0) * 100, 1)}%",
        }
        livro_str = "\n".join(f"  {k}: {v}" for k, v in campos.items())
        prompt = f"""Você é um especialista em literatura e psicologia emocional.

O usuário descreveu seu estado emocional ou o que busca da seguinte forma:
  "{query}"

O sistema de recomendação selecionou este livro como o #1 mais adequado:
{livro_str}

Escreva UMA análise curta (3 a 5 frases), em português, explicando de forma
pessoal e empática por que este livro é a melhor escolha para o que o usuário
está sentindo ou buscando.

Formato de saída — comece exatamente com esta linha:
"Esse livro se encaixa com você porque..."

Seja específico: conecte os sentimentos/emoções do livro com as palavras do
usuário. Não repita o título no início. Sem bullet points. Apenas prosa fluida."""

        resposta = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return resposta.content[0].text.strip()
    except Exception as e:
        return f"Erro na análise de IA: {e}"

# ============================================================
# 8. FEEDBACK
# ============================================================
def registrar_feedback(titulo: str, util: bool, query: str):
    try:
        with open(FEEDBACK_PATH, "r", encoding="utf-8") as f:
            dados = json.load(f)
    except FileNotFoundError:
        dados = []
    dados.append({"titulo": titulo, "util": util, "query": query,
                  "ts": pd.Timestamp.now().isoformat()})
    with open(FEEDBACK_PATH, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)

def livros_penalizados() -> set:
    try:
        with open(FEEDBACK_PATH, "r", encoding="utf-8") as f:
            dados = json.load(f)
    except FileNotFoundError:
        return set()
    contagem = {}
    for item in dados:
        t = item["titulo"]
        contagem[t] = contagem.get(t, 0) + (1 if item["util"] else -1)
    return {t for t, s in contagem.items() if s < 0}

def ler_historico_feedback():
    try:
        with open(FEEDBACK_PATH, "r", encoding="utf-8") as f:
            dados = json.load(f)
    except FileNotFoundError:
        return []
    contagem = {}
    for item in dados:
        t = item["titulo"]
        contagem[t] = contagem.get(t, 0) + (1 if item["util"] else -1)
    return sorted([{"titulo": t, "score": s} for t, s in contagem.items()],
                  key=lambda x: -x["score"])

# ============================================================
# 9. HTML (design original do Streamlit preservado)
# ============================================================
HTML = r"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Recomendador Literário</title>
<link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Lato:wght@300;400;700&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background-color: #f4ecd8;
    font-family: 'Lato', 'Georgia', serif;
    color: #3d2b1f;
    min-height: 100vh;
    padding: 0 16px 60px;
  }

  .container {
    max-width: 740px;
    margin: 0 auto;
    padding-top: 48px;
  }

  /* ── Título ── */
  .titulo {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #5a4634;
    font-family: 'Playfair Display', 'Georgia', serif;
    margin-bottom: 8px;
  }
  .subtexto {
    text-align: center;
    font-size: 18px;
    color: #6e5c48;
    font-family: 'Playfair Display', 'Georgia', serif;
    margin-bottom: 32px;
  }

  /* ── Label + textarea ── */
  .label {
    font-size: 17px;
    font-weight: 600;
    color: #5a4634;
    margin-bottom: 8px;
    display: block;
  }
  textarea {
    width: 100%;
    height: 120px;
    border: 2px solid #8b5e3c !important;
    border-radius: 10px !important;
    padding: 12px 14px;
    font-size: 15px;
    font-family: 'Lato', sans-serif;
    background: #fdf8f0;
    color: #3d2b1f;
    resize: vertical;
    outline: none;
    transition: border-color .2s;
  }
  textarea:focus { border-color: #5a4634 !important; }

  /* ── Botões ── */
  .btn-row {
    display: flex;
    justify-content: space-between;
    margin-top: 14px;
  }
  button {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 8px 20px;
    font-size: 14px;
    font-family: 'Lato', sans-serif;
    border-radius: 8px;
    cursor: pointer;
    border: none;
    transition: opacity .15s, transform .1s;
    white-space: nowrap;
  }
  button:active { transform: scale(.97); }
  #btn-recomendar {
    background: #8b5e3c;
    color: #fff;
    font-weight: 700;
  }
  #btn-recomendar:hover { opacity: .88; }
  #btn-limpar {
    background: #e8dcc8;
    color: #5a4634;
    font-weight: 600;
  }
  #btn-limpar:hover { opacity: .80; }

  /* ── Spinner ── */
  #spinner {
    display: none;
    text-align: center;
    color: #6e5c48;
    font-style: italic;
    margin-top: 18px;
    font-size: 15px;
  }

  /* ── Divisor ── */
  hr { border: none; border-top: 1px solid #d6cbb5; margin: 28px 0; }

  /* ── Chip de emoção ── */
  .emocao-chip {
    display: inline-block;
    background: #d4a86a;
    color: #fff;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 13px;
    font-weight: 700;
    letter-spacing: .5px;
    margin-bottom: 16px;
    text-transform: uppercase;
  }

  /* ── Card do livro ── */
  .livro-card {
    background: #fdf8f0;
    padding: 22px 24px;
    border-radius: 12px;
    border: 1px solid #e0d6c3;
    box-shadow: 2px 2px 8px rgba(0,0,0,.05);
    font-family: 'Georgia', serif;
    margin-bottom: 14px;
    line-height: 1.65;
  }
  .livro-card b { color: #5a4634; }
  .livro-titulo {
    font-size: 20px;
    font-weight: 700;
    font-family: 'Playfair Display', serif;
    color: #3d2b1f;
    margin-bottom: 10px;
  }
  .badge {
    display: inline-block;
    background: #e8dcc8;
    color: #5a4634;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 12px;
    font-family: 'Lato', sans-serif;
    margin: 2px 4px 2px 0;
  }
  .match-badge {
    background: #8b5e3c;
    color: #fff;
  }

  /* ── Bloco de análise IA ── */
  .bloco-ia {
    background: #f9f3e8;
    border-left: 4px solid #8b5e3c;
    padding: 14px 18px;
    border-radius: 6px;
    margin-top: 14px;
    font-style: italic;
    color: #5a4634;
    font-family: 'Georgia', serif;
    line-height: 1.7;
  }
  .bloco-ia b { font-style: normal; }

  /* ── Botões de feedback ── */
  .feedback-row {
    display: flex;
    gap: 10px;
    margin-top: 14px;
  }
  .btn-fb {
    padding: 6px 16px;
    border-radius: 8px;
    font-size: 14px;
    cursor: pointer;
    border: 1.5px solid #c9b89a;
    background: transparent;
    color: #5a4634;
    transition: background .15s;
  }
  .btn-fb:hover { background: #e8dcc8; }
  .btn-fb.ativo-ok  { background: #d4edda; border-color: #4caf50; color: #256029; }
  .btn-fb.ativo-nok { background: #f8d7da; border-color: #e74c3c; color: #7b1d2a; }

  /* ── Demais livros (accordion) ── */
  details {
    margin-top: 10px;
  }
  summary {
    cursor: pointer;
    font-size: 15px;
    color: #5a4634;
    font-weight: 600;
    padding: 8px 0;
    list-style: none;
    display: flex;
    align-items: center;
    gap: 6px;
  }
  summary::-webkit-details-marker { display: none; }
  .outros-livros { margin-top: 12px; }
  .outro-item {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    padding: 10px 14px;
    border-radius: 8px;
    background: #fdf8f0;
    border: 1px solid #e0d6c3;
    margin-bottom: 8px;
    gap: 12px;
  }
  .outro-titulo { font-size: 14px; font-weight: 600; color: #3d2b1f; }
  .outro-meta   { font-size: 12px; color: #7a6a58; margin-top: 3px; }
  .outro-fb     { display: flex; gap: 6px; flex-shrink: 0; }
  .outro-fb button {
    padding: 4px 10px;
    font-size: 13px;
    border-radius: 6px;
    border: 1px solid #c9b89a;
    background: transparent;
    color: #5a4634;
    cursor: pointer;
  }
  .outro-fb button:hover { background: #e8dcc8; }

  /* ── Citação ── */
  .citacao {
    text-align: center;
    font-style: italic;
    color: #7a6a58;
    font-size: 14px;
    margin-top: 24px;
    font-family: 'Georgia', serif;
    line-height: 1.6;
  }

  /* ── Histórico ── */
  .historico-titulo {
    font-size: 18px;
    font-weight: 700;
    color: #5a4634;
    font-family: 'Playfair Display', serif;
    margin-bottom: 10px;
  }
  .hist-item { padding: 5px 0; font-size: 14px; color: #3d2b1f; }
  .hist-item .score { float: right; font-weight: 700; }
  .hist-ok  .score { color: #256029; }
  .hist-nok .score { color: #7b1d2a; }

  /* ── Rodapé ── */
  .rodape {
    text-align: center;
    font-size: 14px;
    color: #7a6a58;
    margin-top: 50px;
    padding-top: 20px;
    border-top: 1px solid #d6cbb5;
    font-family: 'Georgia', serif;
    line-height: 1.8;
  }
</style>
</head>
<body>
<div class="container">

  <!-- Título -->
  <p class="titulo">📖 Recomendador Literário</p>
  <p class="subtexto">Descubra histórias que combinam com o seu momento</p>

  <!-- Input -->
  <label class="label" for="input-texto">💬 Como você está se sentindo hoje?</label>
  <textarea id="input-texto" placeholder="Descreva seu estado emocional, o que busca ou o tipo de história que quer ler..."></textarea>

  <div class="btn-row">
    <button id="btn-recomendar" onclick="recomendar()">🔍 Recomendar</button>
    <button id="btn-limpar"     onclick="limpar()">🧹 Limpar</button>
  </div>

  <div id="spinner">📖 Analisando seu sentimento…</div>

  <!-- Resultado -->
  <div id="resultado"></div>

  <!-- Histórico de feedback -->
  <div id="historico-section"></div>

  <!-- Rodapé -->
  <div class="rodape">
    Projeto acadêmico desenvolvido para a disciplina de Processamento de Linguagem Natural<br>
    FATEC Cotia — Curso de Ciência de Dados<br><br>
    📖 Incentivando a leitura através da tecnologia
  </div>
</div>

<script>
let queryAtual = "";

function limpar() {
  document.getElementById("input-texto").value = "";
  document.getElementById("resultado").innerHTML = "";
  document.getElementById("historico-section").innerHTML = "";
}

async function recomendar() {
  const texto = document.getElementById("input-texto").value.trim();
  if (!texto) { alert("⚠️ Escreva algo antes de continuar."); return; }

  queryAtual = texto;
  document.getElementById("spinner").style.display = "block";
  document.getElementById("resultado").innerHTML = "";
  document.getElementById("historico-section").innerHTML = "";

  try {
    const resp = await fetch("/recomendar", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query: texto })
    });
    const data = await resp.json();
    document.getElementById("spinner").style.display = "none";

    if (data.erro) { document.getElementById("resultado").innerHTML = `<p style="color:red">${data.erro}</p>`; return; }

    renderResultado(data);
    carregarHistorico();
  } catch(e) {
    document.getElementById("spinner").style.display = "none";
    document.getElementById("resultado").innerHTML = `<p style="color:red">Erro de conexão: ${e}</p>`;
  }
}

function renderResultado(data) {
  const livros = data.livros;
  const emocao = data.emocao || "";
  const analiseIA = data.analise_ia || "";
  const top = livros[0];

  // Badge de emoção
  let html = `<hr>`;
  if (emocao) html += `<div class="emocao-chip">🧠 Emoção: ${emocao}</div>`;

  html += `<h3 style="font-family:'Playfair Display',serif;color:#5a4634;margin-bottom:14px;font-size:19px;">★ Livro #1 — Melhor recomendação para você</h3>`;

  // Card principal
  const matchPct = top._score_sem ? (top._score_sem * 100).toFixed(1) : "—";
  html += `<div class="livro-card">
    <div class="livro-titulo">📚 ${esc(top["Título do Livro"] || "")}</div>
    <span class="badge match-badge">Match ${matchPct}%</span>
    <span class="badge">${esc(top["Gênero"] || "")}</span>
    <span class="badge">${esc(top["Sentimento I"] || "")}</span>
    <span class="badge">${esc(top["Público-Alvo"] || "")}</span>
    <br><br>
    <b>💡 Sinopse:</b><br>${esc(top["Sinopse"] || "Sem descrição")}<br><br>
    <b>🎭 Emoções:</b> ${esc(top["Emoção I"] || "")}${top["Emoção II"] ? ", " + esc(top["Emoção II"]) : ""}
    &nbsp;|&nbsp;<b>📄 Páginas:</b> ${top["Páginas"] || "—"}
  </div>`;

  // Bloco IA
  if (analiseIA) {
    html += `<div class="bloco-ia">💛 <b>Por que este livro é o ideal para você?</b><br><br>${esc(analiseIA)}</div>`;
  }

  // Feedback do livro #1
  html += `<div class="feedback-row">
    <button class="btn-fb" id="fb-ok-0"  onclick="feedback(0, true)"  >👍 Foi útil</button>
    <button class="btn-fb" id="fb-nok-0" onclick="feedback(0, false)" >👎 Não foi útil</button>
  </div>`;

  // Demais livros
  if (livros.length > 1) {
    html += `<details><summary>📋 Ver todos os ${livros.length} livros recomendados</summary>
    <div class="outros-livros">`;
    livros.forEach((l, i) => {
      const pct = l._score_sem ? (l._score_sem * 100).toFixed(1) : "—";
      html += `<div class="outro-item">
        <div>
          <div class="outro-titulo">${i === 0 ? "★ " : (i+1) + ". "}${esc(l["Título do Livro"] || "")}</div>
          <div class="outro-meta">${esc(l["Gênero"] || "")} &nbsp;|&nbsp; ${esc(l["Sentimento I"] || "")} &nbsp;|&nbsp; Match ${pct}%</div>
        </div>
        <div class="outro-fb">
          <button onclick="feedback(${i}, true)"  title="Útil">👍</button>
          <button onclick="feedback(${i}, false)" title="Não útil">👎</button>
        </div>
      </div>`;
    });
    html += `</div></details>`;
  }

  // Citação
  html += `<div class="citacao">"Um leitor vive mil vidas antes de morrer…"<br>— George R.R. Martin</div>`;

  document.getElementById("resultado").innerHTML = html;
  window._livros = livros;
}

async function feedback(idx, util) {
  const livro = window._livros[idx];
  if (!livro) return;
  await fetch("/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ titulo: livro["Título do Livro"], util, query: queryAtual })
  });
  // Marcar botão
  const emoji = util ? "ok" : "nok";
  const outro  = util ? "nok" : "ok";
  const btn = document.getElementById(`fb-${emoji}-${idx}`);
  const btnOther = document.getElementById(`fb-${outro}-${idx}`);
  if (btn) { btn.classList.add(`ativo-${emoji}`); }
  if (btnOther) { btnOther.classList.remove(`ativo-ok`, `ativo-nok`); }
  carregarHistorico();
}

async function carregarHistorico() {
  const resp = await fetch("/historico");
  const data = await resp.json();
  if (!data.historico || data.historico.length === 0) return;
  let html = `<hr><div class="historico-titulo">📜 Histórico de avaliações</div>`;
  data.historico.forEach(item => {
    const cls = item.score > 0 ? "hist-ok" : (item.score < 0 ? "hist-nok" : "");
    const emoji = item.score > 0 ? "👍" : (item.score < 0 ? "👎" : "➖");
    html += `<div class="hist-item ${cls}">• ${esc(item.titulo)}<span class="score">${emoji} ${item.score > 0 ? "+" : ""}${item.score}</span></div>`;
  });
  document.getElementById("historico-section").innerHTML = html;
}

function esc(str) {
  return String(str)
    .replace(/&/g,"&amp;").replace(/</g,"&lt;")
    .replace(/>/g,"&gt;").replace(/"/g,"&quot;");
}
</script>
</body>
</html>
"""

# ============================================================
# 10. ROTAS FLASK
# ============================================================
@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/recomendar", methods=["POST"])
def rota_recomendar():
    body  = request.get_json(force=True)
    query = body.get("query", "").strip()
    if not query:
        return jsonify({"erro": "Query vazia."})

    # Remove livros penalizados antes
    penalizados = livros_penalizados()

    resultados = recomendar_livros(query, top_k=10)

    if not resultados.empty and penalizados:
        resultados = resultados[
            ~resultados["Título do Livro"].isin(penalizados)
        ].reset_index(drop=True)

    if resultados.empty:
        return jsonify({"erro": "Nenhum livro encontrado."})

    livros = resultados.to_dict(orient="records")

    # Análise IA do #1
    analise = analisar_top_livro(query, livros[0]) if livros else ""

    return jsonify({
        "livros":    livros,
        "emocao":    livros[0].get("Sentimento I", ""),
        "analise_ia": analise,
    })

@app.route("/feedback", methods=["POST"])
def rota_feedback():
    body = request.get_json(force=True)
    registrar_feedback(body.get("titulo",""), bool(body.get("util")), body.get("query",""))
    return jsonify({"ok": True})

@app.route("/historico")
def rota_historico():
    return jsonify({"historico": ler_historico_feedback()})

# ============================================================
# 11. INICIALIZAÇÃO (roda uma vez ao subir o servidor)
# ============================================================
print("Carregando dados do MongoDB...")
df = carregar_dados()
print(f"Dataset: {len(df)} livros")

print("Carregando modelo e embeddings...")
modelo, embeddings_livros = carregar_modelo_embeddings(df)

print("Treinando classificador...")
pipeline_clf = treinar_classificador(df)

print("\n✅ Tudo pronto! Acesse: http://localhost:5000\n")

if __name__ == "__main__":
    app.run(debug=False, port=5000)
