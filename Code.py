"""
Recomendador Literário — app.py
Junção do frontend Streamlit com o backend do notebook (NLP + embeddings + IA).

Instale antes de rodar:
    pip install streamlit pymongo sentence-transformers scikit-learn numpy pandas anthropic

Rode com:
    streamlit run app.py
"""

# ============================================================
# IMPORTS
# ============================================================
import os
import re
import json
import pickle
import textwrap
import unicodedata

import numpy as np
import pandas as pd
import streamlit as st

from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# ============================================================
# CONFIGURAÇÃO DA PÁGINA
# ============================================================
st.set_page_config(
    page_title="Recomendador Literário",
    page_icon="📖",
    layout="centered",
)

# ============================================================
# ESTILOS (frontend original intacto + melhorias visuais)
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400&family=Lato:wght@300;400;600&display=swap');

    .stApp { background-color: #f4ecd8; }

    .titulo {
        text-align: center;
        font-size: 42px;
        font-weight: bold;
        color: #5a4634;
        font-family: 'Playfair Display', Georgia, serif;
    }

    .subtexto {
        text-align: center;
        font-size: 18px;
        color: #6e5c48;
        font-family: 'Lato', Georgia, serif;
        margin-bottom: 8px;
    }

    textarea {
        border: 2px solid #8b5e3c !important;
        border-radius: 10px !important;
        font-family: 'Lato', serif !important;
    }

    .stButton > button {
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        gap: 6px !important;
        white-space: nowrap !important;
        width: auto !important;
        padding: 6px 14px !important;
        font-size: 14px !important;
        border-radius: 8px;
    }

    .livro-card {
        background: #fffdf7;
        padding: 22px 24px;
        border-radius: 14px;
        border: 1px solid #e0d6c3;
        box-shadow: 2px 4px 12px rgba(90,70,52,0.08);
        font-family: 'Lato', Georgia, serif;
        margin-bottom: 14px;
        line-height: 1.7;
    }

    .score-badge {
        display: inline-block;
        background: #8b5e3c;
        color: #f4ecd8;
        border-radius: 20px;
        padding: 2px 12px;
        font-size: 12px;
        font-family: 'Lato', sans-serif;
        letter-spacing: 0.5px;
        margin-left: 8px;
    }

    .analise-ia {
        background: linear-gradient(135deg, #fdf6e9 0%, #f9f0dd 100%);
        border-left: 4px solid #8b5e3c;
        padding: 18px 20px;
        border-radius: 8px;
        margin-top: 16px;
        font-style: italic;
        color: #5a4634;
        font-family: 'Playfair Display', Georgia, serif;
        font-size: 15px;
        line-height: 1.8;
    }

    .feedback-bar {
        background: #f0e8d8;
        border-radius: 10px;
        padding: 12px 18px;
        margin-top: 10px;
        border: 1px solid #ddd0bb;
    }

    .rank-item {
        padding: 10px 14px;
        border-radius: 8px;
        border: 1px solid #e0d6c3;
        margin-bottom: 8px;
        background: #fffdf7;
        font-family: 'Lato', sans-serif;
        font-size: 14px;
        color: #5a4634;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .rodape {
        text-align: center;
        font-size: 13px;
        color: #7a6a58;
        margin-top: 60px;
        padding-top: 20px;
        border-top: 1px solid #d6cbb5;
        font-family: 'Lato', sans-serif;
    }

    .tag {
        display: inline-block;
        background: #f0e8d8;
        border: 1px solid #c9b89a;
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 12px;
        color: #7a5c3c;
        margin: 2px;
        font-family: 'Lato', sans-serif;
    }

    div[data-testid="stSpinner"] { color: #8b5e3c; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# BACKEND — CONSTANTES E HELPERS
# ============================================================

MODELO_NOME  = "paraphrase-multilingual-MiniLM-L12-v2"
CACHE_PATH   = "embeddings_cache.pkl"
FEEDBACK_PATH = "feedback_usuario.json"

SINONIMOS = {
    "triste":      "melancólico saudade tristeza luto",
    "feliz":       "alegre otimista esperança eufórico",
    "tenso":       "ansioso suspense medo angústia",
    "épico":       "grandioso heróico batalha guerra",
    "romântico":   "amor romance paixão afeto",
    "intenso":     "emocionante forte dramático",
    "misterioso":  "suspense thriller enigma crime",
    "repensar":    "reflexão autoconhecimento mudança propósito",
    "vida":        "existência jornada propósito transformação",
    "crescimento": "desenvolvimento pessoal aprendizado evolução",
    "ansioso":     "angústia nervoso preocupação inquieto",
    "solitário":   "isolamento solidão introspecção contemplação",
}


def limpar_texto(texto: str) -> str:
    """Normaliza texto preservando caracteres unicode PT-BR."""
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
        "Sentimento I": 3, "Emoção I": 3,
        "Sentimento II": 2, "Emoção II": 2,
        "Gênero": 1,
    }
    partes = []
    for campo, peso in campos.items():
        val = limpar_texto(str(row.get(campo, "")))
        if val:
            partes.extend([val] * peso)
    return ' '.join(partes)


# ============================================================
# BACKEND — CONEXÃO E DADOS (cache Streamlit)
# ============================================================

@st.cache_resource(show_spinner="Conectando ao MongoDB...")
def carregar_dados():
    client = MongoClient("mongodb://localhost:27017", serverSelectionTimeoutMS=5000)
    try:
        client.admin.command("ping")
    except ConnectionFailure as e:
        st.error(f"❌ MongoDB não está rodando: {e}")
        st.stop()

    collection = client["admin"]["Book_Dataset_V2"]
    docs = list(collection.find())
    df = pd.DataFrame(docs).drop(columns=["_id"], errors="ignore")
    df = df.reset_index(drop=True)

    # Perfil emocional ponderado
    df["perfil_emocional"] = df.apply(construir_perfil, axis=1)

    # Rating normalizado
    scaler = MinMaxScaler()
    if "rating" in df.columns:
        df["rating_norm"] = scaler.fit_transform(
            pd.to_numeric(df["rating"], errors="coerce").fillna(0).values.reshape(-1, 1)
        ).flatten()
    else:
        df["rating_norm"] = 0.5

    return df


@st.cache_resource(show_spinner="Carregando modelo de linguagem...")
def carregar_modelo_e_embeddings(df):
    modelo = SentenceTransformer(MODELO_NOME)

    try:
        with open(CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
        if cache.get("modelo") == MODELO_NOME and len(cache["embeddings"]) == len(df):
            return modelo, cache["embeddings"]
        raise ValueError("Cache desatualizado")
    except Exception:
        pass

    embeddings = modelo.encode(
        df["perfil_emocional"].tolist(),
        show_progress_bar=False,
        batch_size=32,
        normalize_embeddings=True,
    )
    with open(CACHE_PATH, "wb") as f:
        pickle.dump({"modelo": MODELO_NOME, "embeddings": embeddings}, f)
    return modelo, embeddings


@st.cache_resource(show_spinner="Treinando classificador...")
def treinar_classificador(df):
    df_clf = df[["Gênero", "Emoção II", "Sentimento I"]].dropna()
    contagem = df_clf["Sentimento I"].value_counts()
    classes_validas = contagem[contagem >= 2].index
    df_clf = df_clf[df_clf["Sentimento I"].isin(classes_validas)]

    X = df_clf[["Gênero", "Emoção II"]].apply(
        lambda r: limpar_texto(' '.join(r.astype(str))), axis=1
    )
    y = df_clf["Sentimento I"]

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=5000)),
        ("clf",   LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")),
    ])
    pipeline.fit(X, y)
    return pipeline


# ============================================================
# BACKEND — MMR
# ============================================================

def mmr_rerank(query_emb, candidatos_idx, embeddings, top_k=10, lambda_param=0.7):
    selecionados = []
    candidatos = list(candidatos_idx)

    for _ in range(min(top_k, len(candidatos))):
        if not candidatos:
            break

        sims_query = cosine_similarity(
            query_emb.reshape(1, -1), embeddings[candidatos]
        )[0]

        sims_sel = (
            cosine_similarity(embeddings[candidatos], embeddings[selecionados]).max(axis=1)
            if selecionados else np.zeros(len(candidatos))
        )

        mmr_scores = lambda_param * sims_query - (1 - lambda_param) * sims_sel
        melhor = int(np.argmax(mmr_scores))
        selecionados.append(candidatos[melhor])
        candidatos.pop(melhor)

    return selecionados


# ============================================================
# BACKEND — RECOMENDAÇÃO
# ============================================================

def recomendar_livros(query, df, modelo, embeddings_livros,
                      top_k=10, peso_semantico=0.7, peso_rating=0.3,
                      usar_mmr=True, lambda_mmr=0.7, pre_filtro=50):

    query_expandida = expandir_query(query)
    emb_query = modelo.encode([query_expandida], normalize_embeddings=True)

    sims = cosine_similarity(emb_query, embeddings_livros)[0]
    score_final = peso_semantico * sims + peso_rating * df["rating_norm"].values

    df = df.copy()
    df["_score_sem"]   = sims
    df["_score_final"] = score_final

    top_idx = np.argsort(score_final)[::-1][:pre_filtro].tolist()

    if usar_mmr and len(top_idx) > top_k:
        idx_final = mmr_rerank(emb_query[0], top_idx, embeddings_livros,
                               top_k=top_k, lambda_param=lambda_mmr)
    else:
        idx_final = top_idx[:top_k]

    resultado = df.iloc[idx_final].copy()

    def gerar_explicacao(row):
        sem_pct = round(row["_score_sem"] * 100, 1)
        sent = row.get("Sentimento I", "") or row.get("Emoção I", "")
        genero = row.get("Gênero", "")
        return f"Match {sem_pct}% — {sent}, {genero}"

    resultado["explicacao"] = resultado.apply(gerar_explicacao, axis=1)

    # Fallback
    if resultado["_score_sem"].max() < 0.30 and pipeline_clf is not None:
        emocao_pred = pipeline_clf.predict([limpar_texto(query)])[0]
        fallback = df[df["Sentimento I"].str.lower() == emocao_pred.lower()].nlargest(top_k, "rating_norm")
        if not fallback.empty:
            return fallback

    return resultado.reset_index(drop=True)


# ============================================================
# BACKEND — ANÁLISE IA (Claude API)
# ============================================================

def analisar_top_livro(query: str, livro: dict) -> str:
    try:
        import anthropic
    except ImportError:
        return "⚠️ Biblioteca 'anthropic' não instalada. Rode: pip install anthropic"

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return "⚠️ ANTHROPIC_API_KEY não definida nas variáveis de ambiente."

    client = anthropic.Anthropic(api_key=api_key)

    campos = {
        "Título":        livro.get("Título do Livro", "N/A"),
        "Autor":         livro.get("Autor", "N/A"),
        "Gênero":        livro.get("Gênero", "N/A"),
        "Sentimento I":  livro.get("Sentimento I", "N/A"),
        "Sentimento II": livro.get("Sentimento II", "N/A"),
        "Emoção I":      livro.get("Emoção I", "N/A"),
        "Emoção II":     livro.get("Emoção II", "N/A"),
        "Sinopse":       livro.get("Sinopse", "N/A"),
        "Público-Alvo":  livro.get("Público-Alvo", "N/A"),
        "Match":         f"{round(livro.get('_score_sem', 0) * 100, 1)}%",
    }
    livro_str = "\n".join(f"  {k}: {v}" for k, v in campos.items())

    prompt = f"""Você é um especialista em literatura e psicologia emocional.

O usuário descreveu seu estado emocional assim:
  "{query}"

O sistema selecionou este livro como #1:
{livro_str}

Escreva UMA análise curta (3 a 5 frases), em português, explicando de forma
pessoal e empática por que este livro é a melhor escolha para o momento do usuário.

Comece exatamente com: "Esse livro se encaixa com você porque..."

Conecte os sentimentos e emoções do livro com as palavras do usuário.
Sem bullet points. Apenas prosa fluida e calorosa."""

    try:
        resposta = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return resposta.content[0].text.strip()
    except Exception as e:
        return f"⚠️ Erro na API Claude: {e}"


# ============================================================
# BACKEND — FEEDBACK
# ============================================================

def registrar_feedback(titulo_livro: str, util: bool, query: str) -> None:
    try:
        with open(FEEDBACK_PATH, "r", encoding="utf-8") as f:
            dados = json.load(f)
    except FileNotFoundError:
        dados = []

    dados.append({
        "titulo": titulo_livro,
        "util":   util,
        "query":  query,
        "ts":     pd.Timestamp.now().isoformat(),
    })

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


def historico_feedback() -> dict:
    try:
        with open(FEEDBACK_PATH, "r", encoding="utf-8") as f:
            dados = json.load(f)
    except FileNotFoundError:
        return {}

    contagem = {}
    for item in dados:
        t = item["titulo"]
        contagem[t] = contagem.get(t, 0) + (1 if item["util"] else -1)
    return contagem


# ============================================================
# CARREGAMENTO (roda uma vez, fica em cache)
# ============================================================

df             = carregar_dados()
modelo, embs   = carregar_modelo_e_embeddings(df)
pipeline_clf   = treinar_classificador(df)


# ============================================================
# UI — CABEÇALHO
# ============================================================

st.markdown('<p class="titulo">📖 Recomendador Literário</p>', unsafe_allow_html=True)
st.markdown('<p class="subtexto">Descubra histórias que combinam com o seu momento</p>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# UI — INPUT E BOTÕES
# ============================================================

def limpar_campos():
    st.session_state["input_texto"] = ""
    st.session_state.pop("resultados", None)
    st.session_state.pop("analise_ia", None)
    st.toast("🧹 Campos limpos!")

st.markdown("### 💬 Como você está se sentindo hoje?")
texto_usuario = st.text_area(
    "Digite aqui:",
    height=120,
    key="input_texto",
    placeholder="Ex: estou ansioso e preciso de algo que me acalme..."
)

col1, col_spacer, col2 = st.columns([1, 4, 1])
with col1:
    recomendar = st.button("🔍 Recomendar")
with col2:
    st.button("🧹 Limpar", on_click=limpar_campos)


# ============================================================
# UI — AÇÃO PRINCIPAL
# ============================================================

if recomendar:
    if not texto_usuario.strip():
        st.warning("⚠️ Escreva algo antes de continuar.")
    else:
        penalizados = livros_penalizados()

        with st.spinner("📖 Analisando seu sentimento e buscando livros..."):
            resultados = recomendar_livros(
                texto_usuario, df, modelo, embs,
                top_k=10, peso_semantico=0.7, peso_rating=0.3,
                usar_mmr=True, lambda_mmr=0.7,
            )
            # Remove penalizados
            resultados = resultados[
                ~resultados["Título do Livro"].isin(penalizados)
            ].reset_index(drop=True)

        st.session_state["resultados"]   = resultados
        st.session_state["query_atual"]  = texto_usuario
        st.session_state.pop("analise_ia", None)


# ============================================================
# UI — EXIBIÇÃO DOS RESULTADOS
# ============================================================

if "resultados" in st.session_state:
    resultados  = st.session_state["resultados"]
    query_atual = st.session_state.get("query_atual", "")

    st.markdown("---")

    # ── Livro #1 em destaque ─────────────────────────────────
    top1 = resultados.iloc[0]
    titulo_top1 = top1.get("Título do Livro", "")
    autor_top1  = top1.get("Autor", "")
    score_top1  = round(top1.get("_score_sem", 0) * 100, 1)

    st.markdown(f"""
    <div class="livro-card">
        <div style="display:flex; justify-content:space-between; align-items:flex-start;">
            <div>
                <span style="font-size:11px; letter-spacing:2px; color:#9a7a5a; font-family:'Lato',sans-serif; text-transform:uppercase;">★ Melhor recomendação</span><br>
                <span style="font-size:22px; font-family:'Playfair Display',serif; color:#3a2a1a; font-weight:700;">{titulo_top1}</span>
                {"<br><span style='font-size:14px; color:#7a5c3c; font-family:Lato,sans-serif;'>por " + autor_top1 + "</span>" if autor_top1 else ""}
            </div>
            <span class="score-badge">Match {score_top1}%</span>
        </div>
        <br>
        {"<b>📖 Sinopse:</b><br>" + str(top1.get('Sinopse','')) + "<br><br>" if top1.get('Sinopse') else ""}
        <div>
            {"<span class='tag'>" + str(top1.get('Gênero','')) + "</span>" if top1.get('Gênero') else ""}
            {"<span class='tag'>" + str(top1.get('Sentimento I','')) + "</span>" if top1.get('Sentimento I') else ""}
            {"<span class='tag'>" + str(top1.get('Emoção I','')) + "</span>" if top1.get('Emoção I') else ""}
            {"<span class='tag'>" + str(top1.get('Público-Alvo','')) + "</span>" if top1.get('Público-Alvo') else ""}
            {"<span class='tag'>" + str(top1.get('Páginas','')) + " pág.</span>" if top1.get('Páginas') else ""}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Análise IA ────────────────────────────────────────────
    col_ia, _ = st.columns([2, 1])
    with col_ia:
        if st.button("✨ Gerar análise personalizada com IA"):
            with st.spinner("🤖 Gerando análise..."):
                analise = analisar_top_livro(query_atual, top1.to_dict())
            st.session_state["analise_ia"] = analise

    if "analise_ia" in st.session_state:
        st.markdown(f"""
        <div class="analise-ia">
            💛 <b>Por que este livro para o seu momento?</b><br><br>
            {st.session_state['analise_ia']}
        </div>
        """, unsafe_allow_html=True)

    # ── Feedback livro #1 ─────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="feedback-bar">Esta recomendação foi útil?</div>',
                unsafe_allow_html=True)
    col_ok, col_no, _ = st.columns([1, 1, 4])
    with col_ok:
        if st.button("👍 Sim"):
            registrar_feedback(titulo_top1, True, query_atual)
            st.toast(f"👍 Obrigado pelo feedback!")
    with col_no:
        if st.button("👎 Não"):
            registrar_feedback(titulo_top1, False, query_atual)
            st.toast(f"👎 Anotado! Não mostraremos mais este.")

    # ── Top 2–10 ─────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("📚 Ver todos os 10 livros recomendados"):
        for i, row in resultados.iterrows():
            titulo = str(row.get("Título do Livro", ""))
            autor  = str(row.get("Autor", ""))
            score  = round(row.get("_score_sem", 0) * 100, 1)
            genero = str(row.get("Gênero", ""))
            sent   = str(row.get("Sentimento I", ""))

            st.markdown(f"""
            <div class="rank-item">
                <span style="font-size:18px; font-weight:700; color:#c9a84c; min-width:28px;">#{i+1}</span>
                <div style="flex:1;">
                    <span style="font-weight:600; font-family:'Playfair Display',serif;">{titulo}</span>
                    {"<span style='color:#9a7a5a; font-size:12px; margin-left:6px;'>· " + autor + "</span>" if autor else ""}
                    <br>
                    <span style="font-size:12px; color:#9a7a5a;">{genero} · {sent}</span>
                </div>
                <span class="score-badge">{score}%</span>
            </div>
            """, unsafe_allow_html=True)

            # Feedback individual para cada livro
            col_s, col_n, _ = st.columns([1, 1, 8])
            with col_s:
                if st.button("👍", key=f"ok_{i}"):
                    registrar_feedback(titulo, True, query_atual)
                    st.toast(f"👍 {titulo[:30]}...")
            with col_n:
                if st.button("👎", key=f"no_{i}"):
                    registrar_feedback(titulo, False, query_atual)
                    st.toast(f"👎 Anotado!")

    # ── Citação ───────────────────────────────────────────────
    st.markdown("""
    > *"Um leitor vive mil vidas antes de morrer..."*  
    > — George R.R. Martin
    """)


# ============================================================
# UI — HISTÓRICO DE RECOMENDAÇÕES DA SESSÃO
# ============================================================

if "resultados" in st.session_state:
    hist = historico_feedback()
    if hist:
        st.markdown("---")
        with st.expander("📊 Histórico de feedback"):
            for titulo, score in sorted(hist.items(), key=lambda x: -x[1]):
                emoji = "👍" if score > 0 else "👎"
                st.write(f"{emoji} {titulo[:60]} ({score:+d})")


# ============================================================
# UI — RODAPÉ
# ============================================================

st.markdown("""
<div class="rodape">
    Projeto acadêmico desenvolvido para a disciplina de Processamento de Linguagem Natural<br>
    FATEC Cotia — Curso de Ciência de Dados<br><br>
    📖 Incentivando a leitura através da tecnologia
</div>
""", unsafe_allow_html=True)
