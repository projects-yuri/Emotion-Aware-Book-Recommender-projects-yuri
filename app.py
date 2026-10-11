"""Emotion Book — Streamlit nativo, recomendação semântica FastEmbed/ONNX.
Configure MONGO_URI em Streamlit Cloud > Settings > Secrets.
"""
import hashlib
import html
import os
import re
from urllib.parse import quote, urlparse

# Limita concorrência numérica em ambientes de memória restrita.
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import streamlit as st
from pymongo import MongoClient

st.set_page_config(page_title="Emotion Book · Descubra sua próxima leitura", page_icon="📚", layout="wide", initial_sidebar_state="expanded")

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DB_NAME = os.getenv("MONGO_DB", "dataset")
COLLECTION_NAME = os.getenv("MONGO_COL", "dataset")

STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@500;600;700&display=swap');
html,body,[class*='css'],[data-testid='stApp']{font-family:'DM Sans',sans-serif}
[data-testid='stApp']{background:linear-gradient(135deg,#faf7f4 0%,#f2f3f8 55%,#f8f5fb 100%);color:#282b39}
[data-testid='stSidebar']{background:#f4edf4}
h1,h2,h3{font-family:'Playfair Display',serif!important;color:#35304d!important}
.hero{padding:1.6rem 2rem 1.7rem;background:linear-gradient(120deg,#eae1f2,#f6e8e4,#e1efec);border-radius:25px;margin-bottom:1.2rem;border:1px solid #e8dfe9}
.hero h1{font-family:'Playfair Display',serif;font-size:2.5rem;margin:.1rem 0 .4rem;color:#302943}
.hero p{font-size:1.04rem;color:#5e6170;max-width:730px}
.kicker{letter-spacing:.18em;font-weight:700;font-size:.72rem;color:#876590}
.book{background:rgba(255,255,255,.82);border:1px solid #e8e2ec;border-radius:18px;padding:1.25rem;min-height:290px;box-shadow:0 7px 26px rgba(58,47,78,.04);margin-bottom:.7rem}
.book .genre{color:#997b99;font-size:.77rem;font-weight:700;text-transform:uppercase;letter-spacing:.07em}
.book h3{font-family:'Playfair Display',serif;font-size:1.26rem;margin:.7rem 0 .3rem;line-height:1.3}
.book .author{font-size:.86rem;color:#6e6878;margin-bottom:.85rem}
.book .summary{font-size:.89rem;color:#505264;line-height:1.55}
.book .tags{margin-top:1rem;font-size:.78rem;color:#695681}
.metricline{color:#737083;font-size:.89rem;margin-bottom:1.2rem}
[data-testid='stButton'] button[kind='primary']{background:#846a9e;border:0;border-radius:12px}
[data-testid='stButton'] button{border-radius:11px}
@media (prefers-color-scheme: dark){[data-testid='stApp']{background:#1b1c26;color:#efecf5}[data-testid='stSidebar']{background:#272431} h1,h2,h3{color:#f3edf8!important}.book{background:#292737;border-color:#494153}.book .summary,.book .author,.metricline{color:#c9c4d5}.hero{background:linear-gradient(120deg,#383044,#483340,#293c3a);border-color:#524455}.hero h1,.hero p{color:#f0edf5}}
</style>
"""
st.markdown(STYLES, unsafe_allow_html=True)


def get_uri():
    uri = os.environ.get("MONGO_URI", "")
    if not uri:
        try:
            uri = st.secrets.get("MONGO_URI", "")
        except Exception:
            pass
    return uri


@st.cache_resource(show_spinner=False)
def mongo_client(uri):
    return MongoClient(uri, serverSelectionTimeoutMS=9000, connectTimeoutMS=9000, maxPoolSize=5)


@st.cache_data(ttl=900, max_entries=1, show_spinner=False)
def load_catalog():
    uri = get_uri()
    if not uri:
        raise RuntimeError("Defina MONGO_URI em Manage app → Settings → Secrets.")
    client = mongo_client(uri)
    client.admin.command("ping")
    projection = {"_id": 0}
    return list(client[DB_NAME][COLLECTION_NAME].find({}, projection))


def text(value):
    return str(value).strip() if value is not None else ""


def summary(book):
    # Não enviar sinopses inteiras quando o texto é muito longo.
    fields = ["Título do Livro", "Gênero", "Sentimento I", "Sentimento II", "Emoção I", "Emoção II", "Sinopse"]
    return " | ".join(text(book.get(f))[:580] for f in fields if text(book.get(f)))[:1150]


def cache_key(books):
    h = hashlib.sha256()
    for b in books:
        h.update(summary(b).encode("utf-8", errors="replace"))
        h.update(b"\0")
    return h.hexdigest()


@st.cache_resource(show_spinner=False)
def embedding_model():
    # Carregamento sob demanda — nunca carrega PyTorch ou SentenceTransformer.
    from fastembed import TextEmbedding
    return TextEmbedding(model_name=MODEL_NAME, threads=2, parallel=1, cache_dir="/tmp/fastembed_models")


@st.cache_data(max_entries=1, show_spinner=False)
def catalog_vectors(_books, dataset_key):
    model = embedding_model()
    texts = [summary(b) for b in _books]
    vectors = np.asarray(list(model.embed(texts, batch_size=8)), dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-8)


def safe_url(b):
    raw = text(b.get("url"))
    if not raw:
        return "https://openlibrary.org/search?q=" + quote(text(b.get("Título do Livro")))
    if not raw.startswith(("https://", "http://")):
        raw = "https://" + raw
    p = urlparse(raw)
    if p.scheme not in ("https", "http") or not p.netloc:
        return "https://openlibrary.org/search?q=" + quote(text(b.get("Título do Livro")))
    return raw


def tags_for(b):
    return {text(b.get(k)).casefold() for k in ("Emoção I", "Emoção II", "Sentimento I", "Sentimento II") if text(b.get(k))}


def pick(books, q, emotion, genre, limit_pages, min_rating, n, variety, mode):
    eligible = []
    for i, b in enumerate(books):
        if genre != "Todos" and text(b.get("Gênero")) != genre:
            continue
        pages = b.get("Páginas")
        try:
            pages = int(pages) if pages is not None else None
        except (ValueError, TypeError):
            pages = None
        if limit_pages and pages is not None and pages > limit_pages:
            continue
        try:
            rating = float(b.get("rating") or 0)
        except (ValueError, TypeError):
            rating = 0
        if min_rating > 0 and rating < min_rating:
            continue
        eligible.append(i)
    if not eligible:
        return [], None

    vector_query = " ".join(t for t in [q, emotion if emotion != "Qualquer emoção" else ""] if t).strip()
    if not vector_query:
        vector_query = "livro envolvente emocionante interessante"
    if mode == "Semântica (IA)":
        model = embedding_model()
        vectors = catalog_vectors(books, cache_key(books))
        query_emb = np.asarray(next(iter(model.embed([vector_query]))), dtype=np.float32)
        query_emb /= max(float(np.linalg.norm(query_emb)), 1e-8)
        sim = vectors @ query_emb
    else:
        # Modo econômico sem download de pesos de IA — mantém busca textual como contingência.
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        docs = [summary(b) for b in books]
        features = TfidfVectorizer(strip_accents="unicode", ngram_range=(1, 2), max_features=12000).fit_transform(docs + [vector_query])
        sim = cosine_similarity(features[:-1], features[-1]).ravel().astype(np.float32)
        vectors = None
    scores = sim.copy()
    for i, b in enumerate(books):
        if emotion != "Qualquer emoção" and emotion.casefold() in tags_for(b):
            scores[i] += 0.30
        try:
            rating = float(b.get("rating") or 0)
            scores[i] += 0.045 * min(max(rating / 50, 0), 1)
        except (ValueError, TypeError):
            pass
    order = sorted(eligible, key=lambda i: float(scores[i]), reverse=True)[:max(40, n * 8)]
    chosen = []
    while order and len(chosen) < n:
        if not chosen or variety == 0 or vectors is None:
            idx = order[0]
        else:
            idx = max(order, key=lambda k: (1 - variety) * float(scores[k]) - variety * max(float(vectors[k] @ vectors[c]) for c in chosen))
        chosen.append(idx)
        order.remove(idx)
    return [(books[i], float(scores[i])) for i in chosen], mode


def render_book(book, score, key):
    title = html.escape(text(book.get("Título do Livro")) or "Livro sem título")
    author = html.escape(text(book.get("Autor")) or "Autor não informado")
    genre = html.escape(text(book.get("Gênero")) or "Literatura")
    desc = html.escape(text(book.get("Sinopse")) or "Sem sinopse disponível.")
    if len(desc) > 400:
        desc = desc[:397].rsplit(" ", 1)[0] + "…"
    tags = " · ".join(html.escape(text(book.get(t))) for t in ("Emoção I", "Emoção II") if text(book.get(t)))
    st.markdown(f'<div class="book"><div class="genre">{genre}</div><h3>{title}</h3><div class="author">{author}</div><div class="summary">{desc}</div><div class="tags">✦ {tags}</div></div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.link_button("Conhecer livro ↗", safe_url(book), use_container_width=True)
    with c2:
        ident = f"{title}|{author}"
        if st.button("♡ Salvar" if ident not in st.session_state.saved else "♥ Salvo", key=f"save-{key}", use_container_width=True):
            if ident in st.session_state.saved:
                st.session_state.saved.remove(ident)
            else:
                st.session_state.saved.add(ident)
            st.rerun()


if "saved" not in st.session_state:
    st.session_state.saved = set()
if "results" not in st.session_state:
    st.session_state.results = []

st.markdown('<div class="hero"><div class="kicker">EMOTION BOOK · SUA ESTANTE EMOCIONAL</div><h1>Encontre histórias que combinam com você.</h1><p>Conte o que está sentindo e receba recomendações de livros guiadas por emoções, afinidade semântica e seus gostos.</p></div>', unsafe_allow_html=True)

try:
    books = load_catalog()
except Exception as exc:
    st.error(f"Não foi possível carregar o catálogo do MongoDB: {type(exc).__name__}.")
    st.info("Confira o Secret MONGO_URI, a disponibilidade do Atlas e os nomes MONGO_DB/MONGO_COL.")
    st.stop()

if not books:
    st.warning("A coleção não contém documentos.")
    st.stop()

all_genres = sorted({text(b.get("Gênero")) for b in books if text(b.get("Gênero"))})
all_emotions = sorted({text(b.get(k)) for b in books for k in ("Emoção I", "Emoção II", "Sentimento I", "Sentimento II") if text(b.get(k))})

with st.sidebar:
    st.markdown("## ✦ Seu momento")
    query = st.text_area("Como você está se sentindo?", placeholder="Ex.: Estou ansioso e procuro uma aventura acolhedora, com esperança e amizade.", height=110)
    emotion = st.selectbox("Emoção / sentimento", ["Qualquer emoção"] + all_emotions)
    genre = st.selectbox("Gênero literário", ["Todos"] + all_genres)
    number = st.slider("Quantos livros?", 2, 12, 6, 2)
    max_pages = st.slider("Máximo de páginas (0 = sem filtro)", 0, 1000, 0, 50)
    min_rating = st.slider("Nota mínima (0 = sem filtro)", 0, 50, 0, 5)
    diversity = st.slider("Diversidade das recomendações", 0.0, 0.65, 0.22, 0.05)
    mode = st.radio("Mecanismo de recomendação", ["Semântica (IA)", "Econômico (TF-IDF)"], help="IA usa o mesmo modelo multilíngue original, com execução ONNX; econômico é alternativa para limites de memória.")
    search = st.button("✨ Descobrir leituras", type="primary", use_container_width=True)
    st.divider()
    st.caption("Dados privados do Atlas são consultados pelo servidor. Favoritos ficam somente nesta sessão.")
    if st.button("↻ Atualizar catálogo", use_container_width=True):
        load_catalog.clear()
        st.session_state.results = []
        st.rerun()

st.markdown(f'<div class="metricline">📖 <b>{len(books)}</b> livros disponíveis &nbsp; · &nbsp; 🎭 <b>{len(all_emotions)}</b> emoções e sentimentos &nbsp; · &nbsp; 📚 <b>{len(all_genres)}</b> gêneros</div>', unsafe_allow_html=True)

if search:
    try:
        with st.spinner("Encontrando histórias que combinam com seu momento…" if mode == "Semântica (IA)" else "Analisando seu perfil de leitura…"):
            st.session_state.results, st.session_state.last_mode = pick(books, query, emotion, genre, max_pages, min_rating, number, diversity, mode)
    except Exception as exc:
        st.session_state.results = []
        st.error(f"Não foi possível executar o mecanismo {mode}: {type(exc).__name__}.")
        st.info("Se for falta de memória ou falha ao baixar o modelo, selecione 'Econômico (TF-IDF)' no menu lateral e tente novamente.")

if st.session_state.results:
    st.subheader("📚 Sua seleção personalizada")
    st.caption(f"Ordenação: {st.session_state.get('last_mode', 'IA')}. A pontuação é de afinidade interna, não uma avaliação pública dos livros.")
    items = st.session_state.results
    for start in range(0, len(items), 3):
        columns = st.columns(3, gap="medium")
        for offset, (book, score) in enumerate(items[start:start + 3]):
            with columns[offset]:
                render_book(book, score, f"{start+offset}-{text(book.get('id'))}")
else:
    st.markdown("### 🌷 Uma leitura para cada emoção")
    st.write("Descreva um momento ou escolha um sentimento na barra lateral para descobrir livros de acordo com o seu perfil.")
    st.info("O modelo semântico só é carregado após a primeira recomendação. A primeira execução pode demorar enquanto o modelo ONNX é baixado.")

with st.expander(f"♥ Minha estante ({len(st.session_state.saved)})"):
    if st.session_state.saved:
        titles = sorted(st.session_state.saved)
        st.write("\n\n".join(f"• {t.split('|')[0]}" for t in titles))
        st.download_button("Exportar estante TXT", "\n".join(titles).encode("utf-8"), file_name="minha_estante_emotion_book.txt", mime="text/plain")
    else:
        st.caption("Seus livros salvos nesta sessão aparecerão aqui.")
