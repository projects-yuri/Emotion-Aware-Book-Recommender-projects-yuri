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

st.set_page_config(page_title="Emotional Book · Histórias para o seu momento", page_icon="📖", layout="wide", initial_sidebar_state="collapsed")

MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DB_NAME = os.getenv("MONGO_DB", "dataset")
COLLECTION_NAME = os.getenv("MONGO_COL", "dataset")

STYLES = """
<style>
:root{--bg:#17231d;--surface:#223129;--surface-2:#2b3c32;--ink:#eef2e9;--muted:#b9c5ba;--line:#405447;--accent:#bad3ad;--sage:#324536;--lilac:#40394b;--pink:#493936;--yellow:#46432d}
.stApp{background:var(--bg);color:var(--ink);font-family:'Segoe UI',Arial,sans-serif}
[data-testid="stHeader"]{background:rgba(23,35,29,.96)}
[data-testid="stMainBlockContainer"]{max-width:1190px;padding-top:2.1rem;padding-bottom:4rem}
[data-testid="stSidebar"]{background:#1d2b23;border-right:1px solid var(--line)}
[data-testid="stSidebar"] h2{font-family:Georgia,serif!important;color:var(--ink)}
h1,h2,h3{font-family:Georgia,serif!important;color:var(--ink)!important;font-weight:400!important}
hr{border-color:var(--line)!important}
.brand{display:flex;align-items:center;gap:12px;padding:8px 0 22px;border-bottom:1px solid var(--line);margin-bottom:25px}
.brand-icon{background:#b9d1ad;color:#1d3026;border-radius:13px;padding:9px 12px;font-size:23px}
.brand-name{font:23px Georgia,serif;color:var(--ink)}
.brand-sub{font-size:9px;letter-spacing:2px;color:var(--muted);margin-top:4px;font-weight:700}
.hero-main{padding:29px 4px 25px}.eyebrow{color:var(--accent);font-size:11px;letter-spacing:2px;font-weight:700}
.hero-main h1{font:normal clamp(36px,4.2vw,54px)/1.12 Georgia,serif!important;letter-spacing:-1.7px;margin:16px 0}
.hero-main h1 em{color:var(--accent);font-weight:normal}.hero-main p{color:var(--muted);font-size:14px;line-height:1.8;max-width:510px}
.hero-art{height:255px;position:relative;display:flex;align-items:center;justify-content:center;isolation:isolate}
.hero-orbit{width:310px;height:210px;position:absolute;background:var(--sage);border-radius:50%;transform:rotate(-11deg);z-index:-1}
.hero-orbit:after{content:'';position:absolute;inset:-12px 15px 10px -17px;border:1px solid #bccaba;border-radius:50%;transform:rotate(17deg)}
.hero-book{position:absolute;width:122px;height:163px;border:1px solid rgba(0,0,0,.09);border-left:8px solid rgba(0,0,0,.09);border-radius:3px 9px 9px 3px;box-shadow:7px 10px 20px #35483b29;padding:20px 13px;text-align:center;display:flex;align-items:center;justify-content:center;flex-direction:column;font:20px Georgia,serif;color:#4d5141}
.hero-book.one{background:#d6d8b6;transform:translate(-83px,4px) rotate(-16deg)}.hero-book.two{background:#ded4e9;transform:translate(70px,-7px) rotate(13deg)}.hero-book.three{background:#ebd4c6;transform:translate(-5px,21px) rotate(-4deg)}
.hero-caption{position:absolute;right:3%;bottom:1%;background:var(--surface);border:1px solid var(--line);border-radius:14px;padding:11px 17px;font:italic 12px Georgia,serif}
.panel-head{font:27px Georgia,serif;color:var(--ink);margin-bottom:4px}.step{font-size:10px;letter-spacing:1.6px;text-transform:uppercase;color:var(--muted);font-weight:700}
.panel-caption{font-size:12px;color:var(--muted);margin-bottom:14px;line-height:1.7}
.panel-box{padding:19px 20px;border:1px solid var(--line);background:var(--surface);border-radius:21px}
.side-panel{background:var(--surface);border:1px solid var(--line);border-radius:20px;padding:22px;margin-bottom:15px}
.side-panel h3{font:21px Georgia,serif;color:var(--ink);margin:8px 0}.side-panel p{font-size:12px;color:var(--muted);line-height:1.8}
.shelf-note{background:var(--sage);border:1px solid var(--line);border-radius:20px;padding:20px}.shelf-number{font:32px Georgia,serif;color:var(--ink)}
[data-testid="stVerticalBlockBorderWrapper"] > div{border-color:var(--line)!important}
div[data-testid="stForm"]{border:1px solid var(--line)!important;background:var(--surface);border-radius:22px;padding:25px}
[data-testid="stWidgetLabel"] p{font-size:12px!important;color:var(--ink)!important;font-weight:600}
.stTextArea textarea,.stTextInput input,[data-baseweb="select"]>div{background:#1e2c24!important;border-color:var(--line)!important;border-radius:11px!important}
.stButton button[kind="primary"],div[data-testid="stFormSubmitButton"] button{background:var(--accent);color:#18251d!important;border:1px solid var(--accent);border-radius:11px;min-height:45px}
.stButton button, .stDownloadButton button, .stLinkButton a{border-radius:11px;border-color:var(--line);font-size:12px}
.stButton button:hover,.stDownloadButton button:hover,.stLinkButton a:hover{border-color:var(--accent);color:var(--accent)}
[data-testid="stTabs"] button{font-size:13px!important;color:var(--muted)!important}
[data-testid="stTabs"] button[aria-selected="true"]{color:var(--accent)!important}
.book{background:var(--surface);border:1px solid var(--line);border-radius:19px;overflow:hidden;margin-bottom:4px;box-shadow:0 8px 28px #3548370a}
.cover-area{height:183px;display:grid;place-items:center;position:relative;background:var(--coverbg);overflow:hidden;color:var(--coverink)}
.cover-area:before{content:'';width:205px;height:205px;position:absolute;border:1px solid currentColor;opacity:.12;border-radius:50%;left:-30px;top:13px}
.cover-area:after{content:'';position:absolute;width:125px;height:125px;border:1px solid currentColor;opacity:.12;border-radius:50%;right:-12px;top:-49px}
.cover-object{position:relative;z-index:1;width:105px;height:140px;background:var(--cover);border-left:6px solid #00000012;border-radius:2px 7px 7px 2px;box-shadow:6px 7px 0 #ffffff80,10px 12px 13px #00000022;transform:rotate(-5deg);padding:12px;display:flex;flex-direction:column;justify-content:space-between;text-align:center}
.cover-object:before{content:'';position:absolute;inset:7px;border:1px solid currentColor;opacity:.23}
.cover-kicker{font-size:7px;letter-spacing:1.3px;text-transform:uppercase}.cover-title{font:15px/1.15 Georgia,serif;overflow-wrap:anywhere;max-height:86px;overflow:hidden}.cover-author{font-size:7px;letter-spacing:.6px;overflow:hidden;white-space:nowrap}
.book-content{padding:19px 18px}.book-genre{font-size:10px;color:var(--accent);text-transform:uppercase;letter-spacing:.7px;font-weight:700}
.book h3{font:normal 21px/1.25 Georgia,serif!important;margin:10px 0 5px;min-height:52px}.book-author{color:var(--muted);font-size:11px;min-height:17px}
.book-synopsis{margin:14px 0;font-size:11px;color:var(--muted);line-height:1.75;min-height:60px;display:-webkit-box;-webkit-line-clamp:3;-webkit-box-orient:vertical;overflow:hidden}
.book-tags{background:var(--surface-2);padding:5px 8px;border-radius:7px;font-size:10px;color:var(--accent);display:inline-block}
.book-foot{border-top:1px solid var(--line);margin-top:13px;padding-top:10px;font-size:10px;color:var(--muted)}
.metricline{color:var(--muted);font-size:12px;padding:10px 0 18px}
.footer{font-size:11px;color:var(--muted);padding:25px 0;border-top:1px solid var(--line);margin-top:35px}
@media(max-width:730px){.hero-art{height:205px}.hero-main h1{font-size:37px!important}[data-testid="stMainBlockContainer"]{padding-left:1rem;padding-right:1rem}.hero-book{transform:scale(.85)}}

/* Tema verde-escuro inspirado na prévia original */
html,body,.stApp,[data-testid="stAppViewContainer"]{background:#17231d!important;color:#eef2e9!important}
[data-testid="stHeader"],[data-testid="stToolbar"]{background:#17231d!important}
[data-testid="stSidebar"],[data-testid="stSidebarContent"]{background:#1d2b23!important}
.stApp p,.stApp label,.stApp span,.stApp div{color:inherit}
[data-testid="stWidgetLabel"] p,[data-testid="stMarkdownContainer"] p{color:#dce6dc}
.stTextArea textarea,.stTextInput input,[data-baseweb="select"]>div,
[data-baseweb="input"] input,[data-baseweb="textarea"] textarea{
background:#1e2c24!important;color:#eef2e9!important;border-color:#405447!important
}
.stTextArea textarea::placeholder,.stTextInput input::placeholder{color:#a5b5aa!important}
[data-baseweb="popover"]>div,[role="listbox"],[data-baseweb="menu"]{background:#27382e!important;color:#eef2e9!important}
[data-baseweb="select"] svg{fill:#bad3ad!important}
.stButton button[kind="secondary"],.stDownloadButton button{
background:#26382d!important;color:#eef2e9!important;border-color:#405447!important
}
.stButton button[kind="primary"],div[data-testid="stFormSubmitButton"] button{
background:#bad3ad!important;color:#17231d!important;border-color:#bad3ad!important
}
.stButton button[kind="primary"]:hover,div[data-testid="stFormSubmitButton"] button:hover{
background:#d0e2c8!important;color:#17231d!important
}
[data-testid="stTabs"] [role="tablist"]{border-bottom:1px solid #405447}
[data-testid="stTabs"] button[aria-selected="true"]{color:#bad3ad!important;border-bottom-color:#bad3ad!important}
[data-testid="stExpander"]{background:#223129;border:1px solid #405447;border-radius:12px}
[data-testid="stAlert"]{background:#283d30}
.book{box-shadow:0 8px 28px #0000001a}
.hero-orbit:after{border-color:#536a59}
.hero-caption{color:#eef2e9}

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
        if limit_pages and (pages is None or pages > limit_pages):
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
        if emotion != "Qualquer emoção" and any(e.strip().casefold() in tags_for(b) for e in emotion.split(",")):
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
    title_raw = text(book.get("Título do Livro")) or "Livro sem título"
    author_raw = text(book.get("Autor")) or "Autor não informado"
    title = html.escape(title_raw)
    author = html.escape(author_raw)
    genre = html.escape(text(book.get("Gênero")) or "Literatura")
    desc = html.escape(text(book.get("Sinopse")) or "Sem sinopse disponível.")
    if len(desc) > 290:
        desc = desc[:287].rsplit(" ", 1)[0] + "…"
    emotions = [text(book.get(k)) for k in ("Emoção I", "Emoção II") if text(book.get(k))]
    tags = " · ".join(html.escape(t) for t in emotions) or "Leitura especial"
    palette = [("#e8efdf", "#d6d8b6", "#405440"), ("#efebf6", "#ded4e9", "#554a66"), ("#f7e9e5", "#ebd4c6", "#655044"), ("#f5efd8", "#e9daa9", "#5c5336"), ("#e0eded", "#c6ddd8", "#395e57")]
    ix = int(hashlib.sha256(title_raw.encode()).hexdigest(), 16) % len(palette)
    bg, cover, ink = palette[ix]
    cover_title = title if len(title) < 77 else title[:74].rsplit(" ", 1)[0] + "…"
    st.markdown(f'''<div class="book"><div class="cover-area" style="--coverbg:{bg};--cover:{cover};--coverink:{ink}"><div class="cover-object"><span class="cover-kicker">EMOTION BOOK</span><span class="cover-title">{cover_title}</span><span class="cover-author">{author[:24]}</span></div></div><div class="book-content"><div class="book-genre">{genre}</div><h3>{title}</h3><div class="book-author">{author}</div><p class="book-synopsis">{desc}</p><span class="book-tags">✧ {tags}</span><div class="book-foot">✦ Afinidade semântica personalizada · {score:.2f}</div></div></div>''', unsafe_allow_html=True)
    col_a, col_b = st.columns(2, gap="small")
    with col_a:
        st.link_button("Ver livro ↗", safe_url(book), use_container_width=True)
    with col_b:
        ident = f"{title_raw}|{author_raw}"
        if st.button("♥ Salvo" if ident in st.session_state.saved else "♡ Salvar", key=f"save-{key}", use_container_width=True):
            if ident in st.session_state.saved:
                st.session_state.saved.remove(ident)
            else:
                st.session_state.saved.add(ident)
            st.rerun()


if "saved" not in st.session_state:
    st.session_state.saved = set()
if "results" not in st.session_state:
    st.session_state.results = []
if "history" not in st.session_state:
    st.session_state.history = []

st.markdown('''<div class="brand"><div class="brand-icon">▤</div><div><div class="brand-name">emotional book<span style="color:#396550">.</span></div><div class="brand-sub">HISTÓRIAS & SENTIMENTOS</div></div></div>''', unsafe_allow_html=True)
hero_left, hero_right = st.columns([1.15, 0.85], gap="medium", vertical_alignment="center")
with hero_left:
    st.markdown('''<div class="hero-main"><div class="eyebrow">● UMA BOA HISTÓRIA ENCONTRA VOCÊ</div><h1>Um livro para cada<br>versão de <em>você.</em></h1><p>Nem todo dia pede a mesma história.<br>Encontre uma leitura que converse com o seu momento.</p><p style="font-size:11px;margin-top:17px">♧ Um pouco de escuta. Um mundo de possibilidades.</p></div>''', unsafe_allow_html=True)
with hero_right:
    st.markdown('''<div class="hero-art"><div class="hero-orbit"></div><div class="hero-book one">Novos<br>caminhos</div><div class="hero-book two">Pequenos<br>mundos</div><div class="hero-book three">Tempo<br>para<br>sentir</div><div class="hero-caption">♡ No seu ritmo. Do seu jeito.</div></div>''', unsafe_allow_html=True)

try:
    books = load_catalog()
except Exception as exc:
    st.error(f"Não foi possível carregar o catálogo do MongoDB: {type(exc).__name__}.")
    st.info("Confira o Secret MONGO_URI e os nomes MONGO_DB/MONGO_COL.")
    st.stop()
if not books:
    st.warning("A coleção não contém documentos.")
    st.stop()

all_genres = sorted({text(b.get("Gênero")) for b in books if text(b.get("Gênero"))})
all_emotions = sorted({text(b.get(k)) for b in books for k in ("Emoção I", "Emoção II", "Sentimento I", "Sentimento II") if text(b.get(k))})
st.markdown(f'<div class="metricline">◦ {len(books)} livros na biblioteca &nbsp;&nbsp; ◦ {len(all_genres)} gêneros &nbsp;&nbsp; ◦ {len(all_emotions)} emoções e sentimentos</div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### ⚙️ Preferências do aplicativo")
    mode = st.radio("Modo de recomendação", ["Semântica (IA)", "Econômico (TF-IDF)"], help="Semântica: ONNX multilíngue e cache; econômico: contingência de baixo consumo.")
    st.caption("O modelo semântico é carregado somente após buscar uma recomendação.")
    if st.button("↻ Atualizar biblioteca", use_container_width=True):
        load_catalog.clear()
        st.session_state.results = []
        st.rerun()
    st.caption("Conexão segura ao MongoDB Atlas usando Secrets. A estante fica apenas nesta sessão.")

left, right = st.columns([2.05, 1], gap="medium", vertical_alignment="top")
with left:
    st.markdown('<div class="step">SEU PRÓXIMO CAPÍTULO</div><div class="panel-head">Como está seu dia?</div><div class="panel-caption">Escolha um sentimento, escreva sobre seu momento ou faça os dois.</div>', unsafe_allow_html=True)
    moods = [("☀ Alegre", "feliz"), ("☾ Reflexivo", "reflexivo"), ("〰 Ansioso", "ansioso"), ("☁ Triste", "triste"), ("✧ Inspirado", "inspirado"), ("♡ Romântico", "romântico"), ("◷ Nostálgico", "nostálgico"), ("◇ Entediado", "entediado")]
    if "moods" not in st.session_state:
        st.session_state.moods = set()
    st.markdown('**Hoje eu me sinto…** &nbsp; <small>Você pode escolher mais de um.</small>', unsafe_allow_html=True)
    for offset in (0, 4):
        mood_cols = st.columns(4, gap="small")
        for col, (label, value) in zip(mood_cols, moods[offset:offset+4]):
            with col:
                selected = value in st.session_state.moods
                if st.button(("✓ " if selected else "") + label, key="mood_" + value, use_container_width=True, type="primary" if selected else "secondary"):
                    if selected:
                        st.session_state.moods.remove(value)
                    else:
                        st.session_state.moods.add(value)
                    st.rerun()
    with st.form("discovery", clear_on_submit=False):
        emotion = ", ".join(sorted(st.session_state.moods)) if st.session_state.moods else "Qualquer emoção"
        query = st.text_area("Conte um pouco mais (opcional)", placeholder="Por exemplo: preciso de uma história acolhedora, sobre recomeços, amizade e esperança…", height=117, max_chars=600)
        with st.expander("☷ Personalize sua descoberta", expanded=False):
            genre = st.selectbox("Gênero literário", ["Todos"] + all_genres)
            a, b = st.columns(2)
            with a:
                number = st.select_slider("Número de livros", options=[2, 4, 6, 8, 10, 12], value=6)
                max_pages = st.slider("Máximo de páginas (0 = sem filtro)", 0, 1000, 0, 50)
            with b:
                diversity = st.select_slider("Variedade", options=["Fiel", "Equilibrada", "Surpreendente"], value="Equilibrada")
                min_rating = st.slider("Avaliação mínima (0 = sem filtro)", 0, 50, 0, 5)
        search = st.form_submit_button("✦ Encontrar minha leitura", use_container_width=True, type="primary")
        st.caption("Você não precisa encontrar as palavras certas. Deixe que as histórias ajudem.")
with right:
    st.markdown('''<div class="side-panel"><div style="background:var(--lilac);border-radius:10px;display:inline-block;padding:9px;font-size:19px">✧</div><h3>Me surpreenda</h3><p>Talvez sua próxima história favorita seja uma que você ainda não imaginou encontrar.</p></div>''', unsafe_allow_html=True)
    surprise = st.button("↝ Uma leitura surpresa", use_container_width=True)
    st.markdown(f'''<div class="shelf-note"><div>♡ SUA ESTANTE</div><div class="shelf-number">{len(st.session_state.saved):02d}</div><p style="font-size:12px;color:var(--muted)">livros guardados para outro momento</p></div>''', unsafe_allow_html=True)
    st.caption("✧ Pequenos encontros com grandes histórias.")

if search or surprise:
    try:
        if surprise:
            query, emotion, genre = "uma descoberta literária surpreendente", "Qualquer emoção", "Todos"
            number, max_pages, min_rating, diversity = 6, 0, 0, "Surpreendente"
        variety = {"Fiel":0.0,"Equilibrada":0.22,"Surpreendente":0.6}[diversity]
        with st.spinner("Encontrando histórias que combinam com você…"):
            st.session_state.results, st.session_state.last_mode = pick(books, query, emotion, genre, max_pages, min_rating, number, variety, mode)
        st.session_state.history.insert(0, f"{emotion} · {query[:90] or 'Sem descrição'}")
        st.session_state.history = st.session_state.history[:12]
    except Exception as exc:
        st.session_state.results = []
        st.error(f"Não foi possível gerar as recomendações: {type(exc).__name__}.")
        st.info("Se o modelo exceder a memória ou falhar no download, use o modo Econômico (TF-IDF) na barra lateral.")

st.divider()
tab_results, tab_shelf, tab_history = st.tabs(["✦ Descobrir", f"♡ Minha estante ({len(st.session_state.saved)})", "◷ Histórico"])
with tab_results:
    st.markdown('<div class="step">HISTÓRIAS PARA O SEU MOMENTO</div><div class="panel-head">Seu próximo encontro</div><div class="panel-caption">Sugestões de acordo com seu sentimento e com a afinidade de cada livro.</div>', unsafe_allow_html=True)
    if st.session_state.results:
        st.caption(f"Modo: {st.session_state.get('last_mode','IA')}. A pontuação de afinidade não representa avaliação pública.")
        items = st.session_state.results
        for start in range(0, len(items), 3):
            cols = st.columns(3, gap="medium")
            for offset, (book, score) in enumerate(items[start:start + 3]):
                with cols[offset]:
                    render_book(book, score, f"{start+offset}-{text(book.get('id'))}")
    else:
        st.info("Escolha uma emoção e toque em 'Encontrar minhas leituras' para descobrir histórias selecionadas para você.")
with tab_shelf:
    if st.session_state.saved:
        for item in sorted(st.session_state.saved):
            st.write("♡ " + item.replace("|", " — "))
        st.download_button("↓ Exportar minha estante", "\n".join(sorted(st.session_state.saved)).encode("utf-8"), file_name="estante_emotional_book.txt", mime="text/plain")
    else:
        st.info("Sua estante ainda está vazia. Salve livros nos cartões para encontrá-los aqui.")
with tab_history:
    if st.session_state.history:
        for entry in st.session_state.history:
            st.write("✧ " + entry)
    else:
        st.caption("Seu histórico de buscas desta sessão aparecerá aqui.")
st.markdown('<div class="footer"><b>emotional book.</b> &nbsp; · &nbsp; Incentivando a leitura através de encontros e descobertas. &nbsp; · &nbsp; Projeto acadêmico · FATEC Cotia</div>', unsafe_allow_html=True)
