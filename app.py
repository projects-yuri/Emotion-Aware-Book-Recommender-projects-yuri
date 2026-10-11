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


LANGUAGES = {"Português": "pt", "English": "en"}

I18N = {
    "pt": {
        "brand_sub": "HISTÓRIAS & SENTIMENTOS",
        "hero_eyebrow": "● UMA BOA HISTÓRIA ENCONTRA VOCÊ",
        "hero_title_1": "Um livro para cada",
        "hero_title_2": "versão de",
        "hero_title_em": "você.",
        "hero_p": "Nem todo dia pede a mesma história.<br>Encontre uma leitura que converse com o seu momento.",
        "hero_small": "♧ Um pouco de escuta. Um mundo de possibilidades.",
        "hero_book_1": "Novos<br>caminhos",
        "hero_book_2": "Pequenos<br>mundos",
        "hero_book_3": "Tempo<br>para<br>sentir",
        "hero_caption": "♡ No seu ritmo. Do seu jeito.",
        "library_metric": "◦ {books} livros na biblioteca &nbsp;&nbsp; ◦ {genres} gêneros &nbsp;&nbsp; ◦ {emotions} emoções e sentimentos",
        "sidebar_title": "⚙️ Preferências do aplicativo",
        "recommend_mode": "Modo de recomendação",
        "semantic_mode": "Semântica (IA)",
        "economic_mode": "Econômico (TF-IDF)",
        "mode_help": "Semântica: ONNX multilíngue e cache; econômico: contingência de baixo consumo.",
        "semantic_caption": "O modelo semântico é carregado somente após buscar uma recomendação.",
        "refresh": "↻ Atualizar biblioteca",
        "secure_caption": "Conexão segura ao MongoDB Atlas usando Secrets. A estante fica apenas nesta sessão.",
        "step_next": "SEU PRÓXIMO CAPÍTULO",
        "how_day": "Como está seu dia?",
        "how_day_caption": "Escolha um sentimento, escreva sobre seu momento ou faça os dois.",
        "today_feel": "Hoje eu me sinto…",
        "multi_feel": "Você pode escolher mais de um.",
        "more_label": "Conte um pouco mais (opcional)",
        "more_placeholder": "Por exemplo: preciso de uma história acolhedora, sobre recomeços, amizade e esperança…",
        "customize": "☷ Personalize sua descoberta",
        "genre": "Gênero literário",
        "all": "Todos",
        "num_books": "Número de livros",
        "max_pages": "Máximo de páginas (0 = sem filtro)",
        "variety": "Variedade",
        "faithful": "Fiel",
        "balanced": "Equilibrada",
        "surprising": "Surpreendente",
        "min_rating": "Avaliação mínima (0 = sem filtro)",
        "search": "✦ Encontrar minha leitura",
        "search_caption": "Você não precisa encontrar as palavras certas. Deixe que as histórias ajudem.",
        "surprise_title": "Me surpreenda",
        "surprise_text": "Talvez sua próxima história favorita seja uma que você ainda não imaginou encontrar.",
        "surprise_button": "↝ Uma leitura surpresa",
        "shelf_title": "♡ SUA ESTANTE",
        "saved_books": "livros guardados para outro momento",
        "small_encounters": "✧ Pequenos encontros com grandes histórias.",
        "spinner": "Encontrando histórias que combinam com você…",
        "recommend_error": "Não foi possível gerar as recomendações: {error}.",
        "recommend_hint": "Se o modelo exceder a memória ou falhar no download, use o modo Econômico (TF-IDF) na barra lateral.",
        "discover_tab": "✦ Descobrir",
        "shelf_tab": "♡ Minha estante ({count})",
        "history_tab": "◷ Histórico",
        "results_step": "HISTÓRIAS PARA O SEU MOMENTO",
        "results_title": "Seu próximo encontro",
        "results_caption": "Sugestões de acordo com seu sentimento e com a afinidade de cada livro.",
        "mode_caption": "Modo: {mode}. A pontuação de afinidade não representa avaliação pública.",
        "empty_results": "Escolha uma emoção e toque em 'Encontrar minha leitura' para descobrir histórias selecionadas para você.",
        "export_shelf": "↓ Exportar minha estante",
        "empty_shelf": "Sua estante ainda está vazia. Salve livros nos cartões para encontrá-los aqui.",
        "empty_history": "Seu histórico de buscas desta sessão aparecerá aqui.",
        "footer": "Incentivando a leitura através de encontros e descobertas.",
        "project": "Projeto acadêmico",
        "book_no_title": "Livro sem título",
        "book_no_author": "Autor não informado",
        "book_default_genre": "Literatura",
        "book_no_synopsis": "Sem sinopse disponível.",
        "book_special": "Leitura especial",
        "affinity": "✦ Afinidade semântica personalizada · {score:.2f}",
        "view_book": "Ver livro ↗",
        "saved": "♥ Salvo",
        "save": "♡ Salvar",
        "mongo_error": "Não foi possível carregar o catálogo do MongoDB: {error}.",
        "mongo_hint": "Confira o Secret MONGO_URI e os nomes MONGO_DB/MONGO_COL.",
        "empty_collection": "A coleção não contém documentos.",
        "no_description": "Sem descrição",
        "surprise_query": "uma descoberta literária surpreendente",
        "default_query": "livro envolvente emocionante interessante",
        "language": "Idioma / Language",
    },
    "en": {
        "brand_sub": "STORIES & FEELINGS",
        "hero_eyebrow": "● A GOOD STORY FINDS YOU",
        "hero_title_1": "A book for every",
        "hero_title_2": "version of",
        "hero_title_em": "you.",
        "hero_p": "Not every day calls for the same story.<br>Find a book that speaks to the moment you are living.",
        "hero_small": "♧ A little listening. A world of possibilities.",
        "hero_book_1": "New<br>paths",
        "hero_book_2": "Small<br>worlds",
        "hero_book_3": "Time<br>to<br>feel",
        "hero_caption": "♡ At your pace. Your way.",
        "library_metric": "◦ {books} books in the library &nbsp;&nbsp; ◦ {genres} genres &nbsp;&nbsp; ◦ {emotions} emotions and feelings",
        "sidebar_title": "⚙️ App preferences",
        "recommend_mode": "Recommendation mode",
        "semantic_mode": "Semantic (AI)",
        "economic_mode": "Economy (TF-IDF)",
        "mode_help": "Semantic: multilingual ONNX model with caching; economy: low-memory fallback.",
        "semantic_caption": "The semantic model is loaded only after you request recommendations.",
        "refresh": "↻ Refresh library",
        "secure_caption": "Secure MongoDB Atlas connection through Secrets. Your shelf is kept only for this session.",
        "step_next": "YOUR NEXT CHAPTER",
        "how_day": "How is your day going?",
        "how_day_caption": "Choose a feeling, write about your moment, or do both.",
        "today_feel": "Today I feel…",
        "multi_feel": "You can choose more than one.",
        "more_label": "Tell me a little more (optional)",
        "more_placeholder": "For example: I need a comforting story about fresh starts, friendship, and hope…",
        "customize": "☷ Personalize your discovery",
        "genre": "Literary genre",
        "all": "All",
        "num_books": "Number of books",
        "max_pages": "Maximum pages (0 = no filter)",
        "variety": "Variety",
        "faithful": "Faithful",
        "balanced": "Balanced",
        "surprising": "Surprising",
        "min_rating": "Minimum rating (0 = no filter)",
        "search": "✦ Find my next read",
        "search_caption": "You do not need to find the perfect words. Let the stories help.",
        "surprise_title": "Surprise me",
        "surprise_text": "Your next favorite story may be one you never expected to find.",
        "surprise_button": "↝ Surprise me with a book",
        "shelf_title": "♡ YOUR SHELF",
        "saved_books": "books saved for another moment",
        "small_encounters": "✧ Small encounters with great stories.",
        "spinner": "Finding stories that fit your moment…",
        "recommend_error": "Recommendations could not be generated: {error}.",
        "recommend_hint": "If the model runs out of memory or fails to download, use Economy (TF-IDF) in the sidebar.",
        "discover_tab": "✦ Discover",
        "shelf_tab": "♡ My shelf ({count})",
        "history_tab": "◷ History",
        "results_step": "STORIES FOR YOUR MOMENT",
        "results_title": "Your next encounter",
        "results_caption": "Suggestions based on your feelings and each book's semantic affinity.",
        "mode_caption": "Mode: {mode}. The affinity score is not a public review score.",
        "empty_results": "Choose an emotion and select 'Find my next read' to discover stories selected for you.",
        "export_shelf": "↓ Export my shelf",
        "empty_shelf": "Your shelf is still empty. Save books from the cards to find them here.",
        "empty_history": "Your searches from this session will appear here.",
        "footer": "Encouraging reading through encounters and discoveries.",
        "project": "Academic project",
        "book_no_title": "Untitled book",
        "book_no_author": "Author not provided",
        "book_default_genre": "Literature",
        "book_no_synopsis": "No synopsis available.",
        "book_special": "Special read",
        "affinity": "✦ Personalized semantic affinity · {score:.2f}",
        "view_book": "View book ↗",
        "saved": "♥ Saved",
        "save": "♡ Save",
        "mongo_error": "The MongoDB catalog could not be loaded: {error}.",
        "mongo_hint": "Check the MONGO_URI Secret and the MONGO_DB/MONGO_COL names.",
        "empty_collection": "The collection contains no documents.",
        "no_description": "No description",
        "surprise_query": "a surprising literary discovery",
        "default_query": "engaging emotional interesting book",
        "language": "Idioma / Language",
    },
}

EMOTION_LABELS = {
    "feliz": {"pt": "☀ Alegre", "en": "☀ Happy"},
    "reflexivo": {"pt": "☾ Reflexivo", "en": "☾ Reflective"},
    "ansioso": {"pt": "〰 Ansioso", "en": "〰 Anxious"},
    "triste": {"pt": "☁ Triste", "en": "☁ Sad"},
    "inspirado": {"pt": "✧ Inspirado", "en": "✧ Inspired"},
    "romântico": {"pt": "♡ Romântico", "en": "♡ Romantic"},
    "nostálgico": {"pt": "◷ Nostálgico", "en": "◷ Nostalgic"},
    "entediado": {"pt": "◇ Entediado", "en": "◇ Bored"},
}

EMOTION_QUERY_EN = {
    "feliz": "happy",
    "reflexivo": "reflective",
    "ansioso": "anxious",
    "triste": "sad",
    "inspirado": "inspired",
    "romântico": "romantic",
    "nostálgico": "nostalgic",
    "entediado": "bored",
}

GENRE_LABELS_EN = {
    "Aventura": "Adventure",
    "Biografia": "Biography",
    "Clássico": "Classic",
    "Clássicos": "Classics",
    "Comédia": "Comedy",
    "Conto": "Short Story",
    "Drama": "Drama",
    "Fantasia": "Fantasy",
    "Ficção": "Fiction",
    "Ficção Científica": "Science Fiction",
    "Histórico": "Historical",
    "Horror": "Horror",
    "Infantojuvenil": "Young Adult",
    "Mistério": "Mystery",
    "Não Ficção": "Nonfiction",
    "Poesia": "Poetry",
    "Policial": "Crime",
    "Romance": "Romance",
    "Suspense": "Thriller",
    "Terror": "Horror",
}

FIELD_ALIASES_EN = {
    "Título do Livro": ("Título do Livro EN", "Book Title", "Title"),
    "Autor": ("Autor EN", "Author"),
    "Gênero": ("Gênero EN", "Genre"),
    "Sinopse": ("Sinopse EN", "Synopsis", "Description"),
    "Emoção I": ("Emoção I EN", "Emotion I"),
    "Emoção II": ("Emoção II EN", "Emotion II"),
    "Sentimento I": ("Sentimento I EN", "Feeling I", "Sentiment I"),
    "Sentimento II": ("Sentimento II EN", "Feeling II", "Sentiment II"),
}


def tr(key, **kwargs):
    lang = st.session_state.get("lang", "pt")
    value = I18N.get(lang, I18N["pt"]).get(key, I18N["pt"].get(key, key))
    return value.format(**kwargs) if kwargs else value

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
/* Não forçar herança global: capas pastéis usam texto escuro */
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


/* CONTRASTE: texto escuro apenas nas capas claras */
.stApp .hero-book,.stApp .hero-book.one{color:#283f37!important}
.stApp .hero-book.two{color:#403650!important}
.stApp .hero-book.three{color:#523d35!important}
.stApp .hero-book *,.stApp .cover-area *,.stApp .cover-object *{color:inherit!important}
.stApp .cover-area,.stApp .cover-object{color:var(--coverink)!important}
.stApp .hero-caption,.stApp .hero-caption *{color:#eef2e9!important}
/* Botões verdes selecionados: fundo claro, texto e ícones escuros */
.stApp [data-testid="stButton"] button[kind="primary"],
.stApp [data-testid="stFormSubmitButton"] button,
.stApp [data-testid="stButton"] button[data-testid="stBaseButton-primary"]{
 background:#bad3ad!important;color:#18271e!important;border-color:#bad3ad!important;
}
.stApp [data-testid="stButton"] button[kind="primary"] *,
.stApp [data-testid="stFormSubmitButton"] button *,
.stApp [data-testid="stButton"] button[data-testid="stBaseButton-primary"] *{color:#18271e!important;fill:#18271e!important}
.stApp [data-testid="stButton"] button[kind="secondary"] *,
.stApp [data-testid="stDownloadButton"] button *{color:#eef2e9!important}
/* Texto de descrição e labels sobre fundo escuro */
.stApp .panel-caption,.stApp .metricline,.stApp .book-synopsis,.stApp .book-author,.stApp .book-foot,.stApp .footer,
.stApp .hero-main p,.stApp .side-panel p{color:#cbd8cd!important}
.stApp .brand-name,.stApp .panel-head,.stApp .side-panel h3,.stApp .book h3{color:#eef2e9!important}
.stApp .book-genre,.stApp .eyebrow,.stApp .hero-main h1 em{color:#bad3ad!important}
.stApp .hero-main h1{color:#eef2e9!important}
.stApp [data-testid="stWidgetLabel"] p{color:#eef2e9!important}

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


def display_field(book, base_key):
    # If English fields exist in MongoDB, prefer them in English mode.
    if st.session_state.get("lang", "pt") == "en":
        for alias in FIELD_ALIASES_EN.get(base_key, ()):
            value = text(book.get(alias))
            if value:
                return value
    return text(book.get(base_key))


def summary(book):
    # Keep embeddings compact. Include optional English parallel fields if present.
    fields = [
        "Título do Livro", "Gênero", "Sentimento I", "Sentimento II",
        "Emoção I", "Emoção II", "Sinopse",
        "Título do Livro EN", "Book Title", "Title", "Gênero EN", "Genre",
        "Sentimento I EN", "Sentimento II EN", "Feeling I", "Feeling II",
        "Emotion I", "Emotion II", "Sinopse EN", "Synopsis", "Description",
    ]
    return " | ".join(text(book.get(f))[:580] for f in fields if text(book.get(f)))[:1150]


def cache_key(books):
    h = hashlib.sha256()
    for b in books:
        h.update(summary(b).encode("utf-8", errors="replace"))
        h.update(b"\0")
    return h.hexdigest()


@st.cache_resource(show_spinner=False)
def embedding_model():
    # Same multilingual embedding space for Portuguese and English.
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
    keys = (
        "Emoção I", "Emoção II", "Sentimento I", "Sentimento II",
        "Emotion I", "Emotion II", "Feeling I", "Feeling II",
        "Sentiment I", "Sentiment II",
    )
    return {text(b.get(k)).casefold() for k in keys if text(b.get(k))}


def genre_label(raw_genre):
    if st.session_state.get("lang", "pt") == "en":
        return GENRE_LABELS_EN.get(raw_genre, raw_genre)
    return raw_genre


def pick(books, q, emotion, genre, limit_pages, min_rating, n, variety, mode, lang):
    eligible = []
    for i, b in enumerate(books):
        if genre and text(b.get("Gênero")) != genre:
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

    emotion_terms = [e.strip() for e in emotion.split(",") if e.strip()]
    semantic_emotion = ", ".join(
        EMOTION_QUERY_EN.get(e, e) if lang == "en" else e
        for e in emotion_terms
    )

    vector_query = " ".join(t for t in [q, semantic_emotion] if t).strip()
    if not vector_query:
        vector_query = I18N[lang]["default_query"]

    if mode == "semantic":
        model = embedding_model()
        vectors = catalog_vectors(books, cache_key(books))
        query_emb = np.asarray(next(iter(model.embed([vector_query]))), dtype=np.float32)
        query_emb /= max(float(np.linalg.norm(query_emb)), 1e-8)
        sim = vectors @ query_emb
    else:
        # Lexical fallback: best when query and catalog use the same language.
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        docs = [summary(b) for b in books]
        features = TfidfVectorizer(
            strip_accents="unicode",
            ngram_range=(1, 2),
            max_features=12000,
        ).fit_transform(docs + [vector_query])
        sim = cosine_similarity(features[:-1], features[-1]).ravel().astype(np.float32)
        vectors = None

    scores = sim.copy()
    for i, b in enumerate(books):
        if emotion_terms and any(e.casefold() in tags_for(b) for e in emotion_terms):
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
            idx = max(
                order,
                key=lambda k: (1 - variety) * float(scores[k])
                - variety * max(float(vectors[k] @ vectors[c]) for c in chosen),
            )
        chosen.append(idx)
        order.remove(idx)

    return [(books[i], float(scores[i])) for i in chosen], mode


def render_book(book, score, key):
    title_raw = display_field(book, "Título do Livro") or tr("book_no_title")
    author_raw = display_field(book, "Autor") or tr("book_no_author")
    title = html.escape(title_raw)
    author = html.escape(author_raw)

    raw_genre = display_field(book, "Gênero")
    if st.session_state.get("lang", "pt") == "en" and raw_genre == text(book.get("Gênero")):
        raw_genre = GENRE_LABELS_EN.get(raw_genre, raw_genre)
    genre = html.escape(raw_genre or tr("book_default_genre"))

    desc = html.escape(display_field(book, "Sinopse") or tr("book_no_synopsis"))
    if len(desc) > 290:
        desc = desc[:287].rsplit(" ", 1)[0] + "…"

    emotions = [
        display_field(book, k)
        for k in ("Emoção I", "Emoção II")
        if display_field(book, k)
    ]
    tags = " · ".join(html.escape(t) for t in emotions) or tr("book_special")

    palette = [
        ("#e8efdf", "#d6d8b6", "#405440"),
        ("#efebf6", "#ded4e9", "#554a66"),
        ("#f7e9e5", "#ebd4c6", "#655044"),
        ("#f5efd8", "#e9daa9", "#5c5336"),
        ("#e0eded", "#c6ddd8", "#395e57"),
    ]
    ix = int(hashlib.sha256(title_raw.encode()).hexdigest(), 16) % len(palette)
    bg, cover, ink = palette[ix]
    cover_title = title if len(title) < 77 else title[:74].rsplit(" ", 1)[0] + "…"

    st.markdown(
        f'''<div class="book"><div class="cover-area" style="--coverbg:{bg};--cover:{cover};--coverink:{ink}"><div class="cover-object"><span class="cover-kicker">EMOTION BOOK</span><span class="cover-title">{cover_title}</span><span class="cover-author">{author[:24]}</span></div></div><div class="book-content"><div class="book-genre">{genre}</div><h3>{title}</h3><div class="book-author">{author}</div><p class="book-synopsis">{desc}</p><span class="book-tags">✧ {tags}</span><div class="book-foot">{tr("affinity", score=score)}</div></div></div>''',
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2, gap="small")
    with col_a:
        st.link_button(tr("view_book"), safe_url(book), use_container_width=True)
    with col_b:
        ident = f"{title_raw}|{author_raw}"
        if st.button(
            tr("saved") if ident in st.session_state.saved else tr("save"),
            key=f"save-{key}",
            use_container_width=True,
        ):
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
if "moods" not in st.session_state:
    st.session_state.moods = set()
if "lang" not in st.session_state:
    st.session_state.lang = "pt"

# Language selector stays on the home screen, not hidden in the sidebar.
lang_col, spacer_col = st.columns([0.34, 0.66], gap="small")
with lang_col:
    current_label = "Português" if st.session_state.lang == "pt" else "English"
    selected_language = st.radio(
        tr("language"),
        options=["Português", "English"],
        index=0 if current_label == "Português" else 1,
        horizontal=True,
        key="language_selector",
    )

selected_lang = LANGUAGES[selected_language]
if selected_lang != st.session_state.lang:
    st.session_state.lang = selected_lang
    # Results contain localized labels; clear them when switching language.
    st.session_state.results = []
    st.rerun()

lang = st.session_state.lang

st.markdown(
    f'''<div class="brand"><div class="brand-icon">▤</div><div><div class="brand-name">emotional book<span style="color:#396550">.</span></div><div class="brand-sub">{tr("brand_sub")}</div></div></div>''',
    unsafe_allow_html=True,
)

hero_left, hero_right = st.columns([1.15, 0.85], gap="medium", vertical_alignment="center")
with hero_left:
    st.markdown(
        f'''<div class="hero-main"><div class="eyebrow">{tr("hero_eyebrow")}</div><h1>{tr("hero_title_1")}<br>{tr("hero_title_2")} <em>{tr("hero_title_em")}</em></h1><p>{tr("hero_p")}</p><p style="font-size:11px;margin-top:17px">{tr("hero_small")}</p></div>''',
        unsafe_allow_html=True,
    )

with hero_right:
    st.markdown(
        f'''<div class="hero-art"><div class="hero-orbit"></div><div class="hero-book one">{tr("hero_book_1")}</div><div class="hero-book two">{tr("hero_book_2")}</div><div class="hero-book three">{tr("hero_book_3")}</div><div class="hero-caption">{tr("hero_caption")}</div></div>''',
        unsafe_allow_html=True,
    )

try:
    books = load_catalog()
except Exception as exc:
    st.error(tr("mongo_error", error=type(exc).__name__))
    st.info(tr("mongo_hint"))
    st.stop()

if not books:
    st.warning(tr("empty_collection"))
    st.stop()

all_genres = sorted({text(b.get("Gênero")) for b in books if text(b.get("Gênero"))})
all_emotions = sorted(
    {
        text(b.get(k))
        for b in books
        for k in ("Emoção I", "Emoção II", "Sentimento I", "Sentimento II")
        if text(b.get(k))
    }
)

st.markdown(
    f'<div class="metricline">{tr("library_metric", books=len(books), genres=len(all_genres), emotions=len(all_emotions))}</div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(f"### {tr('sidebar_title')}")
    mode = st.radio(
        tr("recommend_mode"),
        ["semantic", "tfidf"],
        format_func=lambda x: tr("semantic_mode") if x == "semantic" else tr("economic_mode"),
        help=tr("mode_help"),
    )
    st.caption(tr("semantic_caption"))

    if st.button(tr("refresh"), use_container_width=True):
        load_catalog.clear()
        catalog_vectors.clear()
        st.session_state.results = []
        st.rerun()

    st.caption(tr("secure_caption"))

left, right = st.columns([2.05, 1], gap="medium", vertical_alignment="top")

with left:
    st.markdown(
        f'<div class="step">{tr("step_next")}</div><div class="panel-head">{tr("how_day")}</div><div class="panel-caption">{tr("how_day_caption")}</div>',
        unsafe_allow_html=True,
    )

    mood_order = [
        "feliz",
        "reflexivo",
        "ansioso",
        "triste",
        "inspirado",
        "romântico",
        "nostálgico",
        "entediado",
    ]

    st.markdown(
        f'**{tr("today_feel")}** &nbsp; <small>{tr("multi_feel")}</small>',
        unsafe_allow_html=True,
    )

    for offset in (0, 4):
        mood_cols = st.columns(4, gap="small")
        for col, value in zip(mood_cols, mood_order[offset:offset + 4]):
            with col:
                selected = value in st.session_state.moods
                label = EMOTION_LABELS[value][lang]
                if st.button(
                    ("✓ " if selected else "") + label,
                    key="mood_" + value,
                    use_container_width=True,
                    type="primary" if selected else "secondary",
                ):
                    if selected:
                        st.session_state.moods.remove(value)
                    else:
                        st.session_state.moods.add(value)
                    st.rerun()

    with st.form("discovery", clear_on_submit=False):
        emotion = ", ".join(sorted(st.session_state.moods))

        query = st.text_area(
            tr("more_label"),
            placeholder=tr("more_placeholder"),
            height=117,
            max_chars=600,
        )

        with st.expander(tr("customize"), expanded=False):
            genre_options = [None] + all_genres
            genre = st.selectbox(
                tr("genre"),
                genre_options,
                format_func=lambda x: tr("all") if x is None else genre_label(x),
            )

            a, b = st.columns(2)
            with a:
                number = st.select_slider(
                    tr("num_books"),
                    options=[2, 4, 6, 8, 10, 12],
                    value=6,
                )
                max_pages = st.slider(
                    tr("max_pages"),
                    0,
                    1000,
                    0,
                    50,
                )

            with b:
                diversity = st.select_slider(
                    tr("variety"),
                    options=["faithful", "balanced", "surprising"],
                    value="balanced",
                    format_func=lambda x: {
                        "faithful": tr("faithful"),
                        "balanced": tr("balanced"),
                        "surprising": tr("surprising"),
                    }[x],
                )
                min_rating = st.slider(
                    tr("min_rating"),
                    0,
                    50,
                    0,
                    5,
                )

        search = st.form_submit_button(
            tr("search"),
            use_container_width=True,
            type="primary",
        )
        st.caption(tr("search_caption"))

with right:
    st.markdown(
        f'''<div class="side-panel"><div style="background:var(--lilac);border-radius:10px;display:inline-block;padding:9px;font-size:19px">✧</div><h3>{tr("surprise_title")}</h3><p>{tr("surprise_text")}</p></div>''',
        unsafe_allow_html=True,
    )

    surprise = st.button(
        tr("surprise_button"),
        use_container_width=True,
    )

    st.markdown(
        f'''<div class="shelf-note"><div>{tr("shelf_title")}</div><div class="shelf-number">{len(st.session_state.saved):02d}</div><p style="font-size:12px;color:var(--muted)">{tr("saved_books")}</p></div>''',
        unsafe_allow_html=True,
    )

    st.caption(tr("small_encounters"))

if search or surprise:
    try:
        if surprise:
            query, emotion, genre = tr("surprise_query"), "", None
            number, max_pages, min_rating, diversity = 6, 0, 0, "surprising"

        variety = {
            "faithful": 0.0,
            "balanced": 0.22,
            "surprising": 0.6,
        }[diversity]

        with st.spinner(tr("spinner")):
            st.session_state.results, st.session_state.last_mode = pick(
                books,
                query,
                emotion,
                genre,
                max_pages,
                min_rating,
                number,
                variety,
                mode,
                lang,
            )

        mood_history = ", ".join(
            EMOTION_LABELS[e][lang].split(" ", 1)[-1]
            for e in sorted(st.session_state.moods)
        ) or tr("all")

        st.session_state.history.insert(
            0,
            f"{mood_history} · {query[:90] or tr('no_description')}",
        )
        st.session_state.history = st.session_state.history[:12]

    except Exception as exc:
        st.session_state.results = []
        st.error(tr("recommend_error", error=type(exc).__name__))
        st.info(tr("recommend_hint"))

st.divider()

tab_results, tab_shelf, tab_history = st.tabs(
    [
        tr("discover_tab"),
        tr("shelf_tab", count=len(st.session_state.saved)),
        tr("history_tab"),
    ]
)

with tab_results:
    st.markdown(
        f'<div class="step">{tr("results_step")}</div><div class="panel-head">{tr("results_title")}</div><div class="panel-caption">{tr("results_caption")}</div>',
        unsafe_allow_html=True,
    )

    if st.session_state.results:
        last_mode = st.session_state.get("last_mode", "semantic")
        mode_label = tr("semantic_mode") if last_mode == "semantic" else tr("economic_mode")
        st.caption(tr("mode_caption", mode=mode_label))

        items = st.session_state.results
        for start in range(0, len(items), 3):
            cols = st.columns(3, gap="medium")
            for offset, (book, score) in enumerate(items[start:start + 3]):
                with cols[offset]:
                    render_book(
                        book,
                        score,
                        f"{start + offset}-{text(book.get('id'))}",
                    )
    else:
        st.info(tr("empty_results"))

with tab_shelf:
    if st.session_state.saved:
        for item in sorted(st.session_state.saved):
            st.write("♡ " + item.replace("|", " — "))

        st.download_button(
            tr("export_shelf"),
            "\n".join(sorted(st.session_state.saved)).encode("utf-8"),
            file_name="emotional_book_shelf.txt" if lang == "en" else "estante_emotional_book.txt",
            mime="text/plain",
        )
    else:
        st.info(tr("empty_shelf"))

with tab_history:
    if st.session_state.history:
        for entry in st.session_state.history:
            st.write("✧ " + entry)
    else:
        st.caption(tr("empty_history"))

st.markdown(
    f'<div class="footer"><b>emotional book.</b> &nbsp; · &nbsp; {tr("footer")} &nbsp; · &nbsp; {tr("project")} · FATEC Cotia</div>',
    unsafe_allow_html=True,
)
