"""
Emotional Book — interface moderna + Flask + ML + HTTPS + Google Gemini

pip install flask pymongo sentence-transformers scikit-learn
            numpy pandas google-generativeai cryptography

Rodar:
  Windows CMD:
    set GEMINI_API_KEY=AIzaSy...suachave
    python Book_code.py

Acesso: https://localhost:5443

Interface atualizada:
  Cores suaves, layout responsivo, modo claro/escuro e seleção de emoções.
  Filtros reais de gênero/páginas, 4/6/8/10 sugestões e controle de variedade.
  Estante pessoal, exportação TXT e histórico local de até 12 buscas.
  Favoritos e buscas são salvos apenas no navegador, sem alterar o MongoDB.
  Os cartões usam ilustrações próprias; não representam capas oficiais.

Mantém as configurações e dependências do projeto original.
Python 3.10+ (o projeto original usa anotações com |).
Para executar: substitua o Book_code.py na mesma pasta do seu projeto.
Prévia HTML separada: somente demonstração com livros fictícios.
"""

import os, re, json, pickle, time, hashlib, ssl, threading, datetime, ipaddress, math
import unicodedata
import logging
import numpy as np
import pandas as pd
from collections import defaultdict

from flask import Flask, request, jsonify, redirect
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ============================================================
# CONFIGURAÇÕES GERAIS — classe centralizada
# ============================================================
class Config:
    CERT_FILE     = "cert.pem"
    KEY_FILE      = "key.pem"
    HTTPS_PORT    = 5443
    HTTP_PORT     = 5000
    CACHE_PATH    = "embeddings_cache.pkl"
    FEEDBACK_PATH = "feedback_usuario.json"
    FEEDBACK_MIN_SCORE = -5          # limite antes de excluir livro do pool
    MONGO_URI     = "mongodb://localhost:27017"
    MONGO_TIMEOUT = 5000             # ms
    MONGO_DB      = "admin"
    MONGO_COL     = "Book_Dataset_V3"

# Atalhos de compatibilidade (mantêm o código restante funcionando)
CERT_FILE     = Config.CERT_FILE
KEY_FILE      = Config.KEY_FILE
HTTPS_PORT    = Config.HTTPS_PORT
HTTP_PORT     = Config.HTTP_PORT
CACHE_PATH    = Config.CACHE_PATH
FEEDBACK_PATH = Config.FEEDBACK_PATH

# ============================================================
# HTTPS — certificado autoassinado via 'cryptography' (sem pyopenssl)
# ============================================================
def gerar_cert_autoassinado():
    if os.path.exists(CERT_FILE) and os.path.exists(KEY_FILE):
        logger.info(f"Certificado SSL existente encontrado ({CERT_FILE}).")
        return
    try:
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import rsa

        chave = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        nome  = x509.Name([
            x509.NameAttribute(NameOID.COUNTRY_NAME,           "BR"),
            x509.NameAttribute(NameOID.STATE_OR_PROVINCE_NAME, "SP"),
            x509.NameAttribute(NameOID.LOCALITY_NAME,          "Cotia"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME,      "FATEC"),
            x509.NameAttribute(NameOID.COMMON_NAME,            "localhost"),
        ])
        agora = datetime.datetime.now(datetime.timezone.utc)
        cert  = (
            x509.CertificateBuilder()
            .subject_name(nome).issuer_name(nome)
            .public_key(chave.public_key())
            .serial_number(x509.random_serial_number())
            .not_valid_before(agora)
            .not_valid_after(agora + datetime.timedelta(days=365))
            .add_extension(
                x509.SubjectAlternativeName([
                    x509.DNSName("localhost"),
                    x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
                ]), critical=False,
            )
            .sign(chave, hashes.SHA256())
        )
        with open(KEY_FILE, "wb") as f:
            f.write(chave.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            ))
        with open(CERT_FILE, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
        logger.info("Certificado SSL autoassinado gerado com sucesso.")
    except ImportError:
        logger.error("Instale o pacote cryptography: pip install cryptography")
        raise

# ============================================================
# APPS FLASK
# ============================================================
app = Flask(__name__)

redirect_app = Flask("redirect_app")

@redirect_app.route("/", defaults={"path": ""})
@redirect_app.route("/<path:path>")
def http_redirect(path):
    host = request.host.split(":")[0]
    url  = f"https://{host}:{HTTPS_PORT}/{path}"
    if request.query_string:
        url += "?" + request.query_string.decode()
    return redirect(url, code=301)

# ============================================================
# RATE LIMITING
# ============================================================
_rate_recomendar: dict = defaultdict(list)
_rate_feedback:   dict = defaultdict(list)

def _check_rate(cache, ip, limite, janela_s=60):
    agora = time.time()
    cache[ip] = [t for t in cache[ip] if agora - t < janela_s]
    if len(cache[ip]) >= limite:
        return False
    cache[ip].append(agora)
    return True

def rate_recomendar(ip): return _check_rate(_rate_recomendar, ip, 20)
def rate_feedback(ip):   return _check_rate(_rate_feedback,   ip, 60)

# ============================================================
# 1. MONGODB — conexão única (singleton)
# ============================================================
_mongo_client: MongoClient | None = None

def _get_mongo_client() -> MongoClient:
    """Retorna cliente MongoDB reutilizável (uma única conexão durante toda a execução)."""
    global _mongo_client
    if _mongo_client is None:
        _mongo_client = MongoClient(Config.MONGO_URI, serverSelectionTimeoutMS=Config.MONGO_TIMEOUT)
    return _mongo_client

def carregar_dados():
    client = _get_mongo_client()
    try:
        client.admin.command("ping")
    except ConnectionFailure as e:
        raise RuntimeError(f"MongoDB não está rodando: {e}")
    collection = client[Config.MONGO_DB][Config.MONGO_COL]
    docs = list(collection.find())
    df = pd.DataFrame(docs).drop(columns=["_id"], errors="ignore").reset_index(drop=True)
    scaler = MinMaxScaler()
    if "rating" in df.columns:
        df["rating_norm"] = scaler.fit_transform(
            pd.to_numeric(df["rating"], errors="coerce").fillna(0).values.reshape(-1, 1)
        ).flatten()
    else:
        df["rating_norm"] = 0.5
    return df

# ============================================================
# 2. PRÉ-PROCESSAMENTO
# ============================================================
SINONIMOS = {
    "feliz":      "alegre otimista esperança eufórico celebração conquista",
    "inspirado":  "superação conquista determinação coragem realizacao vitoria",
    "nostálgico": "nostalgia memórias passado saudade afeto familiaridade",
    "romântico":  "amor romance paixão afeto conexão ternura",
    "reflexivo":  "introspecção autoconhecimento propósito significado crescimento",
    "triste":     "reconfortante esperança superação acolhimento cura renascimento leveza alegria",
    "ansioso":    "calma serenidade equilíbrio paz mindfulness leveza clareza acolhimento",
    "entediado":  "aventura descoberta fascinante emocionante surpreendente viagem imaginação",
    "épico":      "grandioso heróico batalha guerra coragem determinação",
    "misterioso": "suspense thriller enigma crime investigação revelação",
    "intenso":    "emocionante dramático transformador marcante",
}

MAPA_TERAPEUTICO = {
    "triste":     "reconfortante esperança superação acolhimento cura leveza alegria renascimento "
                  "autoajuda bem-estar saúde mental resiliência recomeço força interior "
                  "motivação consolo empatia compaixão",
    "ansioso":    "calma serenidade equilíbrio paz clareza acolhimento leveza tranquilidade",
    "entediado":  "aventura descoberta fascinante emocionante surpreendente imaginação viagem",
    "feliz":      "alegre otimista esperança celebração conquista eufórico",
    "inspirado":  "superação conquista determinação coragem realizacao vitoria",
    "romântico":  "amor romance paixão afeto conexão ternura",
    "reflexivo":  "introspecção autoconhecimento propósito significado crescimento filosofia",
    "nostálgico": "nostalgia memórias passado saudade afeto familiaridade ternura",
    "neutro":     "",
}

CONTRASTES = ["mas", "porém", "contudo", "entretanto", "embora", "apesar", "todavia"]

import unicodedata  # adicione no topo do arquivo junto dos outros imports

def limpar_texto(texto):
    texto = str(texto).lower().strip()
    # Normaliza para NFC: garante que ã, ç, é etc. sejam sempre a mesma forma
    texto = unicodedata.normalize("NFC", texto)
    texto = re.sub(r'[^\w\s]', ' ', texto, flags=re.UNICODE)
    texto = re.sub(r'\b\d+\b', ' ', texto)
    return re.sub(r'\s+', ' ', texto).strip()

def construir_perfil(row):
    campos = {"Sentimento I": 3, "Emoção I": 3, "Sentimento II": 2, "Emoção II": 2, "Gênero": 1}
    partes = []
    for campo, peso in campos.items():
        val = limpar_texto(str(row.get(campo, "")))
        if val:
            partes.extend([val] * peso)
    sinopse = str(row.get("Sinopse", "")).strip()
    if sinopse and sinopse not in ("nan", "N/A", ""):
        partes.extend([limpar_texto(sinopse[:300])] * 2)
    return ' '.join(partes)

# ============================================================
# 3. DETECÇÃO DE EMOÇÕES
# ============================================================
EMOCOES_KEYWORDS = {
    "feliz":      ["feliz","felicidade","alegre","alegria","animado","contente","bem","ótimo",
                   "eufórico","empolgado","esperançoso","positivo","satisfeito","entusiasmado",
                   "realizado","celebrar","comemorando","promoção","conquista","felicíssimo"],
    "triste":     ["triste","tristeza","tristo","mal","deprimido","deprimida","depressão",
                   "depressivo","depressiva","em depressão","com depressão","tenho depressão",
                   "melancólico","melancólica","melancolia","chateado","chateada","abatido",
                   "abatida","desmotivado","desmotivada","desolado","desolada","angustiado",
                   "angustiada","angústia","sofrendo","sofrimento","luto","saudade","vazio",
                   "vazia","perdido","perdida","sozinho","sozinha","solidão","isolado","isolada",
                   "abandonado","abandonada","magoado","magoada","mágoa","decepcionado",
                   "decepcionada","decepção","frustrado","frustrada","frustração","chorando",
                   "choro","lágrimas","dor","péssimo","péssima","horrível","terrível",
                   "arrasado","arrasada","destruído","destruída","sem esperança","sem saída",
                   "esgotado","esgotada","exausto","exausta","pesado","pesada","sufocado",
                   "sufocada","não consigo","nao consigo","não aguento","nao aguento",
                   "quebrado","quebrada","devastado","devastada","sem energia","sem ânimo",
                   "sem animo","triste demais","muito triste","tão triste"],
    "ansioso":    ["ansioso","ansiedade","nervoso","nervosismo","preocupado","preocupação",
                   "tenso","tensão","inquieto","estressado","estresse","agitado","apreensivo",
                   "inseguro","insegurança","assustado","medo","raiva","irritado","irritação",
                   "revoltado","frustração","pressão","sobrecarregado","esgotado","culpa",
                   "culpado","vergonha","envergonhado","constrangido"],
    "inspirado":  ["inspirado","inspiração","motivado","motivação","superação","conquista",
                   "determinado","determinação","força","coragem","quero melhorar",
                   "quero crescer","desenvolvimento","produtividade","objetivo","meta",
                   "carreira","confiança","autoconfiança"],
    "reflexivo":  ["repensar","refletir","reflexivo","reflexão","pensar","questionar","buscar",
                   "entender","descobrir","crescer","mudar","recomeçar","propósito","significado",
                   "identidade","autoconhecimento","curioso","curiosidade","filosófico",
                   "filosofia","sentido","introspecção","meditação","consciência","existencial"],
    "romântico":  ["amor","apaixonado","apaixonada","romântico","romântica","relacionamento",
                   "paixão","quero romance","coração partido","término","separação","ex",
                   "me apaixonei","namoro","namorado","namorada"],
    "entediado":  ["entediado","entediada","entedio","tédio","monótono","monotonia","cansado",
                   "cansada","rotina","sem graça","mesmice","parado","estagnado"],
    "nostálgico": ["nostalgia","nostálgico","nostálgica","infância","passado","memórias",
                   "lembranças","tempos antigos","saudade do passado"],
}

OPOSTOS = {
    "feliz": "triste", "triste": "feliz", "ansioso": "reflexivo",
    "inspirado": "entediado", "entediado": "inspirado", "romântico": "reflexivo",
}

NEGACOES = {"não", "nao", "nunca", "jamais", "nem", "sem"}

CRISIS_KEYWORDS = [
    "me matar","quero morrer","não quero mais viver","nao quero mais viver",
    "tirar minha vida","me suicidar","suicídio","suicidio",
    "me machucar","me ferir","acabar com tudo","não vale a pena viver",
    "pensar em morrer","pensamentos de morte",
]

CRISIS_RESPOSTA = (
    "Percebi que sua mensagem pode indicar um momento muito difícil. "
    "Recomendações de livros podem esperar — o mais importante agora é você. "
    "Por favor, entre em contato com o CVV (Centro de Valorização da Vida): "
    "ligue 188 (24h, gratuito) ou acesse cvv.org.br. "
    "Você não precisa passar por isso sozinho."
)

def detectar_crise(texto):
    return any(kw in texto.lower() for kw in CRISIS_KEYWORDS)

def _normalizar_kw(texto):
    """Normaliza unicode NFC e lowercase — usado para comparar keywords."""
    return unicodedata.normalize("NFC", str(texto).lower().strip())

# Pré-computa keywords normalizadas uma vez só (fora da função, no módulo)
_EMOCOES_KW_NORM = {
    emocao: [_normalizar_kw(kw) for kw in kws]
    for emocao, kws in EMOCOES_KEYWORDS.items()
}

def detectar_emocoes(texto):
    texto_limpo = limpar_texto(texto)
    # Também normaliza NFC diretamente no texto original para pegar variações
    texto_nfc   = _normalizar_kw(texto)
    contagem    = {}

    segmentos = re.split(r'[,;.]|\b(?:mas|porém|contudo|entretanto|embora)\b', texto_limpo)
    segmentos = [s.strip() for s in segmentos if s.strip()]
    if not segmentos:
        segmentos = [texto_limpo]

    for seg in segmentos:
        palavras = set(seg.split())
        negado   = bool(palavras & NEGACOES)
        for emocao, keywords_norm in _EMOCOES_KW_NORM.items():
            hits = 0
            for kw in keywords_norm:
                # Match exato por word boundary
                if re.search(r'\b' + re.escape(kw) + r'\b', seg):
                    hits += 1
                # Fallback: match parcial para palavras compostas como "estou com depressão"
                elif len(kw) >= 5 and kw in seg:
                    hits += 0.8
            if hits == 0:
                continue
            if negado:
                oposto = OPOSTOS.get(emocao)
                if oposto:
                    contagem[oposto] = contagem.get(oposto, 0) + hits * 0.5
            else:
                contagem[emocao] = contagem.get(emocao, 0) + hits

    if not contagem:
        return [("neutro", 1.0)]
    total    = sum(contagem.values())
    ordenado = sorted(contagem.items(), key=lambda x: -x[1])
    return [(e, round(h / total, 3)) for e, h in ordenado]

def classificar_ambiguidade(texto, emocoes):
    if len(emocoes) == 1:
        return "simples"
    texto_limpo   = limpar_texto(texto)
    tem_contraste = any(c in texto_limpo for c in CONTRASTES)
    diferenca     = abs(emocoes[0][1] - emocoes[1][1])
    if tem_contraste and diferenca < 0.35:
        return "contraditório"
    if diferenca < 0.30:
        return "ambíguo"
    return "simples"

def expandir_query_ambigua(texto, emocoes):
    texto_limpo = limpar_texto(texto)
    queries = []
    for emocao, _ in emocoes[:3]:
        perfil = MAPA_TERAPEUTICO.get(emocao, "")
        if perfil:
            q = perfil + " " + texto_limpo
        else:
            sinonimos = SINONIMOS.get(emocao, "")
            q = (texto_limpo + " " + sinonimos).strip() if sinonimos else texto_limpo
        queries.append(q.strip())
    return queries or [texto_limpo]

def formatar_label_emocoes(emocoes, tipo):
    nomes = [e for e, _ in emocoes[:3]]
    base  = " + ".join(nomes)
    if tipo == "contraditório": return f"{base} (contraditório)"
    if tipo == "ambíguo":       return f"{base} (misto)"
    return nomes[0]

# ============================================================
# 4. EMBEDDINGS COM CACHE
# ============================================================
MODELO_NOME = "paraphrase-multilingual-MiniLM-L12-v2"

def _hash_dataset(df):
    conteudo = '|'.join(
        str(t) + str(s) + str(e1) + str(e2) + str(sent)
        for t, s, e1, e2, sent in zip(
            df.get("Título do Livro", pd.Series()).fillna(""),
            df.get("Sinopse",         pd.Series()).fillna(""),
            df.get("Emoção I",        pd.Series()).fillna(""),
            df.get("Emoção II",       pd.Series()).fillna(""),
            df.get("Sentimento I",    pd.Series()).fillna(""),
        )
    )
    return hashlib.md5(conteudo.encode()).hexdigest()

def carregar_modelo_embeddings(df):
    modelo = SentenceTransformer(MODELO_NOME)
    df["perfil_emocional"] = df.apply(construir_perfil, axis=1)
    hash_atual = _hash_dataset(df)
    try:
        with open(CACHE_PATH, "rb") as f:
            cache = pickle.load(f)
        if cache.get("modelo") == MODELO_NOME and cache.get("hash_dataset") == hash_atual:
            logger.info(f"Embeddings do cache. Shape: {cache['embeddings'].shape}")
            return modelo, cache["embeddings"]
        logger.warning("Cache desatualizado. Recomputando...")
    except Exception:
        pass
    logger.info("Computando embeddings (primeira vez ~30s)...")
    embs = modelo.encode(df["perfil_emocional"].tolist(),
                         show_progress_bar=True, batch_size=32, normalize_embeddings=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump({"modelo": MODELO_NOME, "hash_dataset": hash_atual, "embeddings": embs}, f)
    logger.info(f"Embeddings salvos. Shape: {embs.shape}")
    return modelo, embs

# ============================================================
# 5. CLASSIFICADOR — fallback TF-IDF + Regressão Logística
# ============================================================
def treinar_classificador(df):
    cols_feat = ["Gênero", "Emoção I", "Emoção II", "Sinopse"]
    df_clf    = df[cols_feat + ["Sentimento I"]].dropna(subset=["Sentimento I"])
    contagem  = df_clf["Sentimento I"].value_counts()
    df_clf    = df_clf[df_clf["Sentimento I"].isin(contagem[contagem >= 2].index)]

    def montar_features(row):
        genero   = limpar_texto(str(row.get("Gênero",   "")))
        emocao1  = limpar_texto(str(row.get("Emoção I", "")))
        emocao2  = limpar_texto(str(row.get("Emoção II","")))
        sinopse  = limpar_texto(str(row.get("Sinopse",  ""))[:200])
        # Repetir campos mais importantes para dar mais peso no TF-IDF
        return f"{genero} {genero} {emocao1} {emocao1} {emocao2} {sinopse}"

    X = df_clf.apply(montar_features, axis=1)
    y = df_clf["Sentimento I"]
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=5000)),
        ("clf",   LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")),
    ])
    n, n_classes = len(df_clf), y.nunique()
    test_n = max(int(n * 0.2), n_classes + 1)
    if n < 10 or test_n >= int(n * 0.6):
        pipeline.fit(X, y)
        logger.info(f"Classificador treinado ({n} amostras, {n_classes} classes). "
                    f"Dataset pequeno — métricas requerem mais dados.")
    else:
        n_folds = min(3, n // (n_classes + 1))
        if n_folds >= 2:
            try:
                scores = cross_val_score(pipeline, X, y, cv=n_folds, scoring="accuracy")
                logger.info(f"Classificador CV ({n_folds}-fold): "
                            f"accuracy={scores.mean():.2f} ± {scores.std():.2f}")
            except Exception as e:
                logger.warning(f"Cross-val ignorada: {e}")
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_n, random_state=42, stratify=y)
        pipeline.fit(X_tr, y_tr)
        y_pred = pipeline.predict(X_te)
        acc  = accuracy_score(y_te, y_pred)
        prec = precision_score(y_te, y_pred, average="weighted", zero_division=0)
        rec  = recall_score(y_te, y_pred, average="weighted", zero_division=0)
        f1   = f1_score(y_te, y_pred, average="weighted", zero_division=0)
        logger.info(
            f"Classificador treinado ({n} amostras, {n_classes} classes) | "
            f"Accuracy={acc:.2f} | Precision={prec:.2f} | Recall={rec:.2f} | F1={f1:.2f}"
        )
    return pipeline

# ============================================================
# 6. MMR
# ============================================================
def mmr_rerank(query_emb, candidatos_idx, embeddings, top_k=10, lambda_param=0.7):
    selecionados, candidatos = [], list(candidatos_idx)
    for _ in range(min(top_k, len(candidatos))):
        if not candidatos:
            break
        sims_q = cosine_similarity(query_emb.reshape(1, -1), embeddings[candidatos])[0]
        sims_s = (
            cosine_similarity(embeddings[candidatos], embeddings[selecionados]).max(axis=1)
            if selecionados else np.zeros(len(candidatos))
        )
        melhor = int(np.argmax(lambda_param * sims_q - (1 - lambda_param) * sims_s))
        selecionados.append(candidatos.pop(melhor))
    return selecionados

# ============================================================
# 7. RECOMENDAÇÃO — pesos adaptativos por emoção dominante
# ============================================================
# Lambda MMR por tipo de sentimento:
#   simples      → 0.80  (alta relevância, menos diversidade necessária)
#   ambíguo      → 0.60  (equilíbrio entre relevância e diversidade)
#   contraditório → 0.50  (máxima diversidade para capturar tensão emocional)
MMR_LAMBDA = {"simples": 0.80, "ambíguo": 0.60, "contraditório": 0.50}

# Pesos semântico × rating adaptativos por emoção dominante:
#   emoções "pesadas" (triste, ansioso) priorizam semântica para garantir match terapêutico.
#   emoções "leves" (feliz, inspirado) toleram mais influência do rating.
PESOS_POR_EMOCAO = {
    "triste":     (0.90, 0.10),
    "ansioso":    (0.90, 0.10),
    "reflexivo":  (0.85, 0.15),
    "nostálgico": (0.80, 0.20),
    "romântico":  (0.75, 0.25),
    "entediado":  (0.70, 0.30),
    "inspirado":  (0.70, 0.30),
    "feliz":      (0.70, 0.30),
    "neutro":     (0.70, 0.30),
}
def _calcular_pre_filtro(n):
    if n <= 50:   return n
    if n <= 200:  return max(30, n // 2)
    if n <= 1000: return 60
    return min(200, n // 8)

def _normalizar_genero(valor):
    return unicodedata.normalize("NFKC", str(valor)).strip().casefold()


def recomendar_livros(query, emocoes, tipo_sentimento, top_k=10,
                      peso_semantico=None, peso_rating=None,
                      usar_mmr=True, lambda_mmr=None, genero=None,
                      paginas_max=None, titulos_excluidos=None):
    """Filtra o catálogo antes do ranking, preservando a correspondência dos embeddings."""
    elegiveis = pd.Series(True, index=df.index)
    if genero:
        if "Gênero" not in df.columns:
            return pd.DataFrame()
        elegiveis &= df["Gênero"].fillna("").map(_normalizar_genero) == _normalizar_genero(genero)
    if paginas_max is not None:
        if "Páginas" not in df.columns:
            return pd.DataFrame()
        paginas = pd.to_numeric(df["Páginas"], errors="coerce")
        elegiveis &= paginas.notna() & (paginas > 0) & (paginas <= paginas_max)
    if titulos_excluidos:
        elegiveis &= ~df["Título do Livro"].isin(titulos_excluidos)
    indices_elegiveis = np.flatnonzero(elegiveis.to_numpy()).tolist()
    if not indices_elegiveis:
        return pd.DataFrame()

    emocao_dom = emocoes[0][0] if emocoes else "neutro"
    if peso_semantico is None or peso_rating is None:
        peso_semantico, peso_rating = PESOS_POR_EMOCAO.get(emocao_dom, (0.70, 0.30))
    if lambda_mmr is None:
        lambda_mmr = MMR_LAMBDA.get(tipo_sentimento, 0.70)
    pre_filtro = _calcular_pre_filtro(len(indices_elegiveis))
    queries_exp = expandir_query_ambigua(query, emocoes)
    pesos_emoc = [p for _, p in emocoes[:len(queries_exp)]]
    total_peso = sum(pesos_emoc)
    pesos_norm = [p / total_peso for p in pesos_emoc] if total_peso > 0 else [1 / len(queries_exp)] * len(queries_exp)
    embs_q = modelo.encode(queries_exp, normalize_embeddings=True)
    if len(embs_q) == 1:
        emb_final = embs_q[0]
    else:
        emb_final = sum(e * p for e, p in zip(embs_q, pesos_norm))
        norma = np.linalg.norm(emb_final)
        if norma > 0:
            emb_final = emb_final / norma
    sims = cosine_similarity(emb_final.reshape(1, -1), embeddings_livros)[0]
    score_hibrido = peso_semantico * sims + peso_rating * df["rating_norm"].values
    df_res = df.copy()
    df_res["_score_sem"] = sims
    df_res["_score_final"] = score_hibrido
    ordem = np.argsort(score_hibrido[indices_elegiveis])[::-1][:pre_filtro]
    top_idx = [indices_elegiveis[i] for i in ordem]
    idx_final = (
        mmr_rerank(emb_final, top_idx, embeddings_livros, top_k=top_k, lambda_param=lambda_mmr)
        if usar_mmr and len(top_idx) > top_k else top_idx[:top_k]
    )
    resultado = df_res.iloc[idx_final].copy()
    if resultado.empty:
        return resultado

    # O fallback também respeita gênero, páginas e títulos excluídos.
    if resultado["_score_sem"].max() < 0.3 and "Sentimento I" in df_res.columns:
        emocao_pred = pipeline_clf.predict([limpar_texto(query)])[0]
        alvo = emocao_pred if emocao_dom == "neutro" else emocao_dom
        pool = df_res.iloc[indices_elegiveis]
        fallback = pool[pool["Sentimento I"].fillna("").map(_normalizar_genero) == _normalizar_genero(alvo)].nlargest(top_k, "rating_norm")
        if not fallback.empty:
            resultado = fallback.copy()

    def gerar_explicacao(row):
        sentimento = row.get("Sentimento I", "") or row.get("Emoção I", "")
        sim_pct = round(row["_score_sem"] * 100, 1)
        return f"Match {sim_pct}% — {sentimento}, {row.get('Gênero', '')}"
    resultado["explicacao"] = resultado.apply(gerar_explicacao, axis=1)
    colunas = [c for c in [
        "Título do Livro", "Gênero", "Sentimento I", "Emoção I", "Emoção II",
        "Sinopse", "Público-Alvo", "Páginas", "Tamanho do Livro",
        "rating", "_score_sem", "_score_final", "explicacao",
    ] if c in resultado.columns]
    return resultado[colunas].reset_index(drop=True)


# ============================================================
# 8. ANÁLISE DE IA — Google Gemini (GRATUITO)
# Chave em: aistudio.google.com/apikey
# Variável: set GEMINI_API_KEY=AIzaSy...  (CMD Windows)
# ============================================================
_API_KEY        = os.environ.get("GEMINI_API_KEY", "").strip()
_API_KEY_VALIDA = bool(_API_KEY)

_cache_analise: dict = {}

def _chave_cache(query, titulo):
    return hashlib.md5(f"{query}|{titulo}".encode()).hexdigest()

def analisar_top_livro(query, livro, tipo_sentimento="simples"):
    if not _API_KEY_VALIDA:
        return ""

    titulo = livro.get("Título do Livro", "N/A")
    chave  = _chave_cache(query, titulo)
    if chave in _cache_analise:
        return _cache_analise[chave]

    try:
        import google.generativeai as genai

        genai.configure(api_key=_API_KEY)
        modelo_ia = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            system_instruction=(
                "Você é um terapeuta literário especialista em biblioterapia. "
                "Conecte livros ao estado emocional do leitor de forma empática e motivadora. "
                "Escreva sempre em português brasileiro. Nunca use bullet points."
            ),
        )

        campos = {
            "Título":       titulo,
            "Gênero":       livro.get("Gênero", "N/A"),
            "Sentimento I": livro.get("Sentimento I", "N/A"),
            "Emoção I":     livro.get("Emoção I", "N/A"),
            "Emoção II":    livro.get("Emoção II", "N/A"),
            "Sinopse":      livro.get("Sinopse", "N/A"),
            "Páginas":      livro.get("Páginas", "N/A"),
            "Match":        f"{round(livro.get('_score_sem', 0) * 100, 1)}%",
        }
        livro_str = "\n".join(f"  {k}: {v}" for k, v in campos.items())

        instrucao_extra = {
            "ambíguo":       "\nO usuário tem sentimentos MISTOS. Reconheça essa complexidade.",
            "contraditório": "\nO usuário tem sentimentos CONTRADITÓRIOS. Reconheça essa tensão.",
        }.get(tipo_sentimento, "")

        prompt = f"""O usuário descreveu seu estado emocional:
"{query}"{instrucao_extra}

Livro selecionado como #1:
{livro_str}

Escreva UMA análise curta (3 a 5 frases) explicando por que este livro é ideal para o momento do usuário.
Comece EXATAMENTE com: "Esse livro se encaixa com você porque..."
Apenas prosa fluida, sem bullet points."""

        response  = modelo_ia.generate_content(prompt)
        resultado = response.text.strip()
        _cache_analise[chave] = resultado
        return resultado

    except Exception as e:
        erro = str(e)
        logger.error(f"Erro na API Gemini: {erro}")
        if "api_key" in erro.lower() or "invalid" in erro.lower():
            return "⚠️ Chave Gemini inválida."
        if "quota" in erro.lower() or "rate" in erro.lower():
            return "⚠️ Limite de requisições atingido. Tente em instantes."
        return ""

# ============================================================
# 9. FEEDBACK
# ============================================================
def registrar_feedback(titulo, util, query):
    if titulo not in _titulos_validos:
        return False, "Título não encontrado."
    try:
        with open(FEEDBACK_PATH, "r", encoding="utf-8") as f:
            dados = json.load(f)
    except FileNotFoundError:
        dados = []
    dados.append({"titulo": titulo, "util": util, "query": query,
                  "ts": pd.Timestamp.now().isoformat()})
    with open(FEEDBACK_PATH, "w", encoding="utf-8") as f:
        json.dump(dados, f, ensure_ascii=False, indent=2)
    return True, "ok"

def livros_penalizados():
    """
    Retorna livros com score líquido abaixo do limite configurado em Config.FEEDBACK_MIN_SCORE.
    Evita exclusão por um único feedback negativo acidental.
    """
    try:
        with open(FEEDBACK_PATH, "r", encoding="utf-8") as f:
            dados = json.load(f)
    except FileNotFoundError:
        return set()
    contagem = {}
    for item in dados:
        t = item["titulo"]
        contagem[t] = contagem.get(t, 0) + (1 if item["util"] else -1)
    return {t for t, s in contagem.items() if s <= Config.FEEDBACK_MIN_SCORE}

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
    return sorted([{"titulo": t, "score": s} for t, s in contagem.items()], key=lambda x: -x["score"])

# ============================================================
# 10. HTML
# ============================================================
HTML = r"""<!DOCTYPE html>
<html lang="pt-BR" data-theme="light">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<meta name="color-scheme" content="light dark">
<meta name="theme-color" content="#f7f8f3">
<meta name="description" content="Emotional Book: descubra livros que conversam com o seu momento.">
<title>Emotional Book · Histórias para o seu momento</title>
<style>
/* Interface independente de fontes, imagens e bibliotecas externas. */
:root{--bg:#f7f8f3;--surface:#fffefa;--surface-2:#f0f2eb;--ink:#283f37;--muted:#606e65;--line:#dce3d8;--accent:#396550;--accent-hover:#294c3c;--sage:#e7efdf;--lilac:#efebf6;--pink:#f7e9e5;--yellow:#f5efd8;--shadow:0 10px 35px rgba(49,70,55,.035);--error:#a0443d;--error-bg:#fae9e5;--focus:#628063;--radius:22px}
[data-theme="dark"]{--bg:#18221d;--surface:#222e27;--surface-2:#29372e;--ink:#eef2e8;--muted:#b9c4b7;--line:#3f5044;--accent:#b6ceaa;--accent-hover:#d0e0c7;--sage:#344536;--lilac:#40384a;--pink:#4a3835;--yellow:#454230;--shadow:0 10px 35px rgba(0,0,0,.08);--error:#ffc0b4;--error-bg:#48332e;--focus:#cadfc0}
*{box-sizing:border-box}body,h1,h2,h3,p{margin:0}button,input,textarea,select{font:inherit}button,a,input,select,textarea{-webkit-tap-highlight-color:transparent}button{cursor:pointer}button:disabled{cursor:wait;opacity:.6}button,a{touch-action:manipulation}button{color:inherit}a{color:inherit}button:focus-visible,a:focus-visible,input:focus-visible,select:focus-visible,textarea:focus-visible,summary:focus-visible{outline:3px solid var(--focus);outline-offset:4px}button{border:0;background:none}body{font-family:"Segoe UI",Arial,sans-serif;background:var(--bg);color:var(--ink);line-height:1.55;font-size:14px}svg{display:inline-block;width:20px;height:20px;vertical-align:middle;flex-shrink:0}svg.icon{fill:none;stroke:currentColor;stroke-width:1.6;stroke-linecap:round;stroke-linejoin:round}[hidden]{display:none!important}::selection{background:var(--sage);color:var(--ink)}.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}.skip-link{position:fixed;top:8px;left:8px;padding:12px;background:var(--surface);z-index:99;transform:translateY(-150%)}.skip-link:focus{transform:none}.shell{max-width:1192px;margin:auto;padding:0 40px}.app-header{border-bottom:1px solid var(--line)}.header-inner{min-height:96px;display:flex;align-items:center;justify-content:space-between;gap:24px}.brand{display:flex;align-items:center;gap:12px;text-decoration:none;flex-shrink:0}.brand-mark{width:42px;height:42px;display:grid;place-items:center;background:var(--accent);color:var(--surface);border-radius:13px}.brand-mark svg{width:23px;height:23px}.brand-name{display:block;font-family:Georgia,serif;font-size:21px;letter-spacing:-.7px;line-height:1.2}.brand-sub{display:block;font-size:9px;font-weight:600;letter-spacing:1.8px;color:var(--muted);margin-top:5px;text-transform:uppercase}.nav-group{display:flex;gap:7px;align-items:center}.nav-button{padding:10px 15px;border-radius:12px;font-size:13px;color:var(--muted);display:flex;align-items:center;gap:8px;white-space:nowrap}.nav-button:hover{background:var(--surface-2);color:var(--ink)}.nav-button.active{background:var(--sage);color:var(--ink);font-weight:600}.nav-count{font-size:10px;min-width:18px;height:18px;display:grid;place-items:center;border-radius:7px;background:var(--surface);color:var(--ink)}.header-tools{display:flex;gap:6px;align-items:center}.icon-button{display:grid;place-items:center;width:40px;height:40px;border:1px solid var(--line);border-radius:50%;background:var(--surface);color:var(--ink)}.icon-button:hover{background:var(--sage)}.demo-banner{background:var(--yellow);padding:10px 20px;text-align:center;font-size:12px;border-bottom:1px solid var(--line)}.hero{display:grid;grid-template-columns:1.2fr 1fr;align-items:center;min-height:343px;gap:45px;padding:44px 0 36px}.eyebrow{display:flex;align-items:center;gap:8px;text-transform:uppercase;letter-spacing:1.8px;font-size:10px;font-weight:600;color:var(--accent)}.eyebrow .dot{width:6px;height:6px;background:currentColor;border-radius:100%}.hero h1{font:normal clamp(35px,3.6vw,48px)/1.14 Georgia,serif;letter-spacing:-1.8px;margin:18px 0 16px;max-width:550px}.hero h1 em{font-weight:normal;color:var(--accent)}.hero-description{max-width:400px;font-size:14px;line-height:1.8;color:var(--muted)}.hero-bottom{display:flex;gap:9px;align-items:center;margin-top:23px;font-size:11px;color:var(--muted)}.hero-bottom svg{width:16px;height:16px}.art-scene{position:relative;height:260px;display:grid;place-items:center;isolation:isolate}.art-orbit{position:absolute;width:308px;height:238px;border-radius:49% 51% 46% 54%;background:var(--sage);transform:rotate(-9deg);z-index:-2}.art-orbit:after{content:"";position:absolute;inset:-18px 21px 11px -23px;border:1px solid var(--line);border-radius:50%;transform:rotate(22deg)}.art-book{position:absolute;width:124px;height:169px;border-radius:3px 10px 10px 3px;box-shadow:8px 13px 20px rgba(42,62,45,.12);border-left:8px solid rgba(0,0,0,.09);display:flex;flex-direction:column;justify-content:space-between;padding:19px 16px 15px;color:#3e5246}.art-book:after{content:"";position:absolute;right:4px;left:0;bottom:-5px;height:6px;background:repeating-linear-gradient(0deg,#dedfd1 0px,#dedfd1 1px,#f7f6ec 1px,#f7f6ec 2px);border-radius:0 0 5px 2px;z-index:-1}.art-book.one{background:#d6d8b6;transform:translate(-84px,6px) rotate(-16deg)}.art-book.two{background:#ded4e9;transform:translate(67px,-8px) rotate(13deg);color:#504b63}.art-book.three{background:#ebd4c6;transform:translate(-7px,20px) rotate(-3deg);z-index:2}.art-book .tiny{font-size:6px;letter-spacing:2px;text-transform:uppercase}.art-book .book-art-title{font:19px/1.18 Georgia,serif;letter-spacing:-.7px}.art-book svg{width:29px;height:29px;opacity:.7}.art-flower{position:absolute;right:29px;top:14px;transform:rotate(8deg);color:var(--accent);width:48px;height:48px}.art-spark{position:absolute;left:13px;top:30px;transform:rotate(-9deg);color:var(--accent);width:25px;height:25px}.art-caption{position:absolute;right:4px;bottom:0;z-index:3;padding:10px 16px;background:var(--surface);border:1px solid var(--line);border-radius:14px;font:italic 12px Georgia,serif;box-shadow:var(--shadow)}.art-caption svg{width:14px;height:14px;margin-right:7px;color:var(--accent)}.workspace{display:grid;grid-template-columns:minmax(0,1fr) 295px;align-items:start;gap:23px;padding-bottom:36px}.panel{background:var(--surface);border:1px solid var(--line);border-radius:var(--radius);box-shadow:var(--shadow)}.discovery-panel{padding:27px 29px 24px}.panel-heading{display:flex;gap:15px;justify-content:space-between;align-items:flex-start}.step-label{font-size:9px;font-weight:600;letter-spacing:1.5px;color:var(--muted);text-transform:uppercase;margin-bottom:6px}h2{font:normal 27px/1.3 Georgia,serif;letter-spacing:-.7px}.panel-subtitle{color:var(--muted);font-size:12px;margin:7px 0 20px}.gentle-badge{display:inline-flex;align-items:center;gap:6px;font-size:10px;color:var(--muted);border:1px solid var(--line);border-radius:20px;padding:5px 10px;white-space:nowrap;margin-top:4px}.gentle-badge svg{width:12px;height:12px}.field-label{display:block;font-size:11px;font-weight:600;margin-bottom:9px}.field-hint{font-size:10px;color:var(--muted);font-weight:400;display:inline;margin-left:4px}.mood-grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:8px;margin-bottom:21px}.mood-button{display:flex;align-items:center;justify-content:center;gap:6px;border:1px solid var(--line);border-radius:10px;padding:11px 3px;font-size:11px;background:var(--surface);transition:background .15s,border-color .15s,transform .15s}.mood-button:hover{background:var(--surface-2);transform:translateY(-1px)}.mood-button svg{width:16px;height:16px}.mood-button[aria-pressed="true"]{background:var(--mood-color,var(--sage));border-color:var(--accent);box-shadow:inset 0 0 0 .5px var(--accent)}.text-input-wrap{position:relative}.prompt-input{display:block;width:100%;min-height:118px;resize:vertical;border:1px solid var(--line);border-radius:13px;padding:13px 15px 30px;background:var(--bg);color:var(--ink);font-size:12px;line-height:1.7}.prompt-input::placeholder{color:var(--muted);opacity:.85}.char-counter{position:absolute;bottom:9px;right:13px;font-size:9px;color:var(--muted);pointer-events:none}.example-row{display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin:11px 0 22px}.example-label{font-size:9px;color:var(--muted);margin-right:3px}.example-button{font-size:9px;border-radius:7px;padding:4px 7px;background:var(--surface-2);color:var(--muted)}.example-button:hover{background:var(--sage);color:var(--ink)}.preference-toggle{display:flex;align-items:center;gap:8px;font-size:11px;font-weight:600;cursor:pointer;padding:14px 0 12px;border-top:1px solid var(--line);list-style:none}.preference-toggle::-webkit-details-marker{display:none}.preference-toggle svg{width:15px;height:15px}.preference-toggle .chevron{margin-left:auto;transition:transform .2s}details[open]>.preference-toggle .chevron{transform:rotate(180deg)}.preferences-grid{display:grid;grid-template-columns:1.25fr 1fr .7fr;gap:10px;padding:0 0 14px}.preferences-grid label{font-size:10px;color:var(--muted);display:block;margin-bottom:5px}.select-wrap{position:relative}.select-wrap:after{content:"";position:absolute;right:13px;top:15px;width:6px;height:6px;border-right:1.5px solid var(--muted);border-bottom:1.5px solid var(--muted);transform:rotate(45deg);pointer-events:none}select{width:100%;appearance:none;background:var(--surface);border:1px solid var(--line);border-radius:9px;color:var(--ink);font-size:11px;min-height:38px;padding:8px 28px 8px 10px}select option{background:var(--surface);color:var(--ink)}.variety-row{display:flex;align-items:center;gap:10px;margin-bottom:6px;flex-wrap:wrap}.variety-row>span{font-size:10px;color:var(--muted)}.segmented{display:inline-flex;gap:2px;padding:3px;border-radius:10px;background:var(--surface-2);flex-wrap:wrap}.segment-button{font-size:9px;padding:5px 9px;border-radius:7px;color:var(--muted)}.segment-button[aria-pressed="true"]{background:var(--surface);color:var(--ink);box-shadow:0 1px 4px rgba(0,0,0,.06)}.form-footer{display:flex;align-items:center;justify-content:space-between;gap:10px;margin-top:20px}.primary-button{display:inline-flex;align-items:center;justify-content:center;gap:10px;background:var(--accent);color:var(--surface);border-radius:11px;padding:13px 19px;font-size:12px;font-weight:600;min-height:45px}.primary-button:hover:not(:disabled){background:var(--accent-hover);transform:translateY(-1px)}.primary-button svg{width:17px;height:17px}.quiet-button{display:inline-flex;align-items:center;gap:6px;font-size:11px;color:var(--muted);border-radius:8px;padding:8px}.quiet-button:hover{background:var(--surface-2);color:var(--ink)}.quiet-button svg{width:14px;height:14px}.shortcut-hint{font-size:9px;color:var(--muted);margin-top:12px}.status-message{font-size:12px;line-height:1.65;border-radius:12px;padding:12px 14px;background:var(--error-bg);color:var(--error);margin-top:16px;white-space:pre-line}.loading-note{display:flex;gap:8px;align-items:center;font-size:11px;color:var(--muted);margin-top:15px}.loading-dot{width:14px;height:14px;border:2px solid var(--line);border-top-color:var(--accent);border-radius:50%;animation:spin 1s linear infinite}@keyframes spin{to{transform:rotate(360deg)}}.side-column{display:grid;gap:16px}.side-panel{padding:22px}.side-icon{width:34px;height:34px;border-radius:11px;display:grid;place-items:center;background:var(--lilac);margin-bottom:17px}.side-icon svg{width:17px;height:17px}.side-panel h3{font:normal 20px/1.3 Georgia,serif;letter-spacing:-.4px}.side-panel p{font-size:12px;color:var(--muted);line-height:1.8;margin:10px 0 17px}.surprise-button{display:flex;justify-content:center;align-items:center;gap:8px;padding:10px;font-size:11px;border:1px solid var(--line);border-radius:9px;width:100%;background:var(--surface)}.surprise-button:hover{background:var(--lilac)}.surprise-button svg{width:15px;height:15px}.shelf-note{background:var(--sage);padding:21px;border-radius:var(--radius);border:1px solid var(--line)}.shelf-note-top{display:flex;justify-content:space-between;align-items:center;margin-bottom:13px}.shelf-note-top>svg{width:19px;height:19px}.shelf-number{font:normal 28px Georgia,serif}.shelf-note h3{font-size:12px;font-weight:600}.shelf-note p{font-size:11px;color:var(--muted);margin:7px 0 12px;line-height:1.7}.inline-button{font-size:10px;display:inline-flex;align-items:center;gap:7px;padding:3px 0;font-weight:600}.inline-button svg{width:13px;height:13px}.reading-note{display:flex;gap:10px;padding:0 5px;color:var(--muted);font-size:10px;line-height:1.7}.reading-note svg{width:15px;height:15px;margin-top:2px}.results-section{margin:0 0 40px;border-top:1px solid var(--line);padding-top:29px}.section-head{display:flex;align-items:flex-end;justify-content:space-between;gap:18px;margin-bottom:21px}.section-head h2{font-size:29px}.section-description{font-size:12px;color:var(--muted);margin-top:8px}.emotion-note{display:flex;gap:7px;align-items:center;font-size:10px;background:var(--lilac);padding:8px 11px;border-radius:11px;max-width:300px}.emotion-note svg{width:14px;height:14px}.result-tools{display:flex;align-items:center;gap:12px;margin-bottom:20px;flex-wrap:wrap}.search-wrap{position:relative;flex:1;min-width:190px}.search-wrap>svg{position:absolute;left:12px;top:12px;width:15px;height:15px;color:var(--muted)}.search-input{width:100%;height:39px;background:var(--surface);color:var(--ink);border:1px solid var(--line);border-radius:10px;padding:9px 12px 9px 36px;font-size:11px}.sort-wrap{min-width:160px}.view-controls{display:flex;gap:3px;padding:3px;border:1px solid var(--line);border-radius:9px;background:var(--surface)}.view-controls button{padding:6px;border-radius:6px;display:grid;place-items:center}.view-controls svg{width:15px;height:15px}.view-controls button[aria-pressed="true"]{background:var(--sage)}.book-grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}.book-card{overflow:hidden;transition:transform .18s,box-shadow .18s;display:flex;flex-direction:column;position:relative}.book-card:hover{transform:translateY(-3px);box-shadow:0 12px 32px rgba(38,60,42,.07)}.cover-area{height:190px;background:var(--cover-bg);display:grid;place-items:center;position:relative;overflow:hidden;color:var(--cover-ink)}.cover-area:before{content:"";width:210px;height:210px;border:1px solid currentColor;opacity:.1;border-radius:50%;position:absolute;left:-45px;top:20px}.cover-area:after{content:"";position:absolute;right:-10px;top:-50px;width:140px;height:140px;border:1px solid currentColor;opacity:.15;border-radius:50%}.book-object{width:103px;height:139px;padding:14px 10px 10px 14px;border-radius:2px 5px 5px 2px;background:var(--cover);border-left:5px solid rgba(0,0,0,.08);box-shadow:6px 7px 0 rgba(255,255,255,.48),10px 13px 14px rgba(32,43,33,.1);position:relative;display:flex;flex-direction:column;justify-content:space-between;transform:rotate(-5deg);z-index:1}.book-object:before{content:"";position:absolute;inset:8px;border:1px solid currentColor;opacity:.24;pointer-events:none}.book-object>svg{height:17px;width:17px;align-self:center}.cover-genre{font-size:5px;letter-spacing:1.3px;text-align:center;text-transform:uppercase;max-height:14px;overflow:hidden}.cover-title{font:13px/1.2 Georgia,serif;text-align:center;display:-webkit-box;-webkit-line-clamp:4;-webkit-box-orient:vertical;overflow:hidden;overflow-wrap:anywhere}.cover-foot{font-size:5px;letter-spacing:.8px;text-align:center;text-transform:uppercase}.save-button{position:absolute;right:12px;top:12px;width:32px;height:32px;display:grid;place-items:center;border-radius:50%;background:var(--surface);color:var(--ink);border:1px solid var(--line);z-index:3}.save-button svg{width:15px;height:15px}.save-button[aria-pressed="true"]{background:var(--accent);color:var(--surface);border-color:var(--accent)}.save-button[aria-pressed="true"] svg{fill:currentColor}.top-pick{position:absolute;left:12px;top:13px;font-size:8px;font-weight:600;padding:4px 7px;border-radius:6px;background:var(--surface);color:var(--ink);z-index:2}.book-content{padding:18px 20px 17px;display:flex;flex-direction:column;flex:1;gap:10px}.book-meta-line{display:flex;align-items:center;justify-content:space-between;gap:8px;font-size:9px;color:var(--muted)}.book-genre{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.rating{display:flex;align-items:center;gap:3px;white-space:nowrap}.rating svg{width:11px;height:11px;color:var(--accent)}.book-card h3{font:normal 21px/1.3 Georgia,serif;letter-spacing:-.4px;overflow-wrap:anywhere}.book-synopsis{font-size:11px;color:var(--muted);line-height:1.8;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}.book-tags{display:flex;gap:5px;flex-wrap:wrap;margin:2px 0 0}.book-tag{font-size:8px;padding:4px 7px;border-radius:6px;background:var(--surface-2);color:var(--muted)}.card-bottom{display:flex;align-items:center;justify-content:space-between;gap:9px;padding-top:13px;margin-top:auto;border-top:1px solid var(--line)}.affinity{font-size:9px;color:var(--accent);font-weight:600}.detail-button{display:flex;align-items:center;gap:6px;font-size:10px;padding:4px 0}.detail-button svg{width:13px;height:13px}.book-grid.list-view{grid-template-columns:1fr}.list-view .book-card{display:grid;grid-template-columns:150px minmax(0,1fr)}.list-view .cover-area{height:100%;min-height:220px}.list-view .book-content{gap:9px}.list-view .card-bottom{margin-top:7px}.results-summary{font-size:10px;color:var(--muted);margin:16px 0 0}.explanation-panel{padding:19px 22px;display:flex;align-items:flex-start;gap:13px;background:var(--sage);margin-bottom:21px;border-radius:16px;border:1px solid var(--line)}.explanation-panel>svg{width:20px;height:20px;margin-top:3px;color:var(--accent)}.explanation-panel h3{font-size:12px;font-weight:600;margin-bottom:7px}.explanation-panel p{font-size:12px;line-height:1.85;color:var(--muted)}.explanation-target{font-weight:600;color:var(--ink)}.empty-state{padding:35px 25px;text-align:center;border:1px dashed var(--line);border-radius:var(--radius);background:var(--surface)}.empty-state>svg{width:31px;height:31px;color:var(--muted);margin-bottom:12px}.empty-state h3{font:normal 22px Georgia,serif;margin-bottom:9px}.empty-state p{font-size:12px;color:var(--muted);max-width:390px;margin:0 auto 17px;line-height:1.8}.secondary-view{padding:46px 0 35px;min-height:650px}.secondary-view .section-head{align-items:center}.action-group{display:flex;align-items:center;gap:8px;flex-wrap:wrap}.secondary-button{display:inline-flex;align-items:center;gap:7px;font-size:11px;padding:10px 13px;border:1px solid var(--line);border-radius:10px;background:var(--surface)}.secondary-button:hover{background:var(--surface-2)}.secondary-button svg{width:14px;height:14px}.history-list{display:grid;gap:12px}.history-item{display:flex;align-items:center;justify-content:space-between;gap:20px;padding:19px 23px}.history-query{font-size:14px;line-height:1.6;margin-bottom:7px;overflow-wrap:anywhere}.history-meta{display:flex;gap:9px;flex-wrap:wrap;font-size:10px;color:var(--muted)}.history-meta span+span:before{content:"·";margin-right:9px}.history-actions{display:flex;gap:8px;flex-shrink:0}.history-actions .icon-button{width:34px;height:34px}.history-actions svg{width:14px;height:14px}.history-feedback{margin-top:35px}.history-feedback summary{cursor:pointer;font-size:12px;font-weight:600;padding:14px 0}.feedback-history-row{font-size:12px;display:flex;justify-content:space-between;gap:20px;padding:11px 0;border-bottom:1px solid var(--line)}.feedback-score{white-space:nowrap;color:var(--accent)}.app-footer{padding:24px 0 28px;border-top:1px solid var(--line);display:flex;align-items:flex-start;justify-content:space-between;gap:25px;font-size:10px;color:var(--muted)}.app-footer p+p{margin-top:4px}.footer-brand{font:normal 17px Georgia,serif;letter-spacing:-.4px;color:var(--ink);margin-bottom:7px}.footer-right{text-align:right}.footer-right span{display:inline-flex;align-items:center;gap:5px}.footer-right svg{width:13px;height:13px}.toast{position:fixed;bottom:25px;left:50%;transform:translate(-50%,15px);padding:13px 19px;border-radius:12px;background:var(--ink);color:var(--surface);font-size:12px;box-shadow:0 8px 30px rgba(0,0,0,.16);opacity:0;pointer-events:none;transition:opacity .2s,transform .2s;z-index:50;max-width:calc(100% - 32px);text-align:center}.toast.visible{opacity:1;transform:translate(-50%,0)}dialog{border:1px solid var(--line);border-radius:24px;max-width:640px;width:calc(100% - 32px);padding:0;background:var(--surface);color:var(--ink);box-shadow:0 30px 90px rgba(0,0,0,.2);max-height:85dvh;overscroll-behavior:contain}dialog::backdrop{background:rgba(30,47,38,.42);backdrop-filter:blur(4px)}.modal-close{position:absolute;top:16px;right:16px;z-index:2}.modal-body{padding:32px}.modal-header{padding-right:40px}.modal-header h2{font-size:31px;line-height:1.25;overflow-wrap:anywhere}.modal-header .step-label{margin-bottom:10px}.modal-subtitle{font-size:12px;color:var(--muted);margin-top:10px}.modal-tags{display:flex;gap:7px;flex-wrap:wrap;margin:19px 0}.modal-tags span{font-size:10px;padding:6px 10px;border-radius:8px;background:var(--surface-2)}.modal-section{border-top:1px solid var(--line);padding-top:20px;margin-top:20px}.modal-section h3{font-size:12px;margin-bottom:10px}.modal-section p{color:var(--muted);font-size:13px;line-height:1.85;white-space:pre-line;overflow-wrap:anywhere}.modal-actions{display:flex;align-items:center;gap:8px;margin:24px 0 6px;flex-wrap:wrap}.feedback-buttons{display:flex;gap:8px;align-items:center;margin-top:10px;flex-wrap:wrap}.feedback-button{font-size:11px;display:flex;gap:7px;align-items:center;border:1px solid var(--line);border-radius:9px;padding:9px 12px}.feedback-button svg{width:14px;height:14px}.feedback-button[aria-pressed="true"]{background:var(--sage);border-color:var(--accent)}.feedback-feedback{font-size:10px;color:var(--muted);margin-top:9px}.explain-list{list-style:none;padding:0;display:grid;gap:17px;margin:25px 0}.explain-list li{display:flex;gap:13px;align-items:flex-start}.number-dot{display:grid;place-items:center;width:29px;height:29px;background:var(--sage);border-radius:9px;font-size:12px;flex-shrink:0}.explain-list b{font-size:13px}.explain-list p{font-size:12px;color:var(--muted);margin-top:3px;line-height:1.8}.safety-text{font-size:10px;color:var(--muted);line-height:1.8;border-top:1px solid var(--line);padding-top:17px}.storage-note{font-size:10px;color:var(--muted);margin:16px 0 22px}.loading-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;margin-top:20px}.skeleton-card{height:270px;background:var(--surface-2);border:1px solid var(--line);border-radius:var(--radius);animation:pulse 1.4s ease-in-out infinite}@keyframes pulse{50%{opacity:.45}}
@media(min-width:1500px){.shell{max-width:1300px}.hero{min-height:370px}.hero h1{font-size:52px}.workspace{grid-template-columns:minmax(0,1fr) 320px}.mood-button{font-size:12px}.prompt-input{font-size:13px}}
@media(max-width:1000px){.shell{padding:0 26px}.workspace{grid-template-columns:minmax(0,1fr) 255px;gap:16px}.discovery-panel{padding:24px}.gentle-badge{display:none}.hero{gap:25px}.hero h1{font-size:41px}.mood-button{font-size:10px;gap:4px}.mood-button svg{width:14px;height:14px}.art-book.one{transform:translate(-67px,6px) rotate(-16deg)}.art-book.two{transform:translate(59px,-8px) rotate(13deg)}.art-orbit{width:260px}.header-inner{gap:10px}.nav-button{padding:9px 10px}.side-panel{padding:20px}.preferences-grid{grid-template-columns:1fr 1fr}.preferences-grid>div:last-child{grid-column:1 / -1;max-width:150px}.book-grid{gap:13px}.book-content{padding:16px}.cover-area{height:175px}.book-card h3{font-size:19px}.art-caption{right:-7px;font-size:11px}}
@media(max-width:900px){.header-inner{flex-wrap:wrap;min-height:auto;padding-top:20px;gap:16px}.brand-name{font-size:20px}.header-tools{margin-left:auto}.nav-group{order:3;width:100%;justify-content:space-between;padding-bottom:13px;gap:5px}.nav-button{justify-content:center;flex:1;font-size:11px;padding:9px 5px}.nav-button svg{width:15px;height:15px}.brand-sub{font-size:8px}.hero{grid-template-columns:1fr 220px;gap:12px;padding:32px 0 27px;min-height:285px}.hero h1{font-size:34px;letter-spacing:-1.3px;margin:15px 0}.hero-description{font-size:12px}.hero-bottom{font-size:9px;margin-top:16px}.eyebrow{font-size:8px;letter-spacing:1px}.art-scene{height:220px;transform:scale(.82);transform-origin:center}.art-caption{bottom:-4px;right:-13px}.art-flower{right:-10px}.art-spark{left:-13px}.workspace{grid-template-columns:1fr;gap:18px}.side-column{grid-template-columns:1fr 1fr;align-items:stretch}.side-panel{padding:20px}.side-icon{margin-bottom:12px;width:29px;height:29px}.side-panel h3{font-size:19px}.reading-note{grid-column:1 / -1;padding:0 4px}.mood-button{font-size:12px;gap:7px}.mood-button svg{width:16px;height:16px}.gentle-badge{display:inline-flex}.preferences-grid{grid-template-columns:1.2fr 1fr .7fr}.preferences-grid>div:last-child{grid-column:auto;max-width:none}.book-grid{grid-template-columns:repeat(2,minmax(0,1fr))}.section-head h2{font-size:26px}.emotion-note{max-width:220px}.app-footer{font-size:9px}.secondary-view{padding-top:30px}.secondary-view .section-head{flex-wrap:wrap}.history-item{padding:17px}.history-actions{gap:4px}.history-query{font-size:12px}.loading-grid{grid-template-columns:repeat(2,1fr)}}
@media(max-width:480px){.shell{padding:0 18px}.hero{grid-template-columns:1fr;padding:30px 0 28px;gap:4px;position:relative;min-height:auto}.hero-copy{max-width:85%;position:relative;z-index:2}.hero h1{font-size:36px;max-width:290px}.hero-description{font-size:12px;max-width:280px}.art-scene{display:none}.hero-bottom{font-size:10px}.discovery-panel{padding:23px 19px 21px}h2{font-size:25px}.gentle-badge{display:none}.panel-subtitle{font-size:11px}.mood-grid{gap:7px}.mood-button{font-size:10px;flex-direction:column;gap:6px;padding:10px 3px}.mood-button svg{width:18px;height:18px}.field-label{font-size:11px}.field-hint{display:block;margin:3px 0 0}.preferences-grid{grid-template-columns:1fr 1fr}.preferences-grid>div:last-child{grid-column:1 / -1;max-width:150px}.variety-row{gap:8px}.variety-row>span{font-size:9px}.segment-button{font-size:9px;padding:5px 7px}.example-row{gap:5px}.example-label{width:100%}.example-button{padding:5px 7px}.form-footer{align-items:center}.primary-button{padding:12px 16px;font-size:11px}.side-column{gap:11px}.side-panel{padding:17px}.side-panel p{font-size:11px;margin:8px 0 13px}.side-panel h3{font-size:18px}.shelf-note{padding:17px}.shelf-note p{font-size:10px}.surprise-button{font-size:10px;padding:9px 5px}.shelf-note h3{font-size:11px}.shelf-number{font-size:26px}.section-head{align-items:flex-start;flex-wrap:wrap;gap:12px}.section-head h2{font-size:27px}.emotion-note{max-width:100%}.book-grid{grid-template-columns:1fr;gap:16px}.book-card{display:grid;grid-template-columns:105px minmax(0,1fr)}.cover-area{height:100%;min-height:228px}.book-object{width:67px;height:100px;padding:10px 5px 8px 9px;border-left-width:4px;box-shadow:4px 5px 0 rgba(255,255,255,.48),6px 10px 12px rgba(32,43,33,.09)}.cover-title{font-size:10px;-webkit-line-clamp:5}.cover-genre,.cover-foot{font-size:4px}.book-object>svg{width:13px;height:13px}.book-object:before{inset:5px}.save-button{right:8px;top:8px;width:28px;height:28px}.top-pick{left:5px;top:auto;bottom:10px;font-size:6px;padding:4px}.book-card h3{font-size:20px}.book-content{padding:16px 14px;gap:9px}.book-synopsis{font-size:10px;-webkit-line-clamp:3}.book-genre{font-size:8px}.book-tags{gap:4px}.book-tag{font-size:7px}.card-bottom{flex-wrap:wrap;padding-top:10px;gap:6px}.affinity{font-size:8px}.detail-button{font-size:9px}.book-grid.list-view .book-card{grid-template-columns:90px minmax(0,1fr)}.sort-wrap{min-width:0;flex:1}.view-controls{margin-left:auto}.search-wrap{flex-basis:100%}.result-tools{gap:9px}.explanation-panel{padding:16px;gap:10px}.explanation-panel p{font-size:11px}.app-footer{flex-direction:column;gap:16px;padding:22px 0}.footer-right{text-align:left}.history-item{align-items:flex-start;flex-direction:column;gap:11px}.history-actions{width:100%;justify-content:flex-end}.modal-body{padding:25px 20px}.modal-header h2{font-size:27px}.modal-close{top:13px;right:13px;width:32px;height:32px}.modal-actions .primary-button{width:100%}.secondary-button{font-size:10px}.history-meta{font-size:9px}.toast{font-size:11px;width:max-content;max-width:calc(100% - 30px)}.loading-grid{grid-template-columns:1fr}.skeleton-card{height:210px}}
.thumb-down{transform:rotate(180deg)}
@media(max-width:480px){.save-button{left:8px;right:auto}.nav-group{gap:3px}.nav-button{gap:5px;font-size:10px;padding:9px 3px;min-width:0}.nav-button[data-view="shelf"]{flex:1.4}.nav-button svg{width:13px;height:13px}.nav-count{min-width:15px;height:15px;font-size:9px}.hero-copy{max-width:100%}.hero h1{max-width:none;font-size:clamp(27px,8.2vw,36px)}}
@media(max-width:360px){.brand-mark{width:36px;height:36px;border-radius:11px}.brand-name{font-size:17px}.brand-sub{font-size:7px;letter-spacing:1.2px}.brand{gap:9px}.header-tools .icon-button{width:32px;height:32px}.header-tools svg{width:17px;height:17px}.header-inner{gap:10px}.header-tools{gap:5px}}
@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto!important;animation:none!important;transition:none!important}.book-card:hover,.primary-button:hover,.mood-button:hover{transform:none}}
</style>
</head>
<body>
<a class="skip-link" href="#main-content">Ir para o conteúdo</a>
<svg aria-hidden="true" width="0" height="0" style="position:absolute;overflow:hidden"><defs>
<symbol id="i-book" viewBox="0 0 24 24"><path d="M12 5c-3-2-6-2-9-1v14c3-1 6-1 9 1 3-2 6-2 9-1V4c-3-1-6-1-9 1Z"/><path d="M12 5v14M6 8h3M15 8h3M6 11h3M15 11h3"/></symbol>
<symbol id="i-spark" viewBox="0 0 24 24"><path d="m12 3 2.6 6.4L21 12l-6.4 2.6L12 21l-2.6-6.4L3 12l6.4-2.6L12 3ZM20 2v4M18 4h4"/></symbol>
<symbol id="i-sun" viewBox="0 0 24 24"><circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M2 12h2M20 12h2M5 5l1.5 1.5M17.5 17.5 19 19M5 19l1.5-1.5M17.5 6.5 19 5"/></symbol>
<symbol id="i-moon" viewBox="0 0 24 24"><path d="M20 14A8.5 8.5 0 0 1 10 4 8.5 8.5 0 1 0 20 14Z"/></symbol>
<symbol id="i-wave" viewBox="0 0 24 24"><path d="M3 7c3-5 5 5 9 0s6 5 9 0M3 12c3-5 5 5 9 0s6 5 9 0M3 17c3-5 5 5 9 0s6 5 9 0"/></symbol>
<symbol id="i-cloud" viewBox="0 0 24 24"><path d="M6 15a4 4 0 0 1-.5-8A6 6 0 0 1 17 8a3.5 3.5 0 0 1 1 7H6ZM8 18l-1 2M13 18l-1 2M18 18l-1 2"/></symbol>
<symbol id="i-heart" viewBox="0 0 24 24"><path d="M20.8 4.6a5.5 5.5 0 0 0-7.8 0L12 5.7l-1.1-1.1a5.5 5.5 0 0 0-7.8 7.8L12 21l8.8-8.6a5.5 5.5 0 0 0 0-7.8Z"/></symbol>
<symbol id="i-clock" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></symbol>
<symbol id="i-compass" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="m16 8-2.5 5.5L8 16l2.5-5.5L16 8Z"/></symbol>
<symbol id="i-leaf" viewBox="0 0 24 24"><path d="M20 3C9 2 3 6 4 14c1 6 12 7 15-3l1-8ZM4 21 15 9"/></symbol>
<symbol id="i-arrow" viewBox="0 0 24 24"><path d="M5 12h14M13 6l6 6-6 6"/></symbol>
<symbol id="i-chevron" viewBox="0 0 24 24"><path d="m6 9 6 6 6-6"/></symbol>
<symbol id="i-bookmark" viewBox="0 0 24 24"><path d="M6 3h12v18l-6-4-6 4V3Z"/></symbol>
<symbol id="i-tune" viewBox="0 0 24 24"><path d="M3 6h7M14 6h7M3 12h2M9 12h12M3 18h12M19 18h2"/><circle cx="12" cy="6" r="2"/><circle cx="7" cy="12" r="2"/><circle cx="17" cy="18" r="2"/></symbol>
<symbol id="i-close" viewBox="0 0 24 24"><path d="m6 6 12 12M6 18 18 6"/></symbol>
<symbol id="i-search" viewBox="0 0 24 24"><circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 5 5"/></symbol>
<symbol id="i-star" viewBox="0 0 24 24"><path d="m12 3 2.7 5.8 6.3.8-4.6 4.5 1.2 6.3L12 17.5l-5.6 2.9 1.2-6.3L3 9.6l6.3-.8L12 3Z"/></symbol>
<symbol id="i-grid" viewBox="0 0 24 24"><rect x="3" y="3" width="7" height="7" rx="1"/><rect x="14" y="3" width="7" height="7" rx="1"/><rect x="3" y="14" width="7" height="7" rx="1"/><rect x="14" y="14" width="7" height="7" rx="1"/></symbol>
<symbol id="i-list" viewBox="0 0 24 24"><path d="M8 5h13M8 12h13M8 19h13M3 5h.1M3 12h.1M3 19h.1"/></symbol>
<symbol id="i-download" viewBox="0 0 24 24"><path d="M12 3v12m-5-5 5 5 5-5M5 16v5h14v-5"/></symbol>
<symbol id="i-copy" viewBox="0 0 24 24"><rect x="8" y="8" width="12" height="13" rx="2"/><path d="M16 8V3H3v13h5"/></symbol>
<symbol id="i-thumb" viewBox="0 0 24 24"><path d="M8 21H3V10h5M8 10l5-7c2 0 2 2 1 6h5a2 2 0 0 1 2 2l-2 8a2 2 0 0 1-2 2H8V10Z"/></symbol>
<symbol id="i-info" viewBox="0 0 24 24"><circle cx="12" cy="12" r="9"/><path d="M12 11v6M12 7h.01"/></symbol>
<symbol id="i-trash" viewBox="0 0 24 24"><path d="M3 6h18M9 6V3h6v3M5 6l1 15h12l1-15M10 10v7M14 10v7"/></symbol>
<symbol id="i-check" viewBox="0 0 24 24"><path d="m5 12 4 4L19 6"/></symbol>
<symbol id="i-flower" viewBox="0 0 24 24"><path d="M12 8c-5-10-10-2-5 3-10-1-7 9 1 5-2 10 8 10 8 1 8 6 12-3 4-6 6-7-3-12-7-4Z"/><circle cx="12" cy="12" r="2"/></symbol>
</defs></svg>
<div id="demo-banner" class="demo-banner" hidden>Prévia interativa · livros fictícios para demonstrar o design. O código Python usa o seu catálogo real.</div>
<header class="app-header"><div class="shell header-inner">
  <a class="brand" href="#descobrir" data-view="discover" aria-label="Emotional Book — início"><span class="brand-mark"><svg class="icon" aria-hidden="true"><use href="#i-book"/></svg></span><span><span class="brand-name">emotional book<span style="color:var(--accent)">.</span></span><span class="brand-sub">Histórias & sentimentos</span></span></a>
  <nav class="nav-group" aria-label="Navegação principal">
    <button class="nav-button active" data-view="discover" aria-current="page"><svg class="icon" aria-hidden="true"><use href="#i-compass"/></svg>Descobrir</button>
    <button class="nav-button" data-view="shelf"><svg class="icon" aria-hidden="true"><use href="#i-bookmark"/></svg>Minha estante<span id="nav-shelf-count" class="nav-count">0</span></button>
    <button class="nav-button" data-view="history"><svg class="icon" aria-hidden="true"><use href="#i-clock"/></svg>Histórico</button>
  </nav>
  <div class="header-tools"><button class="icon-button" id="help-button" aria-label="Como funciona" title="Como funciona"><svg class="icon" aria-hidden="true"><use href="#i-info"/></svg></button><button class="icon-button" id="theme-button" aria-label="Ativar modo escuro" title="Ativar modo escuro"><svg class="icon" aria-hidden="true"><use href="#i-moon"/></svg></button></div>
</div></header>
<main id="main-content" class="shell">
  <section id="discover-view" aria-label="Descobrir livros">
    <div class="hero">
      <div class="hero-copy"><p class="eyebrow"><span class="dot"></span>Uma boa história encontra você</p><h1>Um livro para cada<br>versão de <em>você.</em></h1><p class="hero-description">Nem todo dia pede a mesma história.<br>Encontre uma leitura que converse com o seu momento.</p><p class="hero-bottom"><svg class="icon" aria-hidden="true"><use href="#i-leaf"/></svg>Um pouco de escuta. Um mundo de possibilidades.</p></div>
      <div class="art-scene" aria-hidden="true"><div class="art-orbit"></div><svg class="icon art-flower"><use href="#i-flower"/></svg><svg class="icon art-spark"><use href="#i-spark"/></svg><div class="art-book one"><span class="tiny">Uma pausa</span><span class="book-art-title">Novos<br>caminhos</span><svg class="icon"><use href="#i-leaf"/></svg></div><div class="art-book two"><span class="tiny">Uma descoberta</span><span class="book-art-title">Pequenos<br>mundos</span><svg class="icon"><use href="#i-star"/></svg></div><div class="art-book three"><span class="tiny">Um encontro</span><span class="book-art-title">Tempo<br>para<br>sentir</span><svg class="icon"><use href="#i-flower"/></svg></div><div class="art-caption"><svg class="icon"><use href="#i-heart"/></svg>No seu ritmo. Do seu jeito.</div></div>
    </div>
    <div class="workspace">
      <form id="recommend-form" class="panel discovery-panel">
        <div class="panel-heading"><div><p class="step-label">Seu próximo capítulo</p><h2>Como está seu dia?</h2></div><span class="gentle-badge"><svg class="icon" aria-hidden="true"><use href="#i-heart"/></svg>Um espaço para sentir</span></div>
        <p class="panel-subtitle">Escolha um sentimento, escreva sobre seu momento ou faça os dois.</p>
        <span class="field-label" id="mood-label">Hoje eu me sinto… <span class="field-hint">Você pode escolher mais de um.</span></span>
        <div class="mood-grid" id="mood-grid" role="group" aria-labelledby="mood-label"></div>
        <label class="field-label" for="input-texto">Conte um pouco mais <span class="field-hint">Opcional se você escolher um sentimento.</span></label>
        <div class="text-input-wrap"><textarea class="prompt-input" id="input-texto" maxlength="1000" placeholder="Hoje estou pensando em muitas coisas… Quero uma história que me ajude a desacelerar." aria-describedby="char-counter"></textarea><span class="char-counter" id="char-counter">0 / 1000</span></div>
        <div class="example-row"><span class="example-label">Precisa de uma ideia?</span><button class="example-button" type="button" data-example="0">Uma pausa para mim</button><button class="example-button" type="button" data-example="1">Um novo começo</button><button class="example-button" type="button" data-example="2">Uma boa aventura</button></div>
        <details id="preferences" open><summary class="preference-toggle"><svg class="icon" aria-hidden="true"><use href="#i-tune"/></svg>Personalize sua descoberta<svg class="icon chevron" aria-hidden="true"><use href="#i-chevron"/></svg></summary><div class="preferences-grid">
          <div><label for="genre-filter">Gênero literário</label><div class="select-wrap"><select id="genre-filter"><option value="">Todos os gêneros</option></select></div></div>
          <div><label for="pages-filter">Tamanho da leitura</label><div class="select-wrap"><select id="pages-filter"><option value="">Sem preferência</option><option value="200">Até 200 páginas</option><option value="400">Até 400 páginas</option></select></div></div>
          <div><label for="count-filter">Sugestões</label><div class="select-wrap"><select id="count-filter"><option value="4">4 livros</option><option value="6" selected>6 livros</option><option value="8">8 livros</option><option value="10">10 livros</option></select></div></div>
        </div><div class="variety-row"><span>Na seleção, prefiro</span><div class="segmented" role="group" aria-label="Variedade das recomendações"><button type="button" class="segment-button" data-variety="conexao" aria-pressed="false">Mais conexão</button><button type="button" class="segment-button" data-variety="equilibrada" aria-pressed="true">Equilíbrio</button><button type="button" class="segment-button" data-variety="variedade" aria-pressed="false">Mais variedade</button></div></div></details>
        <div class="form-footer"><button type="submit" class="primary-button" id="btn-recomendar"><svg class="icon" aria-hidden="true"><use href="#i-spark"/></svg><span>Encontrar minha leitura</span><svg class="icon" aria-hidden="true"><use href="#i-arrow"/></svg></button><button type="button" class="quiet-button" id="clear-button">Limpar</button></div>
        <p class="shortcut-hint">Dica: Ctrl + Enter também encontra sua próxima leitura.</p>
        <div class="loading-note" id="loading-note" role="status" hidden><span class="loading-dot" aria-hidden="true"></span>Encontrando histórias que conversam com você…</div><div class="status-message" id="form-message" role="alert" hidden></div>
      </form>
      <aside class="side-column" aria-label="Mais formas de descobrir">
        <div class="panel side-panel"><div class="side-icon"><svg class="icon" aria-hidden="true"><use href="#i-spark"/></svg></div><h3>Deixe a curiosidade<br>escolher por você.</h3><p>Às vezes, uma história diferente é tudo o que a gente precisa para abrir novos caminhos.</p><button class="surprise-button" id="surprise-button"><svg class="icon" aria-hidden="true"><use href="#i-compass"/></svg>Me surpreenda</button></div>
        <div class="shelf-note"><div class="shelf-note-top"><svg class="icon" aria-hidden="true"><use href="#i-bookmark"/></svg><span class="shelf-number" id="side-shelf-count">0</span></div><h3>Sua próxima leitura pode esperar.</h3><p>Salve os livros que despertarem sua curiosidade e volte quando quiser.</p><button class="inline-button" data-view="shelf">Abrir minha estante<svg class="icon" aria-hidden="true"><use href="#i-arrow"/></svg></button></div>
        <p class="reading-note"><svg class="icon" aria-hidden="true"><use href="#i-leaf"/></svg>Não precisa terminar um livro de uma vez. Um capítulo já é um começo.</p>
      </aside>
    </div>
    <section class="results-section" id="results-section" aria-labelledby="results-title" hidden>
      <div class="section-head"><div><p class="step-label">Histórias para o seu momento</p><h2 id="results-title" tabindex="-1">Seu próximo encontro</h2><p class="section-description" id="results-description"></p></div><span class="emotion-note" id="emotion-note" hidden></span></div>
      <div class="explanation-panel" id="analysis-panel" hidden></div>
      <div class="result-tools"><div class="search-wrap"><svg class="icon" aria-hidden="true"><use href="#i-search"/></svg><label class="sr-only" for="result-search">Buscar nas sugestões</label><input class="search-input" id="result-search" placeholder="Buscar nas sugestões…" type="search"></div><div class="select-wrap sort-wrap"><label class="sr-only" for="result-sort">Ordenar sugestões</label><select id="result-sort"><option value="recommended">Seleção recomendada</option><option value="affinity">Maior afinidade</option><option value="rating">Melhor avaliação</option><option value="pages">Menos páginas</option><option value="title">Título A–Z</option></select></div><div class="view-controls" role="group" aria-label="Visualização das sugestões"><button data-layout="grid" aria-label="Visualizar em cartões" aria-pressed="true"><svg class="icon" aria-hidden="true"><use href="#i-grid"/></svg></button><button data-layout="list" aria-label="Visualizar em lista" aria-pressed="false"><svg class="icon" aria-hidden="true"><use href="#i-list"/></svg></button></div></div>
      <div class="book-grid" id="results-grid"></div><p class="results-summary" id="results-summary" aria-live="polite"></p>
    </section>
  </section>
  <section class="secondary-view" id="shelf-view" aria-labelledby="shelf-title" hidden><div class="section-head"><div><p class="step-label">Para ler no seu tempo</p><h2 id="shelf-title" tabindex="-1">Minha estante</h2><p class="section-description" id="shelf-description">Um lugar para as histórias que você quer conhecer.</p></div><div class="action-group"><button class="secondary-button" id="export-button"><svg class="icon" aria-hidden="true"><use href="#i-download"/></svg>Exportar lista</button><button class="secondary-button" data-view="discover">Descobrir livros<svg class="icon" aria-hidden="true"><use href="#i-arrow"/></svg></button></div></div><p class="storage-note">Seus livros salvos ficam neste navegador. Exporte a lista para guardar uma cópia.</p><div class="result-tools"><div class="search-wrap"><svg class="icon" aria-hidden="true"><use href="#i-search"/></svg><label class="sr-only" for="shelf-search">Buscar na estante</label><input class="search-input" id="shelf-search" placeholder="Buscar na minha estante…" type="search"></div></div><div class="book-grid" id="shelf-grid"></div></section>
  <section class="secondary-view" id="history-view" aria-labelledby="history-title" hidden><div class="section-head"><div><p class="step-label">Seus momentos de leitura</p><h2 id="history-title" tabindex="-1">Histórico de descobertas</h2><p class="section-description">Volte a uma busca e encontre um novo capítulo.</p></div><button class="secondary-button" id="clear-history-button"><svg class="icon" aria-hidden="true"><use href="#i-trash"/></svg>Limpar histórico</button></div><p class="storage-note">As últimas 12 buscas ficam salvas apenas neste navegador.</p><div id="history-list" class="history-list"></div><details class="history-feedback"><summary>Avaliações recebidas pelo recomendador</summary><p class="storage-note">Resumo de todos os votos registrados na aplicação, incluindo os de outros leitores.</p><div id="feedback-history"></div></details></section>
  <footer class="app-footer"><div><p class="footer-brand">emotional book.</p><p>Incentivando a leitura através de encontros e descobertas.</p></div><div class="footer-right"><p>Projeto acadêmico · FATEC Cotia</p><p>Ciência de Dados · Processamento de Linguagem Natural</p><p><span><svg class="icon" aria-hidden="true"><use href="#i-heart"/></svg>Feito para quem sente e lê.</span></p></div></footer>
</main>
<dialog id="book-dialog" aria-labelledby="dialog-book-title"><button class="icon-button modal-close" data-close="book-dialog" aria-label="Fechar detalhes"><svg class="icon" aria-hidden="true"><use href="#i-close"/></svg></button><div class="modal-body" id="book-dialog-content"></div></dialog>
<dialog id="help-dialog" aria-labelledby="help-title"><button class="icon-button modal-close" data-close="help-dialog" aria-label="Fechar ajuda"><svg class="icon" aria-hidden="true"><use href="#i-close"/></svg></button><div class="modal-body"><div class="modal-header"><p class="step-label">Emotional Book</p><h2 id="help-title">Uma história começa<br>com o seu momento.</h2></div><ol class="explain-list"><li><span class="number-dot">1</span><div><b>Conte como você se sente.</b><p>Escolha um ou mais sentimentos, escreva o que busca e ajuste suas preferências.</p></div></li><li><span class="number-dot">2</span><div><b>Encontre sua próxima leitura.</b><p>Explore a seleção, leia as sinopses e veja quais histórias conversam com você.</p></div></li><li><span class="number-dot">3</span><div><b>Guarde o que despertar curiosidade.</b><p>Salve seus favoritos na estante, exporte sua lista e avalie as sugestões.</p></div></li></ol><p class="safety-text">A afinidade indica a proximidade entre a busca e a descrição do livro. As ilustrações dos cartões são criadas pela interface e não são capas oficiais.</p></div></dialog>
<dialog id="confirm-dialog" aria-labelledby="confirm-title"><div class="modal-body"><h2 id="confirm-title">Limpar seu histórico?</h2><p class="modal-subtitle">Isso remove as buscas salvas neste navegador.</p><div class="modal-actions"><button class="primary-button" id="confirm-clear-history">Limpar histórico</button><button class="secondary-button" data-close="confirm-dialog">Cancelar</button></div></div></dialog>
<div class="toast" id="toast" role="status" aria-live="polite"></div>
<script>
'use strict';
(() => {
  const DEMO = window.EMOTIONAL_BOOK_DEMO || null;
  const PREFIX = DEMO ? 'emotional-book:preview:v2:' : 'emotional-book:v2:';
  const $ = (id) => document.getElementById(id);
  const esc = (value) => String(value ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const icon = (name, extra = '') => `<svg class="icon ${extra}" aria-hidden="true"><use href="#i-${name}"/></svg>`;
  const number = (value) => {
    if (value === null || value === undefined || String(value).trim() === '') return null;
    const result = Number(String(value).replace(',', '.'));
    return Number.isFinite(result) ? result : null;
  };
  const text = value => ['nan', 'none', 'null', 'N/A'].includes(String(value)) ? '' : String(value ?? '');
  const norm = value => text(value).normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase().trim();
  const bookKey = book => norm(book['Título do Livro']);
  const titleOf = book => text(book['Título do Livro']) || 'Título não informado';
  const cleanBook = (book) => {
    const result = {};
    ['Título do Livro','Gênero','Sentimento I','Emoção I','Emoção II','Sinopse','Público-Alvo','Tamanho do Livro','explicacao'].forEach(k => result[k] = text(book[k]));
    ['Páginas','rating','_score_sem','_score_final'].forEach(k => result[k] = number(book[k]));
    return result;
  };
  const readStore = (key, fallback) => {
    try { const value = JSON.parse(localStorage.getItem(PREFIX + key)); return value ?? fallback; }
    catch (_) { return fallback; }
  };
  let storageWarningShown = false;
  const writeStore = (key, value) => {
    try { localStorage.setItem(PREFIX + key, JSON.stringify(value)); return true; }
    catch (_) {
      if (!storageWarningShown) { storageWarningShown = true; toast('Seu navegador não permitiu salvar. Exporte sua estante para guardar uma cópia.'); }
      return false;
    }
  };
  const MOODS = [
    {id:'feliz', label:'Alegre', word:'feliz', icon:'sun', color:'var(--yellow)'},
    {id:'reflexivo', label:'Reflexivo', word:'reflexivo', icon:'moon', color:'var(--lilac)'},
    {id:'ansioso', label:'Ansioso', word:'ansioso', icon:'wave', color:'var(--sage)'},
    {id:'triste', label:'Triste', word:'triste', icon:'cloud', color:'var(--pink)'},
    {id:'inspirado', label:'Inspirado', word:'inspirado', icon:'spark', color:'var(--yellow)'},
    {id:'romântico', label:'Romântico', word:'romântico', icon:'heart', color:'var(--pink)'},
    {id:'nostálgico', label:'Nostálgico', word:'nostálgico', icon:'clock', color:'var(--lilac)'},
    {id:'entediado', label:'Entediado', word:'entediado', icon:'compass', color:'var(--sage)'}
  ];
  const EXAMPLES = [
    {text:'Hoje estou ansioso e quero uma história tranquila para desacelerar e encontrar um pouco de paz.', moods:['ansioso']},
    {text:'Estou inspirado para começar uma nova fase. Quero uma leitura sobre coragem, recomeços e possibilidades.', moods:['inspirado']},
    {text:'Estou entediado com a rotina e quero me perder em uma aventura cheia de descobertas.', moods:['entediado']},
    {text:'Estou reflexivo e quero uma história que me faça pensar sobre escolhas e o sentido da vida.', moods:['reflexivo']},
    {text:'Estou romântico e quero conhecer uma história de afeto, encontros e conexão.', moods:['romântico']},
    {text:'Hoje estou nostálgico. Quero uma leitura sobre memórias, lugares e pessoas que deixam saudade.', moods:['nostálgico']}
  ];
  const coverPalette = [
    {bg:'#eceddf', book:'#cdd4b4', ink:'#3b5346'},
    {bg:'#f0e9e5', book:'#e4c7b5', ink:'#6d4a3c'},
    {bg:'#eeebf4', book:'#d7cee7', ink:'#51485e'},
    {bg:'#e7ede7', book:'#bfcec1', ink:'#375348'},
    {bg:'#f5efdf', book:'#e4d39d', ink:'#62533b'},
    {bg:'#edeaf0', book:'#ccc4d8', ink:'#4d445d'}
  ];
  const rawFavorites = readStore('favorites', []);
  const rawHistory = readStore('history', []);
  const state = {
    view:'discover', selected:new Set(), variety:'equilibrada', layout:readStore('layout', 'grid') === 'list' ? 'list' : 'grid',
    favorites:Array.isArray(rawFavorites) ? rawFavorites.filter(v => v && v.livro && typeof v.livro === 'object' && bookKey(v.livro)).map(v => ({livro:cleanBook(v.livro),query:text(v.query),savedAt:text(v.savedAt)})).slice(0, 300) : [],
    history:Array.isArray(rawHistory) ? rawHistory.filter(v => v && typeof v.query === 'string' && Array.isArray(v.moods)).slice(0,12) : [],
    books:[], resultQuery:'', resultData:null, busy:false, votes:new Map(), pendingVotes:new Set(), genres:[], dialogContext:null
  };
  let toastTimer;
  function toast(message) {
    $('toast').textContent = message;
    $('toast').classList.add('visible');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => $('toast').classList.remove('visible'), 3500);
  }
  function showMessage(message) { $('form-message').textContent = message; $('form-message').hidden = !message; }
  function renderMoods() {
    $('mood-grid').innerHTML = MOODS.map(m => `<button type="button" class="mood-button" data-mood="${m.id}" aria-pressed="${state.selected.has(m.id)}" style="--mood-color:${m.color}" ${state.busy ? 'disabled' : ''}>${icon(m.icon)}${m.label}</button>`).join('');
  }
  function setVariety(value) {
    state.variety = ['conexao','equilibrada','variedade'].includes(value) ? value : 'equilibrada';
    document.querySelectorAll('[data-variety]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.variety === state.variety)));
  }
  function updateCounter() { $('char-counter').textContent = `${$('input-texto').value.length} / 1000`; }
  function formSnapshot() {
    return {inputText:$('input-texto').value,moods:[...state.selected],genre:$('genre-filter').value,pages:$('pages-filter').value,top_k:Number($('count-filter').value),variety:state.variety};
  }
  function saveDraft() { writeStore('draft', formSnapshot()); }
  function restoreForm(snapshot = {}) {
    $('input-texto').value = text(snapshot.inputText).slice(0,1000);
    state.selected = new Set(Array.isArray(snapshot.moods) ? snapshot.moods.filter(m => MOODS.some(x => x.id === m)) : []);
    $('genre-filter').value = state.genres.includes(snapshot.genre) ? snapshot.genre : '';
    $('pages-filter').value = ['200','400'].includes(String(snapshot.pages)) ? String(snapshot.pages) : '';
    $('count-filter').value = [4,6,8,10].includes(Number(snapshot.top_k)) ? String(snapshot.top_k) : '6';
    setVariety(snapshot.variety); renderMoods(); updateCounter(); showMessage('');
  }
  function fillExample(index) {
    if (state.busy) return;
    const example = EXAMPLES[index % EXAMPLES.length];
    $('input-texto').value = example.text;
    state.selected = new Set(example.moods); renderMoods(); updateCounter(); saveDraft(); showMessage('');
    showView('discover', false); $('input-texto').focus();
  }
  function updateShelfCount() {
    const count = state.favorites.length;
    $('nav-shelf-count').textContent = count;
    $('side-shelf-count').textContent = count;
    $('export-button').disabled = count === 0;
    $('shelf-description').textContent = count ? `${count} ${count === 1 ? 'história salva para conhecer' : 'histórias salvas para conhecer'} no seu tempo.` : 'Um lugar para as histórias que você quer conhecer.';
  }
  function showView(view, focus = true) {
    state.view = ['discover','shelf','history'].includes(view) ? view : 'discover';
    ['discover','shelf','history'].forEach(name => $(name + '-view').hidden = name !== state.view);
    document.querySelectorAll('.nav-button[data-view]').forEach(button => {
      const active = button.dataset.view === state.view;
      button.classList.toggle('active', active);
      if (active) button.setAttribute('aria-current', 'page'); else button.removeAttribute('aria-current');
    });
    if (state.view === 'shelf') renderShelf();
    if (state.view === 'history') { renderHistory(); loadFeedbackHistory(); }
    try { history.replaceState(null, '', '#' + ({discover:'descobrir',shelf:'estante',history:'historico'}[state.view])); } catch (_) {}
    if (focus) {
      const heading = $(state.view === 'discover' ? 'input-texto' : state.view + '-title');
      heading.focus({preventScroll:true});
      heading.scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth',block:'center'});
    }
  }
  function toggleTheme() { setTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'); }
  function setTheme(theme, persist = true) {
    const chosen = theme === 'dark' ? 'dark' : 'light';
    document.documentElement.dataset.theme = chosen;
    const label = chosen === 'dark' ? 'Ativar modo claro' : 'Ativar modo escuro';
    $('theme-button').innerHTML = icon(chosen === 'dark' ? 'sun' : 'moon');
    $('theme-button').setAttribute('aria-label', label); $('theme-button').title = label;
    document.querySelector('meta[name="theme-color"]').content = chosen === 'dark' ? '#18221d' : '#f7f8f3';
    if (persist) writeStore('theme', chosen);
  }
  async function api(path, body) {
    if (DEMO) return demoApi(path, body);
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 120000);
    try {
      const response = await fetch(path, {method:body ? 'POST':'GET',headers:body ? {'Content-Type':'application/json'}:{},body:body ? JSON.stringify(body):undefined,signal:controller.signal});
      let data;
      try { data = await response.json(); } catch (_) { throw new Error('O servidor não retornou uma resposta válida. Tente novamente.'); }
      if (!response.ok || data.erro) throw new Error(data.erro || 'Não foi possível concluir a solicitação. Tente novamente.');
      return data;
    } catch (error) {
      if (error.name === 'AbortError') throw new Error('A busca demorou mais que o esperado. Tente novamente em instantes.');
      if (error instanceof TypeError) throw new Error('Não consegui conectar ao recomendador. Verifique se a aplicação está em execução.');
      throw error;
    } finally { clearTimeout(timer); }
  }
  async function loadOptions() {
    try {
      const data = await api('/opcoes');
      state.genres = Array.isArray(data.generos) ? data.generos.filter(g => typeof g === 'string' && g.trim()) : [];
      $('genre-filter').innerHTML = '<option value="">Todos os gêneros</option>' + state.genres.map(g => `<option value="${esc(g)}">${esc(g)}</option>`).join('');
      const draft = readStore('draft', {}); restoreForm(draft && typeof draft === 'object' ? draft : {});
    } catch (_) {
      $('genre-filter').disabled = true;
      $('genre-filter').title = 'Os gêneros não puderam ser carregados. Você ainda pode buscar sem esse filtro.';
      toast('Não consegui carregar os gêneros do catálogo. Tente recarregar a página.');
    }
  }
  function setBusy(busy) {
    state.busy = busy;
    $('recommend-form').setAttribute('aria-busy', String(busy));
    $('recommend-form').querySelectorAll('button,input,textarea,select').forEach(el => {
      if (busy) { el.dataset.wasDisabled = String(el.disabled); el.disabled = true; }
      else { el.disabled = el.dataset.wasDisabled === 'true'; delete el.dataset.wasDisabled; }
    });
    $('surprise-button').disabled = busy;
    $('loading-note').hidden = !busy;
    $('btn-recomendar').innerHTML = busy ? '<span class="loading-dot" aria-hidden="true"></span><span>À procura da sua história…</span>' : `${icon('spark')}<span>Encontrar minha leitura</span>${icon('arrow')}`;
  }
  async function recommend(event) {
    if (event) event.preventDefault();
    if (state.busy) return;
    const snapshot = formSnapshot();
    const query = snapshot.inputText.trim() || (snapshot.moods.length ? 'Estou me sentindo ' + snapshot.moods.map(m => MOODS.find(x => x.id === m).word).join(' e ') + '.' : '');
    if (!query) { showMessage('Escolha um sentimento ou conte um pouco sobre o que você gostaria de ler.'); $('input-texto').focus(); return; }
    showMessage(''); saveDraft(); setBusy(true);
    try {
      const data = await api('/recomendar', {query,emocoes:snapshot.moods,genero:snapshot.genre || null,paginas_max:snapshot.pages ? Number(snapshot.pages) : null,top_k:snapshot.top_k,variedade:snapshot.variety});
      if (data.crise) { $('results-section').hidden = true; showMessage(data.mensagem || 'Sua mensagem indica um momento difícil. Busque apoio de alguém de confiança.'); return; }
      if (!Array.isArray(data.livros) || data.livros.length === 0) throw new Error('Nenhum livro encontrado com essas preferências. Experimente outro gênero ou tamanho de leitura.');
      state.books = data.livros.filter(b => b && typeof b === 'object').map(cleanBook);
      if (!state.books.length) throw new Error('As sugestões retornadas não puderam ser exibidas. Tente novamente.');
      state.resultQuery = query; state.resultData = data; state.votes.clear();
      $('result-search').value = ''; $('result-sort').value = 'recommended';
      $('results-section').hidden = false;
      $('results-description').textContent = `${state.books.length} ${state.books.length === 1 ? 'livro selecionado' : 'livros selecionados'} para você explorar, sem pressa.`;
      $('emotion-note').hidden = !data.emocao;
      $('emotion-note').innerHTML = `${icon('heart')}<span>Seu momento: ${esc(data.emocao)}</span>`;
      $('analysis-panel').hidden = !text(data.analise_ia).trim();
      if (data.analise_ia) $('analysis-panel').innerHTML = `${icon('spark')}<div><h3>Uma conexão com a sua leitura</h3><p class="explanation-target">${esc(titleOf(state.books[0]))}</p><p>${esc(data.analise_ia)}</p></div>`;
      renderResults();
      const entry = {...snapshot,query,label:text(data.emocao),bookCount:state.books.length,createdAt:new Date().toISOString(),id:Date.now().toString(36) + Math.random().toString(36).slice(2,8)};
      state.history = [entry,...state.history].slice(0,12); writeStore('history',state.history);
      if (state.view === 'discover') {
        $('results-title').focus({preventScroll:true});
        $('results-section').scrollIntoView({behavior:matchMedia('(prefers-reduced-motion: reduce)').matches ? 'auto' : 'smooth',block:'start'});
      } else toast('Sua seleção de livros está pronta na aba Descobrir.');
    } catch (error) { showMessage(error.message || 'Não foi possível encontrar livros agora. Tente novamente.'); }
    finally { setBusy(false); }
  }
  function coverFor(book) {
    let hash = 0;
    for (const char of titleOf(book)) hash = (hash * 31 + char.charCodeAt(0)) >>> 0;
    return coverPalette[hash % coverPalette.length];
  }
  function affinity(book) { return book._score_sem === null ? '' : `${Math.round(Math.max(0,Math.min(1,book._score_sem))*100)}% de afinidade`; }
  function isSaved(book) { return state.favorites.some(item => bookKey(item.livro) === bookKey(book)); }
  function bookCard(book, index, source) {
    const color = coverFor(book); const saved = isSaved(book); const score = affinity(book);
    const pages = book['Páginas'] > 0 ? `${book['Páginas'].toLocaleString('pt-BR')} páginas` : '';
    const tags = [text(book['Sentimento I']),pages].filter(Boolean);
    return `<article class="panel book-card"><div class="cover-area" style="--cover-bg:${color.bg};--cover:${color.book};--cover-ink:${color.ink}" aria-hidden="true"><div class="book-object"><span class="cover-genre">${esc(book['Gênero'] || 'Uma história')}</span><span class="cover-title">${esc(titleOf(book))}</span>${icon('flower')}<span class="cover-foot">Emotional Book</span></div>${source === 'results' && index === 0 ? '<span class="top-pick">PRIMEIRA SUGESTÃO</span>' : ''}</div><button class="save-button" data-action="save" data-source="${source}" data-index="${index}" aria-pressed="${saved}" aria-label="${saved ? 'Remover da estante: ' : 'Salvar na estante: '}${esc(titleOf(book))}" title="${saved ? 'Remover da estante':'Salvar na estante'}">${icon('bookmark')}</button><div class="book-content"><div class="book-meta-line"><span class="book-genre">${esc(book['Gênero'] || 'Gênero não informado')}</span>${book.rating !== null ? `<span class="rating" aria-label="Avaliação ${esc(book.rating)}">${icon('star')}${book.rating.toLocaleString('pt-BR',{maximumFractionDigits:1})}</span>` : ''}</div><h3>${esc(titleOf(book))}</h3><p class="book-synopsis">${esc(book['Sinopse'] || 'Abra os detalhes para conhecer esta leitura.')}</p><div class="book-tags">${tags.map(t => `<span class="book-tag">${esc(t)}</span>`).join('')}</div><div class="card-bottom"><span class="affinity" title="Proximidade entre a busca e a descrição do livro">${esc(score || 'Uma leitura para explorar')}</span><button class="detail-button" data-action="details" data-source="${source}" data-index="${index}">Ver detalhes${icon('arrow')}</button></div></div></article>`;
  }
  function emptyState(title, description, button = false) {
    return `<div class="empty-state" style="grid-column:1/-1">${icon('book')}<h3>${esc(title)}</h3><p>${esc(description)}</p>${button ? `<button class="primary-button" data-view="discover">Descobrir minha próxima leitura${icon('arrow')}</button>` : ''}</div>`;
  }
  function renderResults() {
    const search = norm($('result-search').value);
    let entries = state.books.map((book,index) => ({book,index})).filter(({book}) => norm([book['Título do Livro'],book['Gênero'],book['Sentimento I']].join(' ')).includes(search));
    const sort = $('result-sort').value;
    if (sort === 'title') entries.sort((a,b) => titleOf(a.book).localeCompare(titleOf(b.book),'pt-BR'));
    if (sort === 'pages') entries.sort((a,b) => (a.book['Páginas'] > 0 ? a.book['Páginas']:Infinity) - (b.book['Páginas'] > 0 ? b.book['Páginas']:Infinity));
    if (sort === 'rating') entries.sort((a,b) => (b.book.rating ?? -Infinity) - (a.book.rating ?? -Infinity));
    if (sort === 'affinity') entries.sort((a,b) => (b.book._score_sem ?? -Infinity) - (a.book._score_sem ?? -Infinity));
    $('results-grid').classList.toggle('list-view',state.layout === 'list');
    $('results-grid').innerHTML = entries.map(({book,index}) => bookCard(book,index,'results')).join('') || emptyState('Nenhuma história por aqui.','Experimente buscar por outro título, gênero ou sentimento.');
    $('results-summary').textContent = `${entries.length} de ${state.books.length} sugestões exibidas · Ilustrações de capa criadas para esta interface.`;
    document.querySelectorAll('[data-layout]').forEach(el => el.setAttribute('aria-pressed', String(el.dataset.layout === state.layout)));
  }
  function renderShelf() {
    const search = norm($('shelf-search').value);
    const entries = state.favorites.map((item,index) => ({book:item.livro,index})).filter(({book}) => norm([book['Título do Livro'],book['Gênero'],book['Sentimento I']].join(' ')).includes(search));
    $('shelf-grid').innerHTML = entries.map(({book,index}) => bookCard(book,index,'shelf')).join('') || (state.favorites.length ? emptyState('Nenhum favorito com esse nome.','Busque por outro título, gênero ou sentimento.') : emptyState('Sua estante está esperando uma história.','Encontre uma leitura e toque no marcador para guardá-la aqui.',true));
    updateShelfCount();
  }
  function contextFor(source, index) {
    const position = Number(index);
    if (!Number.isInteger(position) || position < 0) return null;
    if (source === 'results' && state.books[position]) return {book:state.books[position],query:state.resultQuery,source,index:position};
    if (source === 'shelf' && state.favorites[position]) return {book:state.favorites[position].livro,query:state.favorites[position].query,source,index:position};
    return null;
  }
  function toggleFavorite(context) {
    const key = bookKey(context.book); const index = state.favorites.findIndex(item => bookKey(item.livro) === key);
    if (index >= 0) { state.favorites.splice(index,1); toast('Livro removido da sua estante.'); }
    else {
      if (state.favorites.length >= 300) { toast('Sua estante tem 300 livros. Exporte a lista ou remova um para salvar outro.'); return; }
      state.favorites.unshift({livro:context.book,query:context.query,savedAt:new Date().toISOString()}); toast('Livro salvo. Sua próxima leitura pode esperar.');
    }
    writeStore('favorites',state.favorites); updateShelfCount(); renderResults(); renderShelf();
    if ($('book-dialog').open) { renderBookDialog(); $('dialog-save-button').focus({preventScroll:true}); }
    else {
      const next = document.querySelector(`#${context.source === 'results' ? 'results' : 'shelf'}-grid [data-action="save"][data-index="${context.index}"]`);
      if (next) next.focus({preventScroll:true}); else $('shelf-title').focus({preventScroll:true});
    }
  }
  function voteKey(context) { return bookKey(context.book) + '|' + context.query; }
  function renderBookDialog() {
    const context = state.dialogContext; if (!context) return;
    const book = context.book; const saved = isSaved(book); const key = voteKey(context); const hasVote = state.votes.has(key); const pending = state.pendingVotes.has(key);
    const tags = [book['Gênero'],book['Páginas'] > 0 ? `${book['Páginas'].toLocaleString('pt-BR')} páginas` : '',book['Público-Alvo'],book.rating !== null ? `Avaliação: ${book.rating.toLocaleString('pt-BR')}`:'',affinity(book)].filter(Boolean);
    const emotions = [book['Sentimento I'],book['Emoção I'],book['Emoção II']].filter(Boolean);
    $('book-dialog-content').innerHTML = `<div class="modal-header"><p class="step-label">Uma história para conhecer</p><h2 id="dialog-book-title">${esc(titleOf(book))}</h2><p class="modal-subtitle">${esc(book['Gênero'] || 'Sua próxima descoberta')}</p></div><div class="modal-tags">${tags.map(t => `<span>${esc(t)}</span>`).join('')}</div><div class="modal-section"><h3>Sobre esta leitura</h3><p>${esc(book['Sinopse'] || 'Este livro ainda não tem uma sinopse no catálogo.')}</p></div>${emotions.length ? `<div class="modal-section"><h3>Sentimentos presentes na história</h3><p>${esc([...new Set(emotions)].join(' · '))}</p></div>` : ''}<div class="modal-actions"><button class="primary-button" id="dialog-save-button" data-action="dialog-save" aria-pressed="${saved}">${icon(saved ? 'check':'bookmark')}${saved ? 'Salvo na minha estante':'Salvar na minha estante'}</button><button class="secondary-button" data-action="copy">${icon('copy')}Copiar indicação</button></div><div class="modal-section"><h3>Essa indicação fez sentido para você?</h3><div class="feedback-buttons"><button class="feedback-button" data-action="vote" data-useful="true" aria-pressed="${hasVote && state.votes.get(key) === true}" ${hasVote || pending ? 'disabled' : ''}>${icon('thumb')}Fez sentido</button><button class="feedback-button" data-action="vote" data-useful="false" aria-pressed="${hasVote && state.votes.get(key) === false}" ${hasVote || pending ? 'disabled' : ''}>${icon('thumb','thumb-down')}Outra leitura, talvez</button></div><p class="feedback-feedback">${pending ? 'Enviando sua avaliação…' : hasVote ? 'Obrigado! Sua avaliação foi registrada.' : 'Sua avaliação ajuda a melhorar as próximas sugestões.'}</p></div>`;
  }
  function openBook(context) { state.dialogContext = context; renderBookDialog(); $('book-dialog').showModal(); }
  async function sendVote(useful) {
    const context = state.dialogContext; if (!context) return;
    const key = voteKey(context); if (state.votes.has(key) || state.pendingVotes.has(key)) return;
    state.pendingVotes.add(key); renderBookDialog();
    try { await api('/feedback',{titulo:titleOf(context.book),util:useful,query:context.query}); state.votes.set(key,useful); toast('Obrigado por avaliar esta indicação.'); }
    catch (error) { toast(error.message || 'Não foi possível registrar a avaliação. Tente novamente.'); }
    finally {
      state.pendingVotes.delete(key);
      if (state.dialogContext && voteKey(state.dialogContext) === key) { renderBookDialog(); const close = $('book-dialog').querySelector('[data-close]'); close.focus({preventScroll:true}); }
    }
  }
  function renderHistory() {
    $('clear-history-button').disabled = state.history.length === 0;
    $('history-list').innerHTML = state.history.map((item,index) => {
      const date = new Date(item.createdAt); const dateLabel = Number.isNaN(date.getTime()) ? 'Busca salva' : date.toLocaleString('pt-BR',{dateStyle:'short',timeStyle:'short'});
      const filters = [item.genre,item.pages ? `Até ${item.pages} páginas`:'',`${Number(item.bookCount) || 0} sugestões`].filter(Boolean);
      return `<article class="panel history-item"><div><p class="history-query">${esc(item.query)}</p><p class="history-meta"><span>${esc(dateLabel)}</span>${filters.map(f => `<span>${esc(f)}</span>`).join('')}</p></div><div class="history-actions"><button class="secondary-button" data-action="restore-history" data-index="${index}">Refazer busca${icon('arrow')}</button><button class="icon-button" data-action="remove-history" data-index="${index}" aria-label="Excluir esta busca">${icon('trash')}</button></div></article>`;
    }).join('') || emptyState('Cada momento tem seu capítulo.','Suas buscas vão aparecer aqui depois da primeira descoberta.',true);
  }
  async function loadFeedbackHistory() {
    $('feedback-history').innerHTML = '<p class="storage-note">Carregando avaliações…</p>';
    try {
      const data = await api('/historico');
      const items = Array.isArray(data.historico) ? data.historico : [];
      $('feedback-history').innerHTML = items.map(item => `<div class="feedback-history-row"><span>${esc(item.titulo)}</span><span class="feedback-score">Saldo: ${Number(item.score)>0?'+':''}${number(item.score) ?? 0}</span></div>`).join('') || '<p class="storage-note">Nenhuma avaliação registrada ainda.</p>';
    } catch (_) { $('feedback-history').innerHTML = '<p class="storage-note">As avaliações não puderam ser carregadas agora.</p>'; }
  }
  function exportShelf() {
    if (!state.favorites.length) return;
    const lines = ['EMOTIONAL BOOK — MINHA ESTANTE',new Date().toLocaleDateString('pt-BR'),''];
    state.favorites.forEach((item,index) => {
      const book = item.livro; lines.push(`${index+1}. ${titleOf(book)}`,`Gênero: ${book['Gênero'] || 'Não informado'}`,`Páginas: ${book['Páginas'] || 'Não informado'}`);
      if (book['Sinopse']) lines.push(`Sinopse: ${book['Sinopse']}`);
      lines.push('');
    });
    const url = URL.createObjectURL(new Blob(['\uFEFF' + lines.join('\n')],{type:'text/plain;charset=utf-8'}));
    const link = document.createElement('a'); link.href = url; link.download = 'minha_estante_emotional_book.txt'; document.body.appendChild(link); link.click(); link.remove(); setTimeout(() => URL.revokeObjectURL(url),1000); toast('Sua lista de leitura foi exportada.');
  }
  async function copyBook() {
    const context = state.dialogContext; if (!context) return;
    const book = context.book;
    const content = [titleOf(book),book['Gênero'],book['Páginas'] > 0 ? `${book['Páginas']} páginas`:'',book['Sinopse'],'Indicação do Emotional Book'].filter(Boolean).join('\n\n');
    try { await navigator.clipboard.writeText(content); toast('Indicação copiada.'); }
    catch (_) {
      const textarea = document.createElement('textarea'); textarea.value = content; textarea.className = 'sr-only'; $('book-dialog-content').appendChild(textarea); textarea.select();
      try { toast(document.execCommand('copy') ? 'Indicação copiada.':'Não foi possível copiar. Selecione o texto da sinopse para copiar.'); }
      catch (_) { toast('Não foi possível copiar automaticamente.'); }
      textarea.remove();
    }
  }
  async function demoApi(path, body) {
    await new Promise(resolve => setTimeout(resolve,200));
    if (path === '/opcoes') return {generos:[...new Set(DEMO.books.map(b => b['Gênero']))].sort((a,b)=>a.localeCompare(b,'pt-BR')),livros:DEMO.books.length};
    if (path === '/historico') return {historico:DEMO.feedback || []};
    if (path === '/feedback') {
      DEMO.feedback = DEMO.feedback || [];
      const previous = DEMO.feedback.find(f => f.titulo === body.titulo);
      if (previous) previous.score += body.util ? 1:-1; else DEMO.feedback.push({titulo:body.titulo,score:body.util ? 1:-1});
      return {ok:true};
    }
    if (path === '/recomendar') {
      const crisis = ['me matar','quero morrer','não quero mais viver','nao quero mais viver','me suicidar','tirar minha vida','me machucar','me ferir','acabar com tudo','suicídio','suicidio','não vale a pena viver','pensar em morrer','pensamentos de morte'];
      if (crisis.some(word => body.query.toLowerCase().includes(word))) return {crise:true,mensagem:'Percebi que sua mensagem pode indicar um momento muito difícil. Recomendações de livros podem esperar — o mais importante agora é você. Entre em contato com o CVV: ligue 188 ou acesse cvv.org.br. Você não precisa passar por isso sozinho.'};
      let books = DEMO.books.filter(b => (!body.genero || b['Gênero'] === body.genero) && (!body.paginas_max || b['Páginas'] <= body.paginas_max));
      if (!books.length) throw new Error('Nenhum livro encontrado com essas preferências. Experimente outro gênero ou tamanho de leitura.');
      books = books.slice(0,body.top_k || 6).map((book,i) => ({...book,_score_sem:.87 - i*.044,_score_final:.82 - i*.035}));
      const emotions = body.emocoes.length ? body.emocoes.join(' + ') : 'reflexivo';
      return {livros:books,emocao:emotions,analise_ia:'Esta é uma explicação de exemplo para a prévia. A história convida a olhar para o próprio ritmo e para as pequenas possibilidades de recomeço. No aplicativo Python, a explicação personalizada aparece quando a análise de IA está disponível.',ia_habilitada:true};
    }
    throw new Error('Operação indisponível nesta prévia.');
  }
  document.addEventListener('click', async event => {
    const view = event.target.closest('[data-view]');
    if (view) { event.preventDefault(); showView(view.dataset.view); return; }
    const mood = event.target.closest('[data-mood]');
    if (mood && !state.busy) { const key = mood.dataset.mood; state.selected.has(key) ? state.selected.delete(key):state.selected.add(key); mood.setAttribute('aria-pressed',String(state.selected.has(key))); saveDraft(); showMessage(''); return; }
    const variety = event.target.closest('[data-variety]'); if (variety && !state.busy) { setVariety(variety.dataset.variety); saveDraft(); return; }
    const example = event.target.closest('[data-example]'); if (example) { fillExample(Number(example.dataset.example)); return; }
    const layout = event.target.closest('[data-layout]'); if (layout) { state.layout = layout.dataset.layout; writeStore('layout',state.layout); renderResults(); return; }
    const close = event.target.closest('[data-close]'); if (close) { $(close.dataset.close).close(); return; }
    const action = event.target.closest('[data-action]'); if (!action || action.disabled) return;
    const name = action.dataset.action;
    if (name === 'save' || name === 'details') { const context = contextFor(action.dataset.source,action.dataset.index); if (context) name === 'save' ? toggleFavorite(context):openBook(context); }
    if (name === 'dialog-save' && state.dialogContext) toggleFavorite(state.dialogContext);
    if (name === 'copy') await copyBook();
    if (name === 'vote') await sendVote(action.dataset.useful === 'true');
    if (name === 'restore-history') { const item = state.history[Number(action.dataset.index)]; if (!item || state.busy) return; restoreForm(item); saveDraft(); showView('discover',false); await recommend(); }
    if (name === 'remove-history') { const index = Number(action.dataset.index); if (!Number.isInteger(index) || !state.history[index]) return; state.history.splice(index,1); writeStore('history',state.history); renderHistory(); toast('Busca removida do histórico.'); }
  });
  $('recommend-form').addEventListener('submit',recommend);
  $('input-texto').addEventListener('input',() => {updateCounter();saveDraft();});
  $('input-texto').addEventListener('keydown',event => {if ((event.ctrlKey || event.metaKey) && event.key === 'Enter' && !event.isComposing) recommend(event);});
  ['genre-filter','pages-filter','count-filter'].forEach(id => $(id).addEventListener('change',saveDraft));
  $('clear-button').addEventListener('click',() => {
    restoreForm();saveDraft();state.books=[];state.resultQuery='';state.resultData=null;state.votes.clear();
    $('results-section').hidden=true;$('result-search').value='';$('result-sort').value='recommended';
    renderResults();$('input-texto').focus();
  });
  $('surprise-button').addEventListener('click',() => fillExample(Math.floor(Math.random()*EXAMPLES.length)));
  $('theme-button').addEventListener('click',toggleTheme);
  $('help-button').addEventListener('click',() => $('help-dialog').showModal());
  $('result-search').addEventListener('input',renderResults);
  $('result-sort').addEventListener('change',renderResults);
  $('shelf-search').addEventListener('input',renderShelf);
  $('export-button').addEventListener('click',exportShelf);
  $('clear-history-button').addEventListener('click',() => {if(state.history.length)$('confirm-dialog').showModal();});
  $('confirm-clear-history').addEventListener('click',() => {state.history=[];writeStore('history',[]);renderHistory();$('confirm-dialog').close();toast('Histórico de buscas removido.');});
  document.querySelectorAll('dialog').forEach(dialog => dialog.addEventListener('click',event => {
    if (event.target !== dialog) return;
    const r = dialog.getBoundingClientRect();
    if (event.clientX < r.left || event.clientX > r.right || event.clientY < r.top || event.clientY > r.bottom) dialog.close();
  }));
  $('demo-banner').hidden = !DEMO;
  setTheme(readStore('theme','light'),false); renderMoods(); updateShelfCount(); renderShelf(); renderHistory();
  restoreForm(readStore('draft',{}) || {});
  showView(({estante:'shelf',historico:'history'}[location.hash.slice(1)]) || 'discover',false);
  loadOptions();
})();

</script>
</body>
</html>
"""

# ============================================================
# 11. ROTAS
# ============================================================
@app.route("/")
def index():
    return HTML

@app.route("/recomendar", methods=["POST"])
def rota_recomendar():
    ip = request.remote_addr or "local"
    if not rate_recomendar(ip):
        return jsonify({"erro": "Muitas requisições. Aguarde um minuto antes de tentar novamente."}), 429
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"erro": "Envie uma solicitação JSON válida."}), 400
    query = body.get("query", "")
    if not isinstance(query, str):
        return jsonify({"erro": "A descrição deve ser um texto."}), 400
    query = query.strip()
    if not query:
        return jsonify({"erro": "Escolha um sentimento ou descreva o que gostaria de ler."}), 400
    if len(query) > 1000:
        return jsonify({"erro": "Use até 1000 caracteres na sua descrição."}), 400
    if detectar_crise(query):
        return jsonify({"crise": True, "mensagem": CRISIS_RESPOSTA})

    top_k = body.get("top_k", 10)
    if type(top_k) is not int or top_k not in (4, 6, 8, 10):
        return jsonify({"erro": "Escolha 4, 6, 8 ou 10 sugestões."}), 400
    paginas_max = body.get("paginas_max")
    if paginas_max is not None and (type(paginas_max) is not int or paginas_max not in (200, 400)):
        return jsonify({"erro": "Escolha até 200, até 400 páginas ou sem preferência."}), 400
    genero = body.get("genero")
    if genero is not None and not isinstance(genero, str):
        return jsonify({"erro": "O gênero escolhido deve ser um texto."}), 400
    genero = genero.strip() if genero else None
    variedade = body.get("variedade", "equilibrada")
    if not isinstance(variedade, str) or variedade not in ("conexao", "equilibrada", "variedade"):
        return jsonify({"erro": "Selecione uma opção de variedade válida."}), 400
    selecionadas = body.get("emocoes", [])
    if not isinstance(selecionadas, list) or len(selecionadas) > 8 or not all(isinstance(e, str) and e in EMOCOES_KEYWORDS for e in selecionadas):
        return jsonify({"erro": "Selecione sentimentos disponíveis na interface."}), 400
    selecionadas = list(dict.fromkeys(selecionadas))
    emocoes = [(e, 1 / len(selecionadas)) for e in selecionadas] if selecionadas else detectar_emocoes(query)
    tipo_sentimento = classificar_ambiguidade(query, emocoes)
    label_emocao = formatar_label_emocoes(emocoes, tipo_sentimento)
    lambda_mmr = {"conexao": 0.85, "equilibrada": None, "variedade": 0.50}[variedade]
    try:
        resultados = recomendar_livros(
            query, emocoes, tipo_sentimento, top_k=top_k, genero=genero,
            paginas_max=paginas_max, lambda_mmr=lambda_mmr,
            titulos_excluidos=livros_penalizados(),
        )
        if resultados.empty:
            return jsonify({"erro": "Nenhum livro encontrado com essas preferências. Experimente outro gênero ou tamanho de leitura."}), 404
        def _san(v):
            if isinstance(v, np.generic):
                v = v.item()
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                return None
            return v
        livros = [{k: _san(v) for k, v in row.items()} for row in resultados.to_dict(orient="records")]
        analise = analisar_top_livro(query, livros[0], tipo_sentimento)
        return jsonify({"livros": livros, "emocao": label_emocao, "analise_ia": analise, "ia_habilitada": _API_KEY_VALIDA})
    except Exception:
        logger.exception("Erro ao recomendar livros")
        return jsonify({"erro": "Não foi possível preparar as sugestões agora. Tente novamente em instantes."}), 500


@app.route("/opcoes")
def rota_opcoes():
    """Oferece os gêneros reais do catálogo para preencher os filtros da interface."""
    generos = []
    if "Gênero" in df.columns:
        generos = sorted({str(g).strip() for g in df["Gênero"].dropna() if str(g).strip() and str(g).strip().lower() not in ("nan", "none", "n/a")}, key=str.casefold)
    return jsonify({"generos": generos, "livros": len(df)})


@app.route("/feedback", methods=["POST"])
def rota_feedback():
    ip = request.remote_addr or "local"
    if not rate_feedback(ip):
        return jsonify({"erro": "Muitos feedbacks. Aguarde."}), 429
    body   = request.get_json(force=True)
    titulo = body.get("titulo", "").strip()
    util   = bool(body.get("util"))
    query  = body.get("query", "")
    ok, msg = registrar_feedback(titulo, util, query)
    if not ok:
        return jsonify({"erro": msg}), 400
    return jsonify({"ok": True})

@app.route("/historico")
def rota_historico():
    return jsonify({"historico": ler_historico_feedback()})

@app.route("/debug_emocao", methods=["POST"])
def debug_emocao():
    body  = request.get_json(force=True)
    texto = body.get("query", "")
    return jsonify({"limpo": limpar_texto(texto), "emocoes": detectar_emocoes(texto)})

@app.route("/status")
def status():
    # Lê métricas reais de feedback
    try:
        with open(Config.FEEDBACK_PATH, "r", encoding="utf-8") as f:
            fb_dados = json.load(f)
        total_fb   = len(fb_dados)
        uteis      = sum(1 for x in fb_dados if x.get("util"))
        nao_uteis  = total_fb - uteis
    except FileNotFoundError:
        total_fb = uteis = nao_uteis = 0

    # Verifica existência do cache de embeddings
    cache_existe = os.path.exists(Config.CACHE_PATH)

    return jsonify({
        "status":           "ok",
        "https":            True,
        "ia_habilitada":    _API_KEY_VALIDA,
        "livros":           len(df),
        "feedback_total":   total_fb,
        "feedback_uteis":   uteis,
        "feedback_nao_uteis": nao_uteis,
        "embeddings_cache": cache_existe,
        "modelo":           MODELO_NOME,
    })

# ============================================================
# 12. INICIALIZAÇÃO
# ============================================================
logger.info("Carregando dados do MongoDB...")
df = carregar_dados()
logger.info(f"Dataset: {len(df)} livros")

_titulos_validos = set(df["Título do Livro"].dropna().tolist())

if _API_KEY_VALIDA:
    logger.info("✅ GEMINI_API_KEY configurada — análise de IA habilitada.")
else:
    logger.warning("ℹ️  GEMINI_API_KEY não definida.")
    logger.warning("    No CMD Windows execute ANTES de rodar o script:")
    logger.warning("    set GEMINI_API_KEY=AIzaSy...suachave")
    logger.warning("    Chave gratuita em: aistudio.google.com/apikey")

logger.info("Carregando modelo e embeddings...")
modelo, embeddings_livros = carregar_modelo_embeddings(df)

logger.info("Treinando classificador...")
pipeline_clf = treinar_classificador(df)

pre_filtro_info = _calcular_pre_filtro(len(df))
logger.info(f"Pre-filtro dinâmico: {pre_filtro_info} candidatos para {len(df)} livros")

gerar_cert_autoassinado()

logger.info(f"\n✅ Tudo pronto!")
logger.info(f"   🔒 HTTPS → https://localhost:{HTTPS_PORT}")
logger.info(f"   🔀 HTTP  → http://localhost:{HTTP_PORT}  (redireciona para HTTPS)\n")

if __name__ == "__main__":
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(CERT_FILE, KEY_FILE)

    def rodar_redirect():
        redirect_app.run(host="0.0.0.0", port=HTTP_PORT, debug=False, use_reloader=False)

    t = threading.Thread(target=rodar_redirect, daemon=True)
    t.start()

    app.run(host="0.0.0.0", port=HTTPS_PORT, debug=False, ssl_context=ctx)
