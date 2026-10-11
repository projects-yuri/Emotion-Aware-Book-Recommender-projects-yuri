
import os
import streamlit as st
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from pymongo.errors import (
    ServerSelectionTimeoutError,
    OperationFailure,
)

st.title("Diagnóstico MongoDB Atlas")

uri = os.environ.get("MONGO_URI")

if not uri:
    st.error("MONGO_URI não foi configurada nos Secrets.")
    st.stop()

if st.button("Testar conexão"):
    try:
        client = MongoClient(
            uri,
            server_api=ServerApi("1"),
            serverSelectionTimeoutMS=10000,
            connectTimeoutMS=10000,
        )

        client.admin.command("ping")
        st.success("Conexão com o MongoDB estabelecida!")

        colecao = client["dataset"]["dataset"]
        total = colecao.count_documents({})
        st.success(f"Documentos encontrados: {total}")

    except ServerSelectionTimeoutError:
        st.error(
            "Timeout: verifique o endereço do cluster, "
            "Network Access e a disponibilidade do Atlas."
        )

    except OperationFailure as e:
        st.error(
            f"Falha de autenticação ou permissão. "
            f"Código MongoDB: {e.code}"
        )

    except Exception as e:
        st.error(f"Tipo de erro: {type(e).__name__}")
