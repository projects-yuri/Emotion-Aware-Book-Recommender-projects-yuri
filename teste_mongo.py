
import os
import streamlit as st
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from pymongo.errors import (
    ServerSelectionTimeoutError,
    OperationFailure,
)

st.set_page_config(
    page_title="Diagnóstico MongoDB Atlas",
    page_icon="🔍"
)

st.title("🔍 Diagnóstico MongoDB Atlas")

uri = os.environ.get("MONGO_URI")

if not uri:
    st.error("MONGO_URI não foi configurada nos Secrets.")
    st.stop()

if st.button("Testar conexão"):
    client = None

    try:
        with st.spinner("Conectando ao MongoDB Atlas..."):
            client = MongoClient(
                uri,
                server_api=ServerApi("1"),
                serverSelectionTimeoutMS=10000,
                connectTimeoutMS=10000,
            )

            client.admin.command("ping")

            st.success("MongoDB conectado com sucesso!")

            colecao = client["dataset"]["dataset"]
            total = colecao.count_documents({})

            st.success(f"Documentos encontrados: {total}")

    except ServerSelectionTimeoutError as e:
        st.error("Falha na conexão com MongoDB Atlas")
        st.warning("Possível problema de rede, DNS ou cluster.")

        st.subheader("Detalhes do erro")
        # Exiba detalhes apenas em ambiente privado.
        # Logs podem conter informações de infraestrutura.
        st.code(str(e))

    except OperationFailure as e:
        st.error("Falha de autenticação ou permissão.")
        st.write(f"Código MongoDB: {e.code}")

    except Exception as e:
        st.error("Erro inesperado.")
        st.write(f"Tipo: {type(e).__name__}")

    finally:
        if client is not None:
            client.close()
