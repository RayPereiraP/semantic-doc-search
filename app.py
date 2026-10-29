import streamlit as st

from search_engine import SearchEngine, extract_text

st.set_page_config(page_title="Busca Semântica", page_icon="🔎")
st.title("🔎 Busca semântica em documentos")
st.caption("Envie PDFs ou TXTs e busque pelo significado, não só por palavra-chave.")


@st.cache_resource
def get_engine() -> SearchEngine:
    return SearchEngine()


engine = get_engine()

files = st.file_uploader(
    "Envie seus documentos", type=["pdf", "txt"], accept_multiple_files=True
)

if files and st.button("Indexar documentos"):
    with st.spinner("Indexando..."):
        docs = {f.name: extract_text(f.name, f.read()) for f in files}
        try:
            engine.add_documents(docs)
            st.success(f"{len(engine.chunks)} trechos indexados.")
        except ValueError as e:
            st.error(str(e))

query = st.text_input("O que você quer encontrar?")

if query:
    if engine.index is None:
        st.warning("Indexe pelo menos um documento primeiro.")
    else:
        for r in engine.search(query, k=3):
            st.markdown(f"**{r['source']}** · similaridade {r['score']:.2f}")
            st.write(r["text"])
            st.divider()