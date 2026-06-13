import streamlit as st
from dotenv import load_dotenv

from db_sql   import init_sql, upsert_embedding, semantic_search, log_search, get_search_stats
from db_mongo import init_mongo, upsert_product, get_product_by_id
from ai       import get_embedding, generate_summary, seed_products

load_dotenv()
init_sql()
init_mongo()

st.set_page_config(page_title="Smart Product Search", page_icon="🔍", layout="wide")
st.title("🔍 Smart Product Search")
st.caption("PostgreSQL pgvector · MongoDB · Redis · Gemini AI")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Admin")
    if st.button("🌱 Seed productos demo", use_container_width=True):
        with st.spinner("Indexando..."):
            n = seed_products(upsert_product, upsert_embedding, get_embedding)
        st.success(f"{n} productos indexados")

    st.divider()
    st.subheader("📊 Estadísticas")
    for label, value in get_search_stats():
        st.metric(label, value)

# ── Search ─────────────────────────────────────────────────────────────────────
col1, col2 = st.columns([4, 1])
with col1:
    query = st.text_input(
        "Busca en lenguaje natural",
        placeholder="ej: algo para cocinar arroz rápido, bebida sin azúcar para deportistas...",
    )
with col2:
    top_k = st.selectbox("Resultados", [3, 5, 10], index=1)

if query:
    with st.spinner("Buscando..."):
        vec  = get_embedding(query)
        hits = semantic_search(vec, top_k=top_k)          # [(product_id, score)]
        docs = [get_product_by_id(pid) for pid, _ in hits]
        docs = [d for d in docs if d]
        log_search(query, len(docs))

    if not docs:
        st.warning("Sin resultados. Usa **Seed productos demo** en el sidebar primero.")
    else:
        with st.spinner("Generando recomendación IA..."):
            summary = generate_summary(query, docs)
        st.info(f"🤖 **IA:** {summary}")
        st.divider()

        cols = st.columns(min(len(docs), 3))
        for i, (doc, (_, score)) in enumerate(zip(docs, hits)):
            with cols[i % 3]:
                st.markdown(f"### {doc.get('name', 'N/A')}")
                st.markdown(f"**Categoría:** {doc.get('category', '-')}")
                st.markdown(f"**Precio:** S/ {doc.get('price', 0):.2f}")
                st.markdown(f"**Stock:** {doc.get('stock', 0)} u.")
                st.markdown(f"**Relevancia:** `{round(score * 100, 1)}%`")
                st.markdown(f"_{doc.get('description', '')}_")
                st.divider()
