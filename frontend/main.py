import streamlit as st

st.set_page_config(
    page_title="TKP — AI Document Assistant",
    page_icon="📄",
    layout="centered",
)

st.title("📄 TKP")
st.subheader("Upload PDFs. Ask Questions. Get Answers.")

# Initialize session state for tracking selected documents
if "active_doc_id" not in st.session_state:
    st.session_state.active_doc_id = None

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.page_link("pages/1_Upload.py", label="📤 Upload Documents", use_container_width=True)
    st.caption("Upload your PDF files for processing.")

with col2:
    st.page_link("pages/2_Documents.py", label="📁 My Documents", use_container_width=True)
    st.caption("View and manage uploaded documents.")

with col3:
    st.page_link("pages/3_Ask.py", label="💬 Ask Questions", use_container_width=True)
    st.caption("Ask questions from your documents.")

st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:grey;'>Powered by OpenAI & Supabase</p>",
    unsafe_allow_html=True,
)
