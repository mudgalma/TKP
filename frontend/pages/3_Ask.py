import requests
import os
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Ask Documents", page_icon="💬", layout="wide")

st.title("💬 Ask Questions")
st.write("Select a document and ask a question based on its content.")

# 1. Fetch available documents
docs = []
try:
    response = requests.get(f"{BACKEND_URL}/api/documents")
    if response.status_code == 200:
        docs = [d for d in response.json().get("data", []) if d["status"] == "ready"]
except requests.exceptions.ConnectionError:
    st.error("Could not connect to backend.")

if not docs:
    st.warning("No ready documents found. Please upload a document first.")
    st.stop()

# 2. Selectbox with session state default
doc_options = {d["document_id"]: f"{d['filename']} ({d['hint']})" for d in docs}
default_index = 0
if st.session_state.get("active_doc_id") in doc_options:
    default_index = list(doc_options.keys()).index(st.session_state.active_doc_id)

selected_id = st.selectbox(
    "Select a document", 
    options=list(doc_options.keys()), 
    format_func=lambda x: doc_options[x],
    index=default_index
)

# update session state
st.session_state.active_doc_id = selected_id

# 3. Query input
question = st.text_input("Your question", placeholder="e.g. What are the key prerequisites?")

if st.button("Ask", type="primary"):
    if not question:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving and generating answer..."):
            try:
                res = requests.post(
                    f"{BACKEND_URL}/api/query",
                    json={"document_id": selected_id, "question": question}
                )
                if res.status_code == 200:
                    data = res.json()["data"]
                    st.markdown("### Answer")
                    st.write(data["answer"])
                    
                    if data["citations"]:
                        st.markdown("### Citations")
                        for idx, c in enumerate(data["citations"], 1):
                            with st.expander(f"[{idx}] {c['page_citation']}: {c['section_title']}"):
                                st.write(f"...{c['snippet']}...")
                else:
                    st.error(f"Error: {res.json().get('detail', res.text)}")
            except Exception as e:
                st.error(f"Request failed: {e}")
