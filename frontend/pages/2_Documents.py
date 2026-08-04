import os
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="My Documents", page_icon="📁", layout="wide")

st.title("📁 My Documents")
st.write("View uploaded documents and their extracted metadata.")

try:
    response = requests.get(f"{BACKEND_URL}/api/documents")
    if response.status_code == 200:
        docs = response.json().get("data", [])
        
        if not docs:
            st.info("No documents uploaded yet.")
        else:
            # Show list of docs
            for doc in docs:
                with st.expander(f"{doc['filename']} - {doc['status'].upper()}", expanded=st.session_state.get("active_doc_id") == doc["document_id"]):
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.write(f"**ID:** `{doc['document_id'][:8]}...`")
                        st.write(f"**Size:** {doc['file_size_bytes'] / 1024:.1f} KB")
                        st.write(f"**Hint:** {doc['hint']}")
                        if st.button("Ask Questions", key=f"btn_{doc['document_id']}"):
                            st.session_state.active_doc_id = doc['document_id']
                            st.switch_page("pages/3_Ask.py")
                            
                    with col2:
                        if doc.get("classification"):
                            st.subheader("Classification")
                            cls = doc["classification"]
                            
                            st.markdown(f"""
                            - **Subject:** {cls.get('subject')}
                            - **Grade:** {cls.get('grade_level')}
                            - **Topic:** {cls.get('topic')}
                            """)
                            
                            st.write("**Key Concepts:**")
                            for kc in cls.get("key_concepts", []):
                                st.write(f"- {kc}")
                                
                            st.json(cls) # raw view
                        elif doc["status"] == "error":
                            st.error(doc.get("error_message"))
                        else:
                            st.info("Classification not ready yet.")
                            
    else:
        st.error("Failed to load documents.")
except requests.exceptions.ConnectionError:
    st.error("Could not connect to backend.")
