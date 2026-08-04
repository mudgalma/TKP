import os
import time
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="Upload Documents", page_icon="📤")

st.title("📤 Upload Documents")
st.write("Upload research papers, textbook chapters, lecture notes, PDFs, or presentations.")

ACCEPTED_TYPES = ["pdf", "pptx", "ppt", "docx", "doc", "txt"]

MIME_MAP = {
    "pdf": "application/pdf",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "ppt": "application/vnd.ms-powerpoint",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "doc": "application/msword",
    "txt": "text/plain",
}

hint = st.selectbox(
    "Document Type (helps routing)",
    options=["auto", "mostly_text", "tables", "equations", "scanned"],
    format_func=lambda x: {
        "auto": "Auto-detect",
        "mostly_text": "Mostly Text (Fast)",
        "tables": "Contains Tables",
        "equations": "Contains Equations",
        "scanned": "Scanned / OCR",
    }[x]
)

uploaded_file = st.file_uploader(
    "Choose a file",
    type=ACCEPTED_TYPES,
    accept_multiple_files=False,
)

if uploaded_file:
    ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
    st.write(f"**Selected:** {uploaded_file.name} ({ext.upper()})")

    if st.button("Upload & Process", type="primary"):
        mime = MIME_MAP.get(ext, "application/octet-stream")
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), mime)}
        data = {"hint": hint}
        
        try:
            with st.spinner("Starting upload..."):
                response = requests.post(f"{BACKEND_URL}/api/upload", files=files, data=data)
            
            if response.status_code == 200:
                res = response.json()
                doc_id = res["data"]["document_id"]
                st.session_state.active_doc_id = doc_id
                
                status_placeholder = st.empty()
                progress_bar = st.progress(0)
                
                # Poll for completion
                max_retries = 300 # 5 mins
                for _ in range(max_retries):
                    poll_res = requests.get(f"{BACKEND_URL}/api/documents/{doc_id}")
                    if poll_res.status_code == 200:
                        doc = poll_res.json()["data"]
                        status = doc["status"]
                        
                        prog_map = {
                            "uploading": 10,
                            "parsing": 30,
                            "chunking": 50,
                            "embedding": 70,
                            "storing": 80,
                            "classifying": 90,
                            "ready": 100,
                            "error": 100,
                        }
                        progress_bar.progress(prog_map.get(status, 0))
                        
                        if status == "ready":
                            status_placeholder.success("Processing complete! Document is ready.")
                            st.button("View Document Metadata", on_click=lambda: st.switch_page("pages/2_Documents.py"))
                            st.button("Ask Questions", on_click=lambda: st.switch_page("pages/3_Ask.py"))
                            break
                        elif status == "error":
                            status_placeholder.error(f"Error during processing: {doc.get('error_message')}")
                            break
                        else:
                            status_placeholder.info(f"Processing... Current step: **{status}**")
                            time.sleep(1)
                    else:
                        st.error("Failed to check status.")
                        break
            else:
                st.error(f"Upload failed: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error(f"Could not connect to backend. Is it running at {BACKEND_URL}?")
