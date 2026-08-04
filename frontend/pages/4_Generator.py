import os
import requests
import json
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(page_title="TKP Generator", page_icon="⚙️", layout="wide")

st.title("⚙️ Teacher Knowledge Package Generator")
st.write("Generate a structured lesson plan and activities using the LangGraph AI Agents.")

# Ensure a document is selected
if "active_doc_id" not in st.session_state:
    st.warning("Please go to **Documents** and select a document first.")
    st.stop()

document_id = st.session_state.active_doc_id
st.write(f"**Target Document:** `{document_id}`")

if st.button("🚀 Start Generation", type="primary"):
    with st.spinner("Initializing AI Agents..."):
        try:
            res = requests.post(f"{BACKEND_URL}/api/generate", json={"document_id": document_id})
            res.raise_for_status()
            job_id = res.json()["data"]["job_id"]
            st.session_state.current_job_id = job_id
        except Exception as e:
            st.error(f"Failed to start generation: {e}")
            st.stop()

if "current_job_id" in st.session_state:
    job_id = st.session_state.current_job_id
    st.info(f"**Job ID:** `{job_id}`")
    
    # Progress console
    progress_placeholder = st.empty()
    status_placeholder = st.empty()
    
    if st.button("Refresh Status"):
        res = requests.get(f"{BACKEND_URL}/api/jobs/{job_id}")
        if res.status_code == 200:
            job = res.json()["data"]
            st.write(f"**Status:** {job['status']}")
            
            with progress_placeholder.container():
                for ev in job.get("progress_events", []):
                    st.text(ev)
                    
            if job["status"] == "completed":
                st.success("Generation Complete!")
                tkp_json = job["final_output"]
                st.download_button(
                    "💾 Download TKP (JSON)", 
                    data=json.dumps(tkp_json, indent=2), 
                    file_name="TeacherKnowledgePackage.json",
                    mime="application/json"
                )
                with st.expander("View Output", expanded=True):
                    st.json(tkp_json)
            elif job["status"] == "failed":
                st.error(f"Generation Failed: {job.get('error_message')}")
