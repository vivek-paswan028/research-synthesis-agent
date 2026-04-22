"""
Streamlit UI for Research Synthesis Agent Demo
"""

import streamlit as st
import httpx
import asyncio
from datetime import datetime

API_BASE = "http://localhost:8080/api/v1"


def init_state():
    """Initialize session state."""
    if "report" not in st.session_state:
        st.session_state.report = None
    if "topic" not in st.session_state:
        st.session_state.topic = ""
    if "monitoring" not in st.session_state:
        st.session_state.monitoring = False
    if "collections" not in st.session_state:
        st.session_state.collections = []


def call_api(method: str, endpoint: str, **kwargs):
    """Make API call to backend."""
    url = f"{API_BASE}{endpoint}"
    try:
        if method == "GET":
            response = httpx.get(url, timeout=60)
        elif method == "POST":
            response = httpx.post(url, json=kwargs.get("data", {}), timeout=120)
        elif method == "DELETE":
            response = httpx.delete(url, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def run_synthesis(topic: str):
    """Run research synthesis via API."""
    try:
        response = httpx.post(
            f"{API_BASE}/research/synthesize",
            json={"topic": topic, "max_sources": 8, "report_type": "comprehensive"},
            timeout=120,
        )
        response.raise_for_status()
        return response.json()
    except Exception as e:
        return {"error": str(e)}


st.set_page_config(
    page_title="Research Synthesis Agent",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 Research Synthesis Agent")
st.markdown("*Autonomous web research with RAG-powered synthesis*")

init_state()

# Sidebar
with st.sidebar:
    st.header("Controls")

    # Topic input
    topic = st.text_input(
        "Research Topic",
        value=st.session_state.topic,
        placeholder="e.g., latest developments in quantum computing",
    )

    # Research button
    if st.button("🔍 Research", type="primary", use_container_width=True):
        if topic:
            st.session_state.topic = topic
            with st.spinner("Searching the web..."):
                result = run_synthesis(topic)
                if "error" not in result:
                    st.session_state.report = result
                    st.success("Research complete!")
                else:
                    st.error(f"Error: {result['error']}")
        else:
            st.warning("Please enter a research topic")

    st.divider()

    # Monitoring toggle
    if st.session_state.report:
        monitoring = st.toggle("🔄 Enable Monitoring", value=st.session_state.monitoring)
        st.session_state.monitoring = monitoring
        if monitoring:
            response = call_api("POST", f"/research/{topic}/monitor", data={"interval_hours": 6})
            if "error" not in response:
                st.success("Monitoring started!")
            else:
                st.info("Backend not running. Start with: uvicorn app.main:app --reload --port 8000")

    st.divider()

    # Collections
    st.subheader("Previous Research")
    collections = call_api("GET", "/collections")
    if "collections" in collections:
        for coll in collections.get("collections", []):
            if st.button(f"📁 {coll}", use_container_width=True):
                st.session_state.topic = coll

# Main content
if st.session_state.report:
    report = st.session_state.report

    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader(f"📊 Report: {report['topic']}")

        # Report content
        st.markdown("---")
        st.markdown(report["content"])

    with col2:
        st.subheader("Sources")
        for i, source in enumerate(report["sources"], 1):
            st.markdown(f"{i}. [{source}]({source})")

        st.markdown("---")
        st.caption(f"Generated: {report['timestamp']}")
        st.caption(f"Iterations: {report['iteration']}")
else:
    st.info("👈 Enter a research topic and click 'Research' to begin")

    # Features display
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🔍 Web Search")
        st.markdown("Tavily-powered multi-query search for comprehensive coverage")
    with col2:
        st.markdown("### 📚 RAG Storage")
        st.markdown("ChromaDB vector store for intelligent context retrieval")
    with col3:
        st.markdown("### 🔄 Monitoring")
        st.markdown("Schedule periodic updates on your research topics")