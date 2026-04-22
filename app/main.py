"""
FastAPI application entry point
"""

import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

from app.agents.synthesizer import ResearchSynthesizer
from app.storage.vector_store import ChromaStore
from app.scheduler.monitor import ResearchMonitor
from app.models.schemas import (
    SynthesizeRequest,
    ResearchReportResponse,
    MonitorRequest,
    HealthResponse,
)


# Load environment
load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not TAVILY_API_KEY or not GEMINI_API_KEY:
    raise RuntimeError(
        "Missing required API keys: TAVILY_API_KEY and GEMINI_API_KEY. "
        "Create a .env file or export these variables before starting the app."
    )

# Global instances
chroma_store: ChromaStore = None
synthesizer: ResearchSynthesizer = None
monitor: ResearchMonitor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup."""
    global chroma_store, synthesizer, monitor

    chroma_store = ChromaStore(persist_directory="./chroma_data")
    synthesizer = ResearchSynthesizer(
        tavily_api_key=TAVILY_API_KEY,
        gemini_api_key=GEMINI_API_KEY,
        chroma_store=chroma_store,
    )
    monitor = ResearchMonitor(synthesizer)
    monitor.start()

    print("Research Synthesis Agent started")
    yield

    # Cleanup
    monitor.stop()
    await synthesizer.close()
    print("Research Synthesis Agent stopped")


app = FastAPI(
    title="Research Synthesis Agent",
    description="Autonomous agent for web research and report synthesis",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/v1/health")
async def health() -> HealthResponse:
    """Health check endpoint."""
    collections = await chroma_store.list_collections()
    return HealthResponse(
        status="healthy",
        collections_count=len(collections),
        timestamp=datetime.now(),
    )


@app.post("/api/v1/research/synthesize", response_model=ResearchReportResponse)
async def synthesize(request: SynthesizeRequest):
    """Trigger a new research synthesis."""
    if not TAVILY_API_KEY or not GEMINI_API_KEY:
        raise HTTPException(
            status_code=500,
            detail="API keys not configured. Set TAVILY_API_KEY and GEMINI_API_KEY in .env"
        )

    try:
        result = await synthesizer.run_agentic_loop(
            topic=request.topic,
            max_sources=request.max_sources,
            report_type=request.report_type,
        )
        return ResearchReportResponse(
            topic=result.report.topic,
            content=result.report.content,
            sources=result.report.sources,
            timestamp=result.report.timestamp,
            iteration=result.report.iteration,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/research/{topic}")
async def get_research(topic: str, q: str = None):
    """Query stored research on a topic."""
    try:
        if q:
            # Answer a question about the topic
            answer = await synthesizer.query_research(topic, q)
            return {"question": q, "answer": answer, "topic": topic}
        else:
            # Get collection stats
            stats = await chroma_store.get_collection_stats(topic)
            return {"topic": topic, **stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/research/{topic}/report")
async def get_report(topic: str):
    """Get the latest report for a topic."""
    try:
        # Query for recent research
        results = await chroma_store.query(topic, topic, top_k=10)
        if not results:
            raise HTTPException(status_code=404, detail=f"No research found for topic: {topic}")

        return {
            "topic": topic,
            "context_chunks": results,
            "timestamp": datetime.now().isoformat(),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/research/{topic}/monitor")
async def start_monitoring(topic: str, request: MonitorRequest):
    """Start monitoring a topic for updates."""
    try:
        job_id = monitor.add_monitoring_job(
            topic=topic,
            interval_hours=request.interval_hours,
        )
        return {"job_id": job_id, "topic": topic, "interval_hours": request.interval_hours}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/research/{topic}/monitor")
async def stop_monitoring(topic: str):
    """Stop monitoring a topic."""
    job_id = f"monitor_{topic.replace(' ', '_')}"
    success = monitor.remove_monitoring_job(job_id)
    if success:
        return {"message": f"Stopped monitoring {topic}"}
    else:
        raise HTTPException(status_code=404, detail=f"No monitoring job found for {topic}")


@app.get("/api/v1/monitoring/status")
async def get_monitoring_status():
    """Get status of all monitoring jobs."""
    return {"monitors": monitor.list_active_monitors()}


@app.get("/api/v1/collections")
async def list_collections():
    """List all research collections."""
    collections = await chroma_store.list_collections()
    return {"collections": collections}