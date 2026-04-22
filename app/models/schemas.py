"""
Pydantic schemas for API requests/responses
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class SynthesizeRequest(BaseModel):
    topic: str = Field(..., description="Research topic to synthesize")
    max_sources: int = Field(10, description="Maximum sources to fetch")
    report_type: str = Field("comprehensive", description="Report type: comprehensive or brief")


class ResearchReportResponse(BaseModel):
    topic: str
    content: str
    sources: List[str]
    timestamp: datetime
    iteration: int


class MonitorRequest(BaseModel):
    topic: str
    interval_hours: int = Field(6, description="Hours between checks")


class HealthResponse(BaseModel):
    status: str
    collections_count: int
    timestamp: datetime


class ErrorResponse(BaseModel):
    detail: str
    code: str