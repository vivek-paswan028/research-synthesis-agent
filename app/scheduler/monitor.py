"""
Background monitoring service using APScheduler
"""

import asyncio
from typing import Dict, Optional, Callable
from datetime import datetime

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers import interval


class ResearchMonitor:
    def __init__(self, synthesizer):
        self.synthesizer = synthesizer
        self.scheduler = AsyncIOScheduler()
        self._monitoring_tasks: Dict[str, dict] = {}
        self._running = False

    def start(self):
        """Start the scheduler."""
        if not self._running:
            self.scheduler.start()
            self._running = True

    def stop(self):
        """Stop the scheduler."""
        if self._running:
            self.scheduler.shutdown(wait=False)
            self._running = False

    def add_monitoring_job(
        self,
        topic: str,
        interval_hours: int = 6,
        callback: Optional[Callable] = None,
    ) -> str:
        """
        Add a monitoring job for a topic.
        Returns job ID.
        """
        job_id = f"monitor_{topic.replace(' ', '_')}"

        # Store monitoring config
        self._monitoring_tasks[job_id] = {
            "topic": topic,
            "interval_hours": interval_hours,
            "last_check": None,
            "callback": callback,
        }

        # Add scheduler job
        self.scheduler.add_job(
            func=self._run_monitoring_check,
            trigger=interval(hours=interval_hours),
            args=[job_id],
            id=job_id,
            replace_existing=True,
        )

        return job_id

    def remove_monitoring_job(self, job_id: str) -> bool:
        """Remove a monitoring job."""
        try:
            self.scheduler.remove_job(job_id)
            if job_id in self._monitoring_tasks:
                del self._monitoring_tasks[job_id]
            return True
        except Exception:
            return False

    def get_monitoring_status(self, job_id: str) -> Optional[Dict]:
        """Get status of a monitoring job."""
        return self._monitoring_tasks.get(job_id)

    def list_active_monitors(self) -> list:
        """List all active monitoring jobs."""
        return [
            {"job_id": job_id, **config}
            for job_id, config in self._monitoring_tasks.items()
        ]

    async def _run_monitoring_check(self, job_id: str):
        """Internal method to run monitoring check."""
        if job_id not in self._monitoring_tasks:
            return

        task_info = self._monitoring_tasks[job_id]
        topic = task_info["topic"]
        callback = task_info.get("callback")

        try:
            # Run a fresh synthesis to check for updates
            new_results = await self.synthesizer._search_topic(topic)

            # Update last check time
            self._monitoring_tasks[job_id]["last_check"] = datetime.now()
            self._monitoring_tasks[job_id]["last_results_count"] = len(new_results)

            # If callback provided, invoke it
            if callback:
                await callback(topic, new_results)

        except Exception as e:
            self._monitoring_tasks[job_id]["last_error"] = str(e)
            self._monitoring_tasks[job_id]["last_error_time"] = datetime.now().isoformat()