"""
Web Search Agent using Tavily API
"""

import asyncio
from dataclasses import dataclass
from typing import List, Optional
from datetime import datetime

import httpx


@dataclass
class SearchResult:
    url: str
    title: str
    snippet: str
    score: float
    published_date: Optional[str] = None


class WebSearchAgent:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.tavily.com/search"
        self._semaphore = asyncio.Semaphore(3)  # Rate limiting

    async def search(self, query: str, max_results: int = 5) -> List[SearchResult]:
        """Single query search via Tavily."""
        async with self._semaphore:
            async with httpx.AsyncClient(timeout=30.0) as client:
                payload = {
                    "api_key": self.api_key,
                    "query": query,
                    "max_results": max_results,
                    "include_answer": "advanced",
                    "include_raw_content": False,
                    "include_images": False,
                }
                response = await client.post(self.base_url, json=payload)
                response.raise_for_status()
                data = response.json()

        results = []
        for item in data.get("results", []):
            results.append(SearchResult(
                url=item.get("url", ""),
                title=item.get("title", ""),
                snippet=item.get("content", ""),
                score=item.get("score", 0.0),
                published_date=item.get("published_date"),
            ))
        return results

    async def deep_search(self, topic: str) -> List[SearchResult]:
        """
        Multi-faceted search for comprehensive coverage.
        Generates multiple search queries for a single topic.
        """
        queries = self._generate_queries(topic)
        all_results = []

        tasks = [self.search(q, max_results=5) for q in queries]
        results_batches = await asyncio.gather(*tasks)

        seen_urls = set()
        for batch in results_batches:
            for result in batch:
                if result.url not in seen_urls:
                    seen_urls.add(result.url)
                    all_results.append(result)

        return all_results

    def _generate_queries(self, topic: str) -> List[str]:
        """Generate search queries for comprehensive coverage."""
        return [
            f"{topic}",
            f"{topic} latest research 2024",
            f"{topic} scientific papers",
            f"{topic} developments trends",
        ]