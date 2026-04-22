"""
Content Extraction Agent — fetches and parses content from URLs
"""

import asyncio
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup


@dataclass
class ExtractedContent:
    url: str
    title: str
    content: str
    summary: str
    metadata: Dict[str, Any]


class ContentExtractor:
    def __init__(self, max_concurrent: int = 3):
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                follow_redirects=True,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; ResearchAgent/1.0)"
                }
            )
        return self._client

    async def extract_from_url(self, url: str) -> ExtractedContent:
        """Extract content from a single URL."""
        async with self._semaphore:
            client = await self._get_client()
            try:
                response = await client.get(url)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, "lxml")

                # Remove script and style elements
                for tag in soup(["script", "style", "nav", "footer", "header"]):
                    tag.decompose()

                title = soup.title.string if soup.title else ""
                if not title:
                    title_tag = soup.find("h1")
                    title = title_tag.get_text(strip=True) if title_tag else ""

                # Extract main content
                article = soup.find("article") or soup.find("main") or soup.find("body")
                if article:
                    text = article.get_text(separator="\n", strip=True)
                else:
                    text = soup.get_text(separator="\n", strip=True)

                # Truncate very long content
                text = text[:8000] if len(text) > 8000 else text

                # Clean up whitespace
                lines = [line.strip() for line in text.split("\n") if line.strip()]
                text = "\n".join(lines)

                # Generate summary (first 500 chars)
                summary = text[:500] + "..." if len(text) > 500 else text

                return ExtractedContent(
                    url=url,
                    title=title or url,
                    content=text,
                    summary=summary,
                    metadata={
                        "domain": urlparse(url).netloc,
                        "content_length": len(text),
                    }
                )

            except Exception as e:
                return ExtractedContent(
                    url=url,
                    title=url,
                    content="",
                    summary=f"Failed to extract: {str(e)}",
                    metadata={"error": str(e)}
                )

    async def batch_extract(self, urls: List[str]) -> List[ExtractedContent]:
        """Extract content from multiple URLs concurrently."""
        tasks = [self.extract_from_url(url) for url in urls]
        return await asyncio.gather(*tasks)

    async def smart_extract(self, url: str, focus_topics: List[str]) -> ExtractedContent:
        """Extract and focus on specific topics."""
        content = await self.extract_from_url(url)
        # Future: could add LLM-based filtering here
        return content

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()