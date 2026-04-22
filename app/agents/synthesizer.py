"""
Research Synthesizer — Main orchestrator with agentic loop
"""

import asyncio
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib

from .web_search import WebSearchAgent, SearchResult
from .extractor import ContentExtractor, ExtractedContent
from .report_gen import ReportGenerator


@dataclass
class ResearchReport:
    topic: str
    content: str
    sources: List[str]
    timestamp: datetime
    iteration: int

@dataclass
class SynthesisResult:
    report: ResearchReport
    search_results: List[SearchResult]
    extracted_contents: List[ExtractedContent]
    iterations: int


class ResearchSynthesizer:
    def __init__(
        self,
        tavily_api_key: str,
        gemini_api_key: str,
        chroma_store,  # Will be injected
    ):
        self.search_agent = WebSearchAgent(tavily_api_key)
        self.extractor = ContentExtractor(max_concurrent=3)
        self.report_gen = ReportGenerator(gemini_api_key)
        self.chroma_store = chroma_store

    async def synthesize(
        self,
        topic: str,
        max_sources: int = 10,
        report_type: str = "comprehensive",
    ) -> ResearchReport:
        """
        Main entry point: run full research pipeline.
        """
        result = await self.run_agentic_loop(topic, max_sources=max_sources)
        return result.report

    async def run_agentic_loop(
        self,
        topic: str,
        max_sources: int = 10,
        max_iterations: int = 3,
        report_type: str = "comprehensive",
    ) -> SynthesisResult:
        """
        Iterative agentic loop with refinement.
        """
        iteration = 0
        all_search_results: List[SearchResult] = []
        all_extracted: List[ExtractedContent] = []

        for iteration in range(1, max_iterations + 1):
            # Step 1: Deep search
            search_results = await self.search_agent.deep_search(topic)

            # Deduplicate by URL
            existing_urls = {r.url for r in all_search_results}
            new_results = [r for r in search_results if r.url not in existing_urls]
            all_search_results.extend(new_results)

            if len(all_search_results) >= max_sources:
                break

            # Step 2: Extract content from new URLs
            urls_to_extract = [r.url for r in all_search_results if r.url not in existing_urls]
            if urls_to_extract:
                extracted = await self.extractor.batch_extract(urls_to_extract[:max_sources - len(all_search_results)])
                all_extracted.extend(extracted)

            # Step 3: Check if we have enough quality content
            valid_content = [e for e in all_extracted if e.content and len(e.content) > 200]
            if len(valid_content) >= 5:
                break

        # Store extracted content in ChromaDB
        await self._store_content(topic, all_extracted)

        # Retrieve relevant chunks for synthesis
        context_chunks = await self._retrieve_context(topic, top_k=8)

        # Generate report
        if report_type == "brief":
            report_content = await self.report_gen.generate_brief(topic, context_chunks)
        else:
            report_content = await self.report_gen.generate_report(topic, context_chunks)

        report = ResearchReport(
            topic=topic,
            content=report_content,
            sources=[r.url for r in all_search_results],
            timestamp=datetime.now(),
            iteration=iteration,
        )

        return SynthesisResult(
            report=report,
            search_results=all_search_results,
            extracted_contents=all_extracted,
            iterations=iteration,
        )

    async def query_research(self, topic: str, question: str) -> str:
        """Query past research with a question."""
        context_chunks = await self._retrieve_context(topic, top_k=5)
        return await self.report_gen.answer_question(topic, context_chunks, question)

    async def _search_topic(self, topic: str) -> List[SearchResult]:
        """Search for a topic without full synthesis (for monitoring)."""
        return await self.search_agent.deep_search(topic)

    async def _store_content(self, topic: str, contents: List[ExtractedContent]) -> None:
        """Store extracted content in ChromaDB."""
        documents = []
        ids = []
        metadatas = []

        for i, content in enumerate(contents):
            if not content.content or len(content.content) < 100:
                continue

            doc_id = hashlib.md5(content.url.encode()).hexdigest()
            documents.append(content.content)
            ids.append(doc_id)
            metadatas.append({
                "topic": topic,
                "source": content.url,
                "title": content.title,
                "domain": content.metadata.get("domain", ""),
                "timestamp": datetime.now().isoformat(),
            })

        if documents:
            await self.chroma_store.add_documents(documents, ids, metadatas, collection_name=topic)

    async def _retrieve_context(self, topic: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant context from ChromaDB."""
        results = await self.chroma_store.query(topic, topic, top_k=top_k)
        return results

    async def close(self):
        """Cleanup resources."""
        await self.extractor.close()