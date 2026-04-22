"""
Report Generator using Google Gemini
"""

from typing import List, Dict, Any

from langchain_google_genai import ChatGoogleGenerativeAI


class ReportGenerator:
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash-lite"):
        self.model_name = model
        self._api_key = api_key
        self._llm = None

    @property
    def llm(self):
        if self._llm is None:
            self._llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                temperature=0.7,
                max_output_tokens=4000,
                google_api_key=self._api_key,
            )
        return self._llm

    async def generate_report(
        self,
        topic: str,
        context_chunks: List[Dict[str, Any]],
        report_type: str = "comprehensive",
    ) -> str:
        """
        Generate a structured research report from context chunks.
        """
        context_text = self._build_context(context_chunks)

        system_prompt = """You are an expert research synthesizer. Create a comprehensive,
well-structured report on the given topic using the provided research context.

Report structure:
1. Executive Summary (2-3 sentences)
2. Key Findings (bullet points)
3. Detailed Analysis (paragraphs)
4. Sources & References
5. Monitoring Recommendations

Be analytical, cite sources inline, and highlight contradictions or consensus in the literature.
Report in markdown format."""

        user_prompt = f"""Topic: {topic}

Research Context:
{context_text}

Generate a comprehensive research report following the required structure.
Focus on facts from the sources and highlight key insights."""

        response = self.llm.invoke([{"role": "user", "content": user_prompt}])
        # Handle different response types from LangChain
        if hasattr(response, "content"):
            return response.content
        elif hasattr(response, "text"):
            return response.text
        else:
            return str(response)

    async def generate_brief(
        self,
        topic: str,
        context_chunks: List[Dict[str, Any]],
    ) -> str:
        """Generate a brief summary instead of full report."""
        context_text = self._build_context(context_chunks)

        response = self.llm.invoke([{
            "role": "user",
            "content": f"Topic: {topic}\n\nContext:\n{context_text}\n\nProvide a brief 3-paragraph summary with key findings."
        }])
        if hasattr(response, "content"):
            return response.content
        elif hasattr(response, "text"):
            return response.text
        return str(response)

    async def answer_question(
        self,
        topic: str,
        context_chunks: List[Dict[str, Any]],
        question: str,
    ) -> str:
        """Answer a specific question about the research."""
        context_text = self._build_context(context_chunks)

        response = self.llm.invoke([{
            "role": "user",
            "content": f"Topic: {topic}\n\nContext:\n{context_text}\n\nQuestion: {question}\n\nAnswer based on the provided research context. Cite sources."
        }])
        if hasattr(response, "content"):
            return response.content
        elif hasattr(response, "text"):
            return response.text
        return str(response)

    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Build context string from retrieved chunks."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.get("metadata", {}).get("source", "Unknown")
            title = chunk.get("title", "Untitled")
            content = chunk.get("content", "")[:1500]
            parts.append(f"[Source {i}: {title} ({source})]\n{content}")
        return "\n\n".join(parts)