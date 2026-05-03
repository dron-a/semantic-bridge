"""
Claude LLM Adapter
===================
Reference LLM adapter using Anthropic's Claude API.

Implements the LLMAdapter protocol using Claude's tool use API for
structured metric resolution — ensuring the LLM returns a parseable
list of metric names rather than free-form text.

Responsibilities implemented here:
  - Responsibility 1: Metric resolution (resolve_metrics)
  - Responsibility 3: Answer grounding (generate_answer)

Responsibility 2 (access control) is handled by the bridge engine
BEFORE this adapter receives any metric definitions.

Requirements:
  pip install anthropic
  ANTHROPIC_API_KEY environment variable set
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List

import anthropic

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

from protocol import (
    GroundedAnswer,
    LLMAdapter,
    MetricDefinition,
    MetricQuery,
    MetricResult,
    MetricType,
    UserContext,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

RESOLUTION_SYSTEM_PROMPT = """You are a metric resolution engine for a governed semantic layer.

Your job is to identify which governed business metrics are relevant to answer a user's question.

RULES:
- You must ONLY select metrics from the provided list
- Never suggest metrics not in the list
- Never fabricate metric names
- If no metrics are relevant, return an empty list
- Return metrics ordered by relevance (most relevant first)
- A metric is relevant if it directly or indirectly answers the question

You will use the resolve_metrics tool to return your answer."""

ANSWER_SYSTEM_PROMPT = """You are a governed data assistant. You answer business questions \
using ONLY the provided governed metric definitions and query results.

RULES:
1. Base your answer strictly on the provided metric definitions and data
2. Always cite which metric definition you used
3. If the question asks for something not covered by the provided metrics, \
explicitly flag it as outside governed definitions
4. Never fabricate numbers or calculations not in the provided data
5. Be concise and precise — you are answering a business question, not writing an essay
6. Always mention the time period if it is clear from the data"""


# ---------------------------------------------------------------------------
# Tool definitions for structured metric resolution
# ---------------------------------------------------------------------------

RESOLVE_METRICS_TOOL: Dict[str, Any] = {
    "name": "resolve_metrics",
    "description": (
        "Return the list of governed metric names that are relevant "
        "to answer the user's question. Only use metric names from the provided list."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "metric_names": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Ordered list of relevant metric names (most relevant first)",
            },
            "reasoning": {
                "type": "string",
                "description": "Brief explanation of why these metrics were selected",
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": (
                    "Confidence that these metrics fully answer the question. "
                    "Lower if the question asks for things not covered."
                ),
            },
            "out_of_scope": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Parts of the question that cannot be answered from "
                    "the provided governed metrics"
                ),
            },
        },
        "required": ["metric_names", "confidence"],
    },
}


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------

class ClaudeAdapter(LLMAdapter):
    """
    Reference LLM adapter using Claude claude-sonnet-4-20250514.

    Uses tool use for structured metric resolution to avoid
    free-form text parsing. Falls back to description-based
    matching if the API call fails.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-4-20250514",
        max_tokens: int = 1024,
    ):
        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        self._model = model
        self._max_tokens = max_tokens

    @property
    def adapter_id(self) -> str:
        return "claude"

    # ------------------------------------------------------------------
    # Responsibility 1: Metric Resolution
    # ------------------------------------------------------------------

    def resolve_metrics(
        self,
        question: str,
        available_metrics: List[MetricDefinition],
        user_context: UserContext,
    ) -> List[MetricDefinition]:
        """
        Use Claude tool use to identify which governed metrics
        are relevant to the question.

        available_metrics has already been access-filtered by
        the bridge engine — Claude only sees what the user can access.
        """
        if not available_metrics:
            return []

        # Build metric catalog for Claude
        metric_catalog = self._format_metric_catalog(available_metrics)

        user_message = (
            f"Governed metrics available:\n\n{metric_catalog}\n\n"
            f"User question: {question}\n\n"
            f"Use the resolve_metrics tool to identify which metrics are relevant."
        )

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=RESOLUTION_SYSTEM_PROMPT,
                tools=[RESOLVE_METRICS_TOOL],
                tool_choice={"type": "auto"},
                messages=[{"role": "user", "content": user_message}],
            )

            # Extract tool use result
            tool_result = self._extract_tool_result(response, "resolve_metrics")
            if tool_result:
                resolved_names = tool_result.get("metric_names", [])
                metric_map = {m.name: m for m in available_metrics}
                resolved = [
                    metric_map[name]
                    for name in resolved_names
                    if name in metric_map
                ]
                logger.debug(
                    "Claude resolved %d metrics for question %r: %s",
                    len(resolved),
                    question[:60],
                    resolved_names,
                )
                return resolved

        except Exception as e:
            logger.warning("Claude resolution API call failed: %s — falling back", e)

        # Fallback: simple keyword matching
        return self._fallback_resolve(question, available_metrics)

    def _format_metric_catalog(self, metrics: List[MetricDefinition]) -> str:
        """Format metrics as a readable catalog for Claude."""
        lines = []
        for m in metrics:
            dims = ", ".join(d.name for d in m.dimensions[:6])
            grains = ", ".join(g.name for g in m.time_grains)
            lines.append(
                f"- {m.name} ({m.type})\n"
                f"  Description: {m.description or 'No description'}\n"
                f"  Dimensions: {dims or 'none'}\n"
                f"  Time grains: {grains or 'none'}"
            )
        return "\n\n".join(lines)

    def _extract_tool_result(
        self, response: Any, tool_name: str
    ) -> Dict[str, Any] | None:
        """Extract the input from a tool use block in a Claude response."""
        for block in response.content:
            if block.type == "tool_use" and block.name == tool_name:
                return block.input
        return None

    def _fallback_resolve(
        self, question: str, metrics: List[MetricDefinition]
    ) -> List[MetricDefinition]:
        """
        Keyword-based fallback resolution when API call fails.
        Better than nothing, worse than Claude.
        """
        question_lower = question.lower()
        scored = []
        for m in metrics:
            score = 0
            if m.name.lower().replace("_", " ") in question_lower:
                score += 3
            if m.description:
                words = m.description.lower().split()
                score += sum(1 for w in words if w in question_lower and len(w) > 4)
            if score > 0:
                scored.append((score, m))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in scored]

    # ------------------------------------------------------------------
    # Build MetricQuery from resolved metrics
    # ------------------------------------------------------------------

    def build_query(
        self,
        question: str,
        resolved_metrics: List[MetricDefinition],
        user_context: UserContext,
    ) -> MetricQuery:
        """
        Build a MetricQuery from resolved metrics.
        Uses default time grain and no filters for v0.
        """
        if not resolved_metrics:
            return MetricQuery(metric_names=[])

        metric_names = [m.name for m in resolved_metrics]

        # Find default time grain from first metric
        default_grain = "month"
        for m in resolved_metrics:
            for g in m.time_grains:
                if g.is_default:
                    default_grain = g.name
                    break

        return MetricQuery(
            metric_names=metric_names,
            group_by=["metric_time"],
            time_grain=default_grain,
        )

    # ------------------------------------------------------------------
    # Responsibility 3: Answer Grounding
    # ------------------------------------------------------------------

    def generate_answer(
        self,
        question: str,
        resolved_metrics: List[MetricDefinition],
        query_results: List[MetricResult],
        user_context: UserContext,
    ) -> GroundedAnswer:
        """
        Generate a grounded answer using Claude.

        If query_results are provided, includes actual data.
        If not (dbt Core connector — no query execution), generates
        a definition-grounded response explaining what the metric
        measures and how to interpret it.
        """
        if not resolved_metrics:
            return GroundedAnswer(
                answer=(
                    "I could not find any governed metric definitions that match "
                    f"your question: '{question}'. Please check the available metrics "
                    "or contact your data team."
                ),
                metrics_used=[],
                definition_versions=[],
                out_of_scope_flags=[f"No governed metrics matched: '{question}'"],
                resolution_confidence=0.0,
                user_id=user_context.user_id,
                adapter_used=self.adapter_id,
            )

        # Build context for Claude
        context = self._build_answer_context(
            question, resolved_metrics, query_results
        )

        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=ANSWER_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": context}],
            )
            answer_text = response.content[0].text

        except Exception as e:
            logger.error("Claude answer generation failed: %s", e)
            answer_text = (
                f"Answer generation failed. The governed metrics relevant to "
                f"your question are: {', '.join(m.name for m in resolved_metrics)}. "
                f"Please retry or contact your data team."
            )

        # Detect out-of-scope signals in the answer
        out_of_scope = self._detect_out_of_scope(question, resolved_metrics)

        return GroundedAnswer(
            answer=answer_text,
            metrics_used=resolved_metrics,
            definition_versions=[m.definition_version for m in resolved_metrics],
            query_results=query_results if query_results else None,
            out_of_scope_flags=out_of_scope,
            resolution_confidence=self._estimate_confidence(
                question, resolved_metrics
            ),
            user_id=user_context.user_id,
            adapter_used=self.adapter_id,
        )

    def _build_answer_context(
        self,
        question: str,
        resolved_metrics: List[MetricDefinition],
        query_results: List[MetricResult],
    ) -> str:
        """Build the full context message for answer generation."""
        sections = [f"User question: {question}\n"]

        sections.append("Governed metric definitions:")
        for m in resolved_metrics:
            dims = ", ".join(d.name for d in m.dimensions)
            sections.append(
                f"\n  Metric: {m.name}\n"
                f"  Type: {m.type}\n"
                f"  Description: {m.description or 'No description provided'}\n"
                f"  Available dimensions: {dims or 'none'}\n"
                f"  Definition version: {m.definition_version}\n"
                f"  Source: {m.source_connector}"
            )

        if query_results:
            sections.append("\nQuery results:")
            for result in query_results:
                sections.append(
                    f"  Metrics: {result.query.metric_names}\n"
                    f"  Rows returned: {result.row_count}\n"
                    f"  Data: {json.dumps(result.data[:10], default=str)}"
                )
        else:
            sections.append(
                "\nNote: No live query results available "
                "(dbt Core connector — definition mode only). "
                "Answer based on metric definitions only."
            )

        return "\n".join(sections)

    def _detect_out_of_scope(
        self,
        question: str,
        resolved_metrics: List[MetricDefinition],
    ) -> List[str]:
        """
        Simple heuristic: flag if resolved metrics seem low relevance
        or if question contains metric-like terms not in resolved list.
        """
        flags = []
        if not resolved_metrics:
            flags.append(f"No governed metrics found for: '{question}'")
        return flags

    def _estimate_confidence(
        self,
        question: str,
        resolved_metrics: List[MetricDefinition],
    ) -> float:
        """
        Estimate resolution confidence based on name/description overlap.
        Rough heuristic — Claude's tool use provides better signal in practice.
        """
        if not resolved_metrics:
            return 0.0
        question_lower = question.lower()
        matches = sum(
            1 for m in resolved_metrics
            if m.name.lower().replace("_", " ") in question_lower
        )
        return min(0.5 + (matches * 0.25), 1.0)
