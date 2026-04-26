"""
semantic-bridge LLM adapter protocol
=======================================
Abstract base class that every LLM adapter must implement.

Rules for adapter implementors:
  1. resolve_metrics() must use ONLY the provided metric list — no hallucination
  2. generate_answer() must populate metrics_used and definition_versions
  3. out_of_scope_flags must be populated when the question exceeds governed definitions
  4. Adapters never receive unfiltered metric lists — access control runs before this
  5. Adapters must never fabricate metric definitions not in the provided list

Reference implementations: claude, openai
"""

from abc import ABC, abstractmethod
from typing import List

from .models import (
    GroundedAnswer,
    MetricDefinition,
    MetricQuery,
    MetricResult,
    UserContext,
)


class LLMAdapter(ABC):
    """
    Contract for all LLM adapter implementations.

    The adapter is responsible for the intelligence layer:
      - Understanding which metrics a question is asking about
      - Formatting governed definitions as LLM context
      - Generating answers grounded strictly in those definitions
      - Flagging anything outside governed scope
    """

    @property
    @abstractmethod
    def adapter_id(self) -> str:
        """
        Stable identifier for this adapter.
        e.g. 'claude', 'openai', 'gemini'
        Stored in GroundedAnswer.adapter_used.
        """

    @abstractmethod
    def resolve_metrics(
        self,
        question: str,
        available_metrics: List[MetricDefinition],
        user_context: UserContext,
    ) -> List[MetricDefinition]:
        """
        Identify which governed metrics are relevant to this question.

        IMPORTANT:
          - available_metrics has already been access-filtered by the bridge
          - Return only metrics from available_metrics — never fabricate new ones
          - Return empty list if no metrics are relevant (triggers out-of-scope flag)
          - Order by relevance descending

        This is the metric resolution step (Responsibility 1).
        It runs BEFORE any answer is attempted.
        """

    @abstractmethod
    def build_query(
        self,
        question: str,
        resolved_metrics: List[MetricDefinition],
        user_context: UserContext,
    ) -> MetricQuery:
        """
        Translate the natural language question + resolved metric definitions
        into a structured MetricQuery for execution against the semantic layer.

        Must respect:
          - Only dimensions present in resolved_metrics.dimensions
          - Only time_grains present in resolved_metrics.time_grains
          - Filters must reference valid dimension names
        """

    @abstractmethod
    def generate_answer(
        self,
        question: str,
        resolved_metrics: List[MetricDefinition],
        query_results: List[MetricResult],
        user_context: UserContext,
    ) -> GroundedAnswer:
        """
        Generate a natural language answer grounded in governed definitions.

        MANDATORY for all implementations:
          1. metrics_used must contain every MetricDefinition referenced in the answer
          2. definition_versions must be populated from metrics_used
          3. out_of_scope_flags must list anything the question asked that
             was not coverable from the provided metric definitions
          4. resolution_confidence must reflect how well the question mapped
             to governed metrics (not how confident the answer is)

        The answer must NOT:
          - Reference metrics not in resolved_metrics
          - Fabricate calculations not defined in the metric definitions
          - Silently ignore out-of-scope parts of the question

        This implements Responsibility 3 (Answer Grounding).
        """

    # ------------------------------------------------------------------
    # Optional — adapters may override
    # ------------------------------------------------------------------

    def explain_resolution(
        self,
        question: str,
        resolved_metrics: List[MetricDefinition],
    ) -> str:
        """
        Return a human-readable explanation of why these metrics
        were resolved for this question.

        Used for debugging and transparency tooling.
        Default implementation returns a simple summary.
        """
        if not resolved_metrics:
            return f"No governed metrics matched the question: '{question}'"
        names = ", ".join(m.name for m in resolved_metrics)
        return f"Question '{question}' resolved to governed metrics: {names}"
