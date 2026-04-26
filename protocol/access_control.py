"""
semantic-bridge access control protocol
=========================================
Abstract base class for access control enforcement.

CRITICAL DESIGN RULE:
  Access control is the bridge's responsibility, never the LLM's.
  The LLM adapter receives only the metrics a user is allowed to see.
  It never receives the full list and decides what to show.

Two-pass enforcement (both passes are mandatory):
  Pass 1 — filter_metrics()  : called before LLM sees any metric names
  Pass 2 — evaluate()        : called after LLM resolves specific metrics

Reference implementations: passthrough, static_yaml
Community implementations: okta, azure_ad, opa (Open Policy Agent), etc.
"""

from abc import ABC, abstractmethod
from typing import List

from .models import AccessDecision, AccessVerdict, MetricDefinition, UserContext


class AccessController(ABC):
    """
    Contract for all access control implementations.

    Implementors must enforce:
      - Metric-level visibility (can user see this metric exists?)
      - Dimension-level masking (can user slice by this dimension?)
      - Audit logging (every decision must be logged)
    """

    @property
    @abstractmethod
    def controller_id(self) -> str:
        """
        Stable identifier for this access controller.
        e.g. 'passthrough', 'static_yaml', 'opa'
        """

    @abstractmethod
    def evaluate(
        self,
        user_context: UserContext,
        metric: MetricDefinition,
    ) -> AccessDecision:
        """
        Evaluate whether a user can access a specific metric.

        Returns AccessDecision with:
          - verdict: ALLOWED, DENIED, or PARTIAL
          - masked_dimensions: dimensions to strip if PARTIAL
          - reason: human-readable explanation (for audit logs)

        PARTIAL means the metric is visible but some dimensions
        are stripped before the MetricDefinition reaches the LLM.
        """

    @abstractmethod
    def audit_log(
        self,
        user_context: UserContext,
        question: str,
        decisions: List[AccessDecision],
    ) -> None:
        """
        Log all access decisions for a single bridge query.

        Called by the bridge engine after every query regardless
        of verdict. Implementations must not raise — log failures
        should be caught internally and not block query execution.

        Minimum log entry should contain:
          - user_id, timestamp, question hash, metric names, verdicts
        """

    # ------------------------------------------------------------------
    # Concrete — not overridable, enforced by protocol
    # ------------------------------------------------------------------

    def filter_metrics(
        self,
        user_context: UserContext,
        metrics: List[MetricDefinition],
    ) -> List[MetricDefinition]:
        """
        Pass 1 enforcement: filter a list of metrics to only those
        the user is allowed to see, applying dimension masking for
        PARTIAL decisions.

        This method is intentionally NOT abstract — the filtering
        logic must be consistent across all implementations.
        Subclasses control the verdict via evaluate(), not here.
        """
        allowed = []
        for metric in metrics:
            decision = self.evaluate(user_context, metric)

            if decision.verdict == AccessVerdict.DENIED:
                # Metric is invisible to this user — not in list at all
                continue

            if decision.verdict == AccessVerdict.PARTIAL:
                # Strip masked dimensions before adding to allowed list
                metric = self._apply_dimension_masking(metric, decision.masked_dimensions)

            allowed.append(metric)
        return allowed

    def _apply_dimension_masking(
        self,
        metric: MetricDefinition,
        masked_dimension_names: List[str],
    ) -> MetricDefinition:
        """
        Return a copy of the metric with masked dimensions removed.
        The original MetricDefinition is never mutated.
        """
        if not masked_dimension_names:
            return metric

        masked_set = set(masked_dimension_names)
        filtered_dimensions = [
            d for d in metric.dimensions if d.name not in masked_set
        ]
        # Pydantic copy with override
        return metric.copy(update={"dimensions": filtered_dimensions})
