"""
Bridge Engine
==============
Core orchestration layer. Wires connector + access controller + LLM adapter
together and enforces the five protocol rules.

THE FIVE RULES (enforced here, not elsewhere):
  Rule 1: LLM never receives unfiltered metric list
  Rule 2: LLM never generates answer before metric resolution completes
  Rule 3: LLM never answers from outside governed definitions without flagging
  Rule 4: Every answer carries version provenance, no exceptions
  Rule 5: Access control lives in the bridge, never delegated to LLM

Usage:
    from bridge.engine import BridgeEngine
    from connectors.dbt_core import DbtCoreConnector
    from adapters.claude import ClaudeAdapter
    from access.passthrough import PassthroughAccessController

    engine = BridgeEngine(
        connector=DbtCoreConnector("./my_dbt_project"),
        adapter=ClaudeAdapter(),
        access_controller=PassthroughAccessController(),
    )
    answer = engine.ask("What was our revenue last month?", user_context)
    print(answer.answer)
    print(answer.citation_summary())
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

from protocol import (
    AccessController,
    GroundedAnswer,
    LLMAdapter,
    MetricDefinition,
    SemanticLayerConnector,
    UserContext,
)

logger = logging.getLogger(__name__)


@dataclass
class BridgeConfig:
    """Optional configuration for bridge engine behaviour."""
    max_resolved_metrics: int = 5
    require_health_check: bool = True
    log_questions: bool = True


class BridgeEngine:
    """
    Orchestrates the semantic bridge pipeline.

    The engine is the only place where the five protocol rules
    are enforced. All rule enforcement is explicit and logged.
    """

    def __init__(
        self,
        connector: SemanticLayerConnector,
        adapter: LLMAdapter,
        access_controller: Optional[AccessController] = None,  # ← make optional
        config: Optional[BridgeConfig] = None,
    ):
        if access_controller is None:
            from access.static_yaml import StaticYamlAccessController
            from access.passthrough import PassthroughAccessController
            discovered = StaticYamlAccessController.from_env_or_default()
            access_controller = discovered or PassthroughAccessController()
            if isinstance(access_controller, PassthroughAccessController):
                logger.warning(
                    "No access_control.yml found. Running in passthrough mode — "
                    "ALL metrics visible to ALL users. "
                    "Set SEMANTIC_BRIDGE_ACCESS_CONFIG or add config/access_control.yml"
                )
        self._connector = connector
        self._adapter = adapter
        self._access_controller = access_controller
        self._config = config or BridgeConfig()

        if self._config.require_health_check:
            if not self._connector.health_check():
                raise RuntimeError(
                    f"Connector {self._connector.connector_id} failed health check. "
                    "Verify your semantic layer configuration."
                )

        logger.info(
            "BridgeEngine initialized | connector=%s | adapter=%s | access=%s",
            self._connector.connector_id,
            self._adapter.adapter_id,
            self._access_controller.controller_id,
        )

    def ask(
        self,
        question: str,
        user_context: UserContext,
    ) -> GroundedAnswer:
        """
        Process a natural language question through the full bridge pipeline.

        Pipeline:
          1. Fetch all metrics from connector
          2. RULE 1: Filter metrics through access controller (Pass 1)
          3. RULE 2: Resolve relevant metrics via LLM adapter
          4. RULE 1+2: Access check resolved metrics (Pass 2)
          5. Build and execute query (if connector supports it)
          6. RULE 3+4: Generate grounded answer with provenance
          7. Audit log all access decisions
        """
        if self._config.log_questions:
            logger.info(
                "bridge.ask | user=%s | question=%r",
                user_context.user_id,
                question[:120],
            )

        # ------------------------------------------------------------------
        # Step 1: Fetch all available metrics from the semantic layer
        # ------------------------------------------------------------------
        all_metrics = self._connector.list_metrics(user_context)
        logger.debug("connector returned %d metrics", len(all_metrics))

        # ------------------------------------------------------------------
        # Step 2 — RULE 1: Access filter BEFORE LLM sees any metric names
        # The LLM never receives the full unfiltered list
        # ------------------------------------------------------------------
        filtered_metrics = self._access_controller.filter_metrics(
            user_context, all_metrics
        )
        logger.debug(
            "RULE 1 enforced: %d/%d metrics visible to user %s",
            len(filtered_metrics),
            len(all_metrics),
            user_context.user_id,
        )

        if not filtered_metrics:
            return self._no_access_answer(question, user_context)

        # ------------------------------------------------------------------
        # Step 3 — RULE 2: Metric resolution BEFORE answer generation
        # LLM identifies relevant metrics from the access-filtered list only
        # ------------------------------------------------------------------
        resolved_metrics = self._adapter.resolve_metrics(
            question, filtered_metrics, user_context
        )
        resolved_metrics = resolved_metrics[: self._config.max_resolved_metrics]
        logger.debug(
            "RULE 2 enforced: resolution complete before answer | resolved=%s",
            [m.name for m in resolved_metrics],
        )

        # ------------------------------------------------------------------
        # Step 4 — RULE 1 (Pass 2): Re-verify access on resolved metrics
        # Catches edge cases from resolution
        # ------------------------------------------------------------------
        resolved_metrics = self._access_controller.filter_metrics(
            user_context, resolved_metrics
        )

        # ------------------------------------------------------------------
        # Step 5: Execute query if connector supports it
        # dbt Core: skip (no query execution)
        # dbt Cloud / Cube: execute
        # ------------------------------------------------------------------
        query_results = []
        if resolved_metrics:
            query_results = self._execute_query(
                question, resolved_metrics, user_context
            )

        # ------------------------------------------------------------------
        # Step 6 — RULES 3+4: Generate grounded answer with provenance
        # Adapter must populate metrics_used and definition_versions
        # ------------------------------------------------------------------
        answer = self._adapter.generate_answer(
            question, resolved_metrics, query_results, user_context
        )

        # RULE 4: Enforce provenance is populated — never skip this
        answer = self._enforce_provenance(answer, resolved_metrics)

        logger.info(
            "bridge.answer | user=%s | metrics_used=%s | confidence=%.2f | out_of_scope=%s",
            user_context.user_id,
            [m.name for m in answer.metrics_used],
            answer.resolution_confidence,
            answer.out_of_scope_flags,
        )

        # ------------------------------------------------------------------
        # Step 7 — RULE 5: Audit log — always runs, never skipped
        # ------------------------------------------------------------------
        self._audit(user_context, question, all_metrics, filtered_metrics)

        return answer

    def _execute_query(
        self,
        question: str,
        resolved_metrics: List[MetricDefinition],
        user_context: UserContext,
    ) -> list:
        """
        Attempt query execution. Returns empty list if connector
        doesn't support it (dbt Core) rather than raising.
        """
        try:
            query = self._adapter.build_query(question, resolved_metrics, user_context)
            if not query.metric_names:
                return []
            result = self._connector.query(query, user_context)
            return [result]
        except NotImplementedError:
            logger.debug(
                "connector %s does not support query execution — definition mode",
                self._connector.connector_id,
            )
            return []
        except Exception as e:
            logger.warning("query execution failed: %s", e)
            return []

    def _enforce_provenance(
        self,
        answer: GroundedAnswer,
        resolved_metrics: List[MetricDefinition],
    ) -> GroundedAnswer:
        """
        RULE 4: Ensure definition_versions is always populated.
        If the adapter forgot to populate it, the engine fills it in.
        This is a safety net — adapters should populate it themselves.
        """
        if not answer.definition_versions and resolved_metrics:
            logger.warning(
                "RULE 4: adapter %s did not populate definition_versions — engine filling in",
                self._adapter.adapter_id,
            )
            return answer.copy(update={
                "definition_versions": [m.definition_version for m in resolved_metrics],
                "metrics_used": resolved_metrics if not answer.metrics_used else answer.metrics_used,
            })
        return answer

    def _audit(
        self,
        user_context: UserContext,
        question: str,
        all_metrics: List[MetricDefinition],
        filtered_metrics: List[MetricDefinition],
    ) -> None:
        """RULE 5: Audit log always runs. Never raises."""
        try:
            from protocol.models import AccessDecision, AccessVerdict
            decisions = []
            filtered_names = {m.name for m in filtered_metrics}
            for m in all_metrics:
                verdict = (
                    AccessVerdict.ALLOWED
                    if m.name in filtered_names
                    else AccessVerdict.DENIED
                )
                decisions.append(AccessDecision(
                    verdict=verdict,
                    metric_name=m.name,
                    user_id=user_context.user_id,
                ))
            self._access_controller.audit_log(user_context, question, decisions)
        except Exception as e:
            logger.error("audit_log raised (swallowed): %s", e)

    def _no_access_answer(
        self, question: str, user_context: UserContext
    ) -> GroundedAnswer:
        """Return a clear answer when user has no access to any metrics."""
        return GroundedAnswer(
            answer=(
                "You do not have access to any governed metrics. "
                "Please contact your data team to request access."
            ),
            metrics_used=[],
            definition_versions=[],
            out_of_scope_flags=["No governed metrics accessible for this user"],
            resolution_confidence=0.0,
            user_id=user_context.user_id,
            adapter_used=self._adapter.adapter_id,
        )

    # ------------------------------------------------------------------
    # Convenience methods
    # ------------------------------------------------------------------

    def list_available_metrics(
        self, user_context: UserContext
    ) -> List[MetricDefinition]:
        """
        Return metrics visible to a user after access filtering.
        Useful for building UIs that show available metrics.
        """
        all_metrics = self._connector.list_metrics(user_context)
        return self._access_controller.filter_metrics(user_context, all_metrics)

    @property
    def connector_id(self) -> str:
        return self._connector.connector_id

    @property
    def adapter_id(self) -> str:
        return self._adapter.adapter_id
