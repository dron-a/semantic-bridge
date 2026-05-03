"""
Passthrough Access Controller
==============================
Allows all metrics for all users. No restrictions.

USE ONLY FOR:
  - Local development
  - Testing and CI
  - Demo environments with no sensitive metrics

NEVER use in production. Every metric is visible to every user.
"""

from __future__ import annotations

import logging
from typing import List

from protocol import AccessController, AccessDecision, AccessVerdict, MetricDefinition, UserContext

logger = logging.getLogger(__name__)


class PassthroughAccessController(AccessController):
    """
    Reference access controller that allows everything.
    Implements the full AccessController interface with no-op enforcement.
    """

    @property
    def controller_id(self) -> str:
        return "passthrough"

    def evaluate(
        self,
        user_context: UserContext,
        metric: MetricDefinition,
    ) -> AccessDecision:
        return AccessDecision(
            verdict=AccessVerdict.ALLOWED,
            metric_name=metric.name,
            user_id=user_context.user_id,
            reason="passthrough — all metrics allowed",
        )

    def audit_log(
        self,
        user_context: UserContext,
        question: str,
        decisions: List[AccessDecision],
    ) -> None:
        logger.debug(
            "passthrough audit | user=%s | question=%r | metrics=%s",
            user_context.user_id,
            question[:80],
            [d.metric_name for d in decisions],
        )
