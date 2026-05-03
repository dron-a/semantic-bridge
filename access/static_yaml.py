"""
Static YAML Access Controller
================================
Enforces access control rules defined in a YAML configuration file.

This is the recommended controller for teams that want real governance
without integrating with an enterprise IAM system.

Config file format:

  version: 1

  # Default policy when no rule matches
  default_policy: allow   # or deny

  rules:
    - metric: revenue
      access_level: restricted
      allowed_roles: [finance, admin]
      masked_dimensions:
        - role: analyst
          dimensions: [internal_cost_center]

    - metric: churn_rate
      access_level: restricted
      allowed_roles: [cs_team, analyst, admin]

    - metric: gross_profit
      access_level: confidential
      allowed_users: [cfo@company.com, vp_finance@company.com]

Community can extend this with OPA, Okta, Azure AD connectors
that implement the same AccessController ABC.
"""

from __future__ import annotations

import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from protocol import (
    AccessController,
    AccessDecision,
    AccessLevel,
    AccessVerdict,
    MetricDefinition,
    UserContext,
)

logger = logging.getLogger(__name__)


class StaticYamlAccessController(AccessController):
    """
    Access controller backed by a YAML rules file.

    Rules are evaluated in order. First matching rule wins.
    If no rule matches, default_policy applies (default: allow).
    """

    def __init__(self, config_path: str):
        self._config_path = Path(config_path)
        self._config: Dict[str, Any] = self._load_config()

    @property
    def controller_id(self) -> str:
        return "static_yaml"
    
    @classmethod
    def from_env_or_default(cls, default_path: str = "config/access_control.yml"):
        """
        Resolve config path in priority order:
        1. SEMANTIC_BRIDGE_ACCESS_CONFIG env var
        2. default_path argument
        3. Returns None if neither exists (caller falls back to passthrough)
        """
        env_path = os.environ.get("SEMANTIC_BRIDGE_ACCESS_CONFIG")
        if env_path and Path(env_path).exists():
            return cls(env_path)
        if Path(default_path).exists():
            return cls(default_path)
        return None

    def _load_config(self) -> Dict[str, Any]:
        if not self._config_path.exists():
            raise FileNotFoundError(
                f"Access control config not found: {self._config_path}"
            )
        with open(self._config_path) as f:
            return yaml.safe_load(f) or {}

    def reload(self) -> None:
        """Reload config from disk — useful for hot-reloading in dev."""
        self._config = self._load_config()

    def _find_rule(self, metric_name: str) -> Optional[Dict[str, Any]]:
        """Find the first matching rule for a metric name."""
        for rule in self._config.get("rules", []):
            if rule.get("metric") == metric_name:
                return rule
        return None

    def _default_verdict(self) -> AccessVerdict:
        policy = self._config.get("default_policy", "allow").lower()
        return AccessVerdict.ALLOWED if policy == "allow" else AccessVerdict.DENIED

    def evaluate(
        self,
        user_context: UserContext,
        metric: MetricDefinition,
    ) -> AccessDecision:
        rule = self._find_rule(metric.name)

        # No rule found — apply default policy
        if rule is None:
            verdict = self._default_verdict()
            return AccessDecision(
                verdict=verdict,
                metric_name=metric.name,
                user_id=user_context.user_id,
                reason=f"no rule matched — default policy: {verdict}",
            )

        # Check allowed_users (explicit allowlist — highest priority)
        allowed_users = rule.get("allowed_users", [])
        if allowed_users:
            if user_context.user_id in allowed_users:
                return AccessDecision(
                    verdict=AccessVerdict.ALLOWED,
                    metric_name=metric.name,
                    user_id=user_context.user_id,
                    reason="user in allowed_users allowlist",
                )
            else:
                return AccessDecision(
                    verdict=AccessVerdict.DENIED,
                    metric_name=metric.name,
                    user_id=user_context.user_id,
                    reason="metric has explicit allowlist — user not included",
                )

        # Check allowed_roles
        allowed_roles = rule.get("allowed_roles", [])
        if allowed_roles:
            user_roles = set(user_context.roles)
            if not user_roles.intersection(set(allowed_roles)):
                return AccessDecision(
                    verdict=AccessVerdict.DENIED,
                    metric_name=metric.name,
                    user_id=user_context.user_id,
                    reason=f"user roles {list(user_roles)} not in allowed_roles {allowed_roles}",
                )

        # User has access — check for dimension masking rules
        masked_dimensions: List[str] = []
        for mask_rule in rule.get("masked_dimensions", []):
            mask_role = mask_rule.get("role")
            if mask_role and mask_role in user_context.roles:
                masked_dimensions.extend(mask_rule.get("dimensions", []))

        if masked_dimensions:
            return AccessDecision(
                verdict=AccessVerdict.PARTIAL,
                metric_name=metric.name,
                user_id=user_context.user_id,
                masked_dimensions=masked_dimensions,
                reason=f"access granted with {len(masked_dimensions)} masked dimension(s)",
            )

        return AccessDecision(
            verdict=AccessVerdict.ALLOWED,
            metric_name=metric.name,
            user_id=user_context.user_id,
            reason="access granted — all dimensions visible",
        )

    def audit_log(
        self,
        user_context: UserContext,
        question: str,
        decisions: List[AccessDecision],
    ) -> None:
        try:
            allowed = [d.metric_name for d in decisions if d.verdict == AccessVerdict.ALLOWED]
            partial = [d.metric_name for d in decisions if d.verdict == AccessVerdict.PARTIAL]
            denied = [d.metric_name for d in decisions if d.verdict == AccessVerdict.DENIED]
            logger.info(
                "access_audit | user=%s | allowed=%s | partial=%s | denied=%s | question=%r",
                user_context.user_id,
                allowed,
                partial,
                denied,
                question[:120],
            )
        except Exception as e:
            logger.error("audit_log failed silently: %s", e)
