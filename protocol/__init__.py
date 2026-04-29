"""
semantic-bridge protocol
=========================
The core contracts and canonical data models of the semantic bridge.

Public API:
  Models:       MetricDefinition, GroundedAnswer, UserContext, AccessDecision, ...
  Connectors:   SemanticLayerConnector
  Adapters:     LLMAdapter
  Access:       AccessController
"""

# Semantic Metrics Protocol version
# Increment this only when protocol contracts change
# Not when semantic-bridge implementation changes
SMP_VERSION = "0.1.0"

from .models import (
    AccessDecision,
    AccessLevel,
    AccessVerdict,
    Dimension,
    DimensionType,
    Filter,
    GroundedAnswer,
    MetricDefinition,
    MetricQuery,
    MetricResult,
    MetricType,
    TimeGrain,
    UserContext,
)
from .connector import SemanticLayerConnector
from .llm_adapter import LLMAdapter
from .access_control import AccessController

__all__ = [
    # Models
    "AccessDecision",
    "AccessLevel",
    "AccessVerdict",
    "Dimension",
    "DimensionType",
    "Filter",
    "GroundedAnswer",
    "MetricDefinition",
    "MetricQuery",
    "MetricResult",
    "MetricType",
    "TimeGrain",
    "UserContext",
    # ABCs
    "SemanticLayerConnector",
    "LLMAdapter",
    "AccessController",
]
