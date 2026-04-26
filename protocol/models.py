"""
semantic-bridge protocol models
================================
Canonical data shapes for the semantic bridge protocol.
Every connector serializes TO these models.
Every LLM adapter receives FROM these models.
Nothing upstream or downstream uses connector-native formats directly.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class MetricType(str, Enum):
    """Supported metric calculation types across all semantic layers."""
    SIMPLE = "simple"
    RATIO = "ratio"
    CUMULATIVE = "cumulative"
    DERIVED = "derived"
    UNKNOWN = "unknown"  # for connectors that don't expose type


class DimensionType(str, Enum):
    CATEGORICAL = "categorical"
    TIME = "time"
    ENTITY = "entity"
    UNKNOWN = "unknown"


class AccessLevel(str, Enum):
    """Coarse access classification before per-user evaluation."""
    PUBLIC = "public"        # any authenticated user
    RESTRICTED = "restricted"  # specific roles only
    CONFIDENTIAL = "confidential"  # explicit allowlist


class AccessVerdict(str, Enum):
    ALLOWED = "allowed"
    DENIED = "denied"
    PARTIAL = "partial"  # allowed but with masked dimensions


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class TimeGrain(BaseModel):
    """A valid time granularity for a metric."""
    name: str = Field(..., description="e.g. 'day', 'week', 'month', 'quarter', 'year'")
    is_default: bool = False


class Dimension(BaseModel):
    """A dimension available for slicing a metric."""
    name: str
    type: DimensionType = DimensionType.UNKNOWN
    description: Optional[str] = None
    is_partition: bool = False  # time partition dimensions
    expr: Optional[str] = None  # underlying SQL expression if exposed


class Filter(BaseModel):
    """A pre-defined filter that can be applied to a metric query."""
    name: str
    description: Optional[str] = None
    expr: str  # filter expression as defined in the semantic layer


class UserContext(BaseModel):
    """
    Who is asking. Passed through the entire bridge pipeline.
    Access control decisions are made against this.
    """
    user_id: str
    roles: List[str] = Field(default_factory=list)
    attributes: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary k/v for attribute-based access control (e.g. region, team)"
    )


# ---------------------------------------------------------------------------
# Core protocol model — MetricDefinition
# ---------------------------------------------------------------------------

class MetricDefinition(BaseModel):
    """
    The canonical representation of a governed metric.

    This is the central object of the semantic bridge protocol.
    Every connector must produce this shape.
    Every LLM adapter must consume this shape.
    No connector-native format leaks past this boundary.
    """

    # Identity
    id: str = Field(..., description="Stable unique identifier within this connector's namespace")
    name: str = Field(..., description="Human-readable metric name")
    description: Optional[str] = Field(None, description="Business definition of the metric")

    # Calculation
    type: MetricType = MetricType.UNKNOWN
    measures: List[str] = Field(
        default_factory=list,
        description="Underlying measure names that compose this metric"
    )
    filters: List[Filter] = Field(default_factory=list)

    # Queryability
    dimensions: List[Dimension] = Field(default_factory=list)
    time_grains: List[TimeGrain] = Field(default_factory=list)
    requires_metric_time: bool = Field(
        False,
        description="True if this metric must be grouped by a time dimension to be meaningful"
    )

    # Provenance — critical for answer grounding (Responsibility 3)
    source_connector: str = Field(
        ...,
        description="Which connector produced this definition e.g. 'dbt_core', 'dbt_cloud', 'cube'"
    )
    definition_version: str = Field(
        ...,
        description="Version identifier: git commit hash, environment ID+timestamp, or similar"
    )
    last_updated: Optional[datetime] = None
    source_file: Optional[str] = Field(
        None,
        description="File path or URL where this metric is defined (for auditability)"
    )

    # Access control
    access_level: AccessLevel = AccessLevel.PUBLIC
    owner: Optional[str] = None  # team or individual responsible for this metric

    class Config:
        use_enum_values = True


# ---------------------------------------------------------------------------
# Access control models — Responsibility 2
# ---------------------------------------------------------------------------

class AccessDecision(BaseModel):
    """
    Result of an access control evaluation for a single metric + user.

    PARTIAL means the metric is visible but some dimensions are stripped.
    The bridge engine enforces that masked_dimensions are never sent to the LLM.
    """
    verdict: AccessVerdict
    metric_name: str
    user_id: str
    reason: Optional[str] = None
    masked_dimensions: List[str] = Field(
        default_factory=list,
        description="Dimension names stripped from the MetricDefinition for this user"
    )
    evaluated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Query + result models
# ---------------------------------------------------------------------------

class MetricQuery(BaseModel):
    """
    A governed metric query produced by the bridge after resolution and access checks.
    This is what gets executed against the semantic layer.
    """
    metric_names: List[str]
    group_by: List[str] = Field(default_factory=list)
    filters: Dict[str, Any] = Field(default_factory=dict)
    time_grain: Optional[str] = None
    time_range_start: Optional[str] = None
    time_range_end: Optional[str] = None
    limit: Optional[int] = None


class MetricResult(BaseModel):
    """Raw result from executing a MetricQuery against the semantic layer."""
    query: MetricQuery
    data: List[Dict[str, Any]]
    row_count: int
    executed_sql: Optional[str] = Field(
        None,
        description="The SQL generated and executed, if the connector exposes it"
    )
    executed_at: datetime = Field(default_factory=datetime.utcnow)
    connector: str


# ---------------------------------------------------------------------------
# Grounded answer model — Responsibility 3
# ---------------------------------------------------------------------------

class GroundedAnswer(BaseModel):
    """
    The final output of the semantic bridge.

    Every answer must carry:
    - The natural language response
    - Which governed metric definitions were used
    - The exact versions of those definitions (for auditability)
    - Explicit flags for anything the question asked that fell outside governed definitions
    """

    # The answer itself
    answer: str

    # Provenance — every metric used is cited by its full definition
    metrics_used: List[MetricDefinition] = Field(
        default_factory=list,
        description="Full MetricDefinition objects used to ground this answer"
    )
    definition_versions: List[str] = Field(
        default_factory=list,
        description="Extracted version identifiers for quick audit reference"
    )

    # Raw data if a query was executed
    query_results: Optional[List[MetricResult]] = None

    # Out-of-scope flagging — never silently hallucinate
    out_of_scope_flags: List[str] = Field(
        default_factory=list,
        description=(
            "Questions or metric requests that could not be answered from governed definitions. "
            "The bridge never allows the LLM to silently answer these — they must be flagged."
        )
    )

    # Confidence of metric resolution (0.0 - 1.0)
    resolution_confidence: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="How confidently the bridge mapped the question to governed metrics"
    )

    # Audit trail
    user_id: str
    answered_at: datetime = Field(default_factory=datetime.utcnow)
    adapter_used: str = Field(..., description="Which LLM adapter produced this answer")

    def has_out_of_scope(self) -> bool:
        return len(self.out_of_scope_flags) > 0

    def citation_summary(self) -> str:
        """Human-readable provenance summary for display."""
        if not self.metrics_used:
            return "No governed metrics used."
        lines = ["Governed definitions used:"]
        for m in self.metrics_used:
            lines.append(f"  - {m.name} (v: {m.definition_version}, source: {m.source_connector})")
        if self.out_of_scope_flags:
            lines.append("Out of governed scope:")
            for flag in self.out_of_scope_flags:
                lines.append(f"  - {flag}")
        return "\n".join(lines)
