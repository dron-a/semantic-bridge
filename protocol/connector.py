"""
semantic-bridge connector protocol
====================================
Abstract base class that every semantic layer connector must implement.

Rules for connector implementors:
  1. All methods receive UserContext — never assume anonymous access
  2. All return types use canonical protocol models — never connector-native types
  3. Connectors are responsible for serializing their native format to MetricDefinition
  4. Connectors are NOT responsible for access control — that lives in the bridge engine
  5. definition_version must be populated — use git hash, timestamp, or env ID
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from .models import (
    Dimension,
    MetricDefinition,
    MetricQuery,
    MetricResult,
    TimeGrain,
    UserContext,
)


class SemanticLayerConnector(ABC):
    """
    Contract for all semantic layer connectors.

    Reference implementations: dbt_core, dbt_cloud, cube
    Community implementations: anything else

    Every method receives UserContext so connectors that have
    native access control (e.g. dbt Cloud service tokens scoped
    to a user) can pass it through. Connectors without native
    access control simply ignore it — the bridge engine handles
    access enforcement independently.
    """

    @property
    @abstractmethod
    def connector_id(self) -> str:
        """
        Stable identifier for this connector.
        Used in MetricDefinition.source_connector and audit logs.
        e.g. 'dbt_core', 'dbt_cloud', 'cube'
        """

    @abstractmethod
    def health_check(self) -> bool:
        """
        Verify the connector can reach its underlying semantic layer.
        Called by the bridge engine on startup and before queries.
        Returns True if healthy, False otherwise.
        """

    @abstractmethod
    def list_metrics(self, user_context: UserContext) -> List[MetricDefinition]:
        """
        Return all metrics available in this semantic layer.

        Returns MetricDefinition objects with at minimum:
          - id, name, source_connector, definition_version populated
          - dimensions and time_grains populated if the layer exposes them

        UserContext is passed for connectors with native auth
        (e.g. dbt Cloud where the token is scoped to a user).
        Access filtering by the bridge engine happens AFTER this call.
        """

    @abstractmethod
    def get_metric(
        self,
        metric_name: str,
        user_context: UserContext,
    ) -> Optional[MetricDefinition]:
        """
        Return a single metric's full definition by name.
        Returns None if the metric does not exist.

        Must include full dimensions, time_grains, filters, and
        a populated definition_version for answer grounding.
        """

    @abstractmethod
    def get_dimensions(
        self,
        metric_name: str,
        user_context: UserContext,
    ) -> List[Dimension]:
        """
        Return all dimensions available for a specific metric.
        Some connectors compute this dynamically (e.g. MetricFlow join graph).
        Others read it statically from YAML.
        """

    @abstractmethod
    def get_time_grains(
        self,
        metric_name: str,
        user_context: UserContext,
    ) -> List[TimeGrain]:
        """
        Return valid time granularities for a specific metric.
        Critical for answer grounding — LLM must not suggest
        granularities the metric doesn't support.
        """

    @abstractmethod
    def query(
        self,
        query: MetricQuery,
        user_context: UserContext,
    ) -> MetricResult:
        """
        Execute a governed metric query and return results.

        The MetricQuery has already been:
          - resolved from natural language by the LLM adapter
          - access-filtered by the bridge engine

        Connector is responsible for translating MetricQuery
        to its native query format and returning MetricResult
        with executed_sql populated if the layer exposes it.
        """

    # ------------------------------------------------------------------
    # Optional — connectors may override for richer behaviour
    # ------------------------------------------------------------------

    def search_metrics(
        self,
        query: str,
        user_context: UserContext,
        limit: int = 20,
    ) -> List[MetricDefinition]:
        """
        Search metrics by name or description.

        Default implementation does a simple case-insensitive
        substring match on name and description.
        Connectors with native search (e.g. vector search on descriptions)
        should override this for better resolution quality.
        """
        query_lower = query.lower()
        all_metrics = self.list_metrics(user_context)
        matches = []
        for metric in all_metrics:
            name_match = query_lower in metric.name.lower()
            desc_match = (
                metric.description is not None
                and query_lower in metric.description.lower()
            )
            if name_match or desc_match:
                matches.append(metric)
        return matches[:limit]

    def get_definition_version(self) -> str:
        """
        Return the current version identifier for the semantic layer state.
        Used to stamp GroundedAnswer.definition_versions.

        Default returns a placeholder — connectors should override
        with a git commit hash, deployment timestamp, or env ID.
        """
        return "unknown"
