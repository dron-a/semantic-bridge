"""
dbt Core YAML Connector
========================
Parses metric and semantic model definitions from a local dbt project
into canonical MetricDefinition objects.

No credentials required. No API calls. Fully local.
Works with dbt Core >= 1.6 (MetricFlow-based metric definitions).

Limitations vs dbt Cloud connector:
  - Cannot execute metric queries (no data platform connection)
  - definition_version uses git commit hash if available, else file mtime
  - Does not support saved queries (dbt Cloud only)
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

# Add project root to path for protocol imports
import sys
sys.path.insert(0, str(Path(__file__).parents[2]))

from protocol import (
    AccessLevel,
    Dimension,
    DimensionType,
    Filter,
    MetricDefinition,
    MetricQuery,
    MetricResult,
    MetricType,
    SemanticLayerConnector,
    TimeGrain,
    UserContext,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

METRIC_TYPE_MAP: Dict[str, MetricType] = {
    "simple": MetricType.SIMPLE,
    "ratio": MetricType.RATIO,
    "cumulative": MetricType.CUMULATIVE,
    "derived": MetricType.DERIVED,
}

DIMENSION_TYPE_MAP: Dict[str, DimensionType] = {
    "categorical": DimensionType.CATEGORICAL,
    "time": DimensionType.TIME,
}

# Standard MetricFlow time grains in ascending order
DEFAULT_TIME_GRAINS = ["day", "week", "month", "quarter", "year"]


def _get_git_commit(project_path: Path) -> Optional[str]:
    """Return short git commit hash for the project, or None if not a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=project_path,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def _file_version(file_path: Path) -> str:
    """Fallback version: hash of file contents."""
    content = file_path.read_bytes()
    return hashlib.sha1(content).hexdigest()[:8]


def _load_yaml_files(directory: Path) -> List[Dict[str, Any]]:
    """Recursively load all YAML files from a directory."""
    results = []
    if not directory.exists():
        return results
    for path in sorted(directory.rglob("*.yml")):
        try:
            with open(path) as f:
                data = yaml.safe_load(f)
                if data:
                    results.append({"path": path, "data": data})
        except yaml.YAMLError:
            pass  # skip malformed files silently
    return results


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------

class DbtCoreConnector(SemanticLayerConnector):
    """
    Reference connector for local dbt Core projects.

    Reads metric definitions from:
      - metrics: YAML files containing `metrics:` blocks
      - semantic_models: YAML files containing `semantic_models:` blocks

    Searched recursively from project_path across all subdirectories.
    """

    def __init__(self, project_path: str):
        self._project_path = Path(project_path).resolve()
        self._version = _get_git_commit(self._project_path) or "local"
        self._metrics_cache: Optional[Dict[str, MetricDefinition]] = None
        self._semantic_models_cache: Optional[Dict[str, Any]] = None

    @property
    def connector_id(self) -> str:
        return "dbt_core"

    def get_definition_version(self) -> str:
        return self._version

    def health_check(self) -> bool:
        """Verify project path exists and contains a dbt_project.yml."""
        dbt_project = self._project_path / "dbt_project.yml"
        return dbt_project.exists()

    # ------------------------------------------------------------------
    # Internal parsing
    # ------------------------------------------------------------------

    def _load_semantic_models(self) -> Dict[str, Any]:
        """
        Parse all semantic_models from YAML files.
        Returns dict keyed by semantic model name.
        """
        if self._semantic_models_cache is not None:
            return self._semantic_models_cache

        models: Dict[str, Any] = {}
        for file_info in _load_yaml_files(self._project_path):
            data = file_info["data"]
            for sm in data.get("semantic_models", []):
                if "name" in sm:
                    sm["_source_file"] = str(file_info["path"])
                    models[sm["name"]] = sm

        self._semantic_models_cache = models
        return models

    def _parse_dimensions_from_semantic_models(
        self, metric_raw: Dict[str, Any], semantic_models: Dict[str, Any]
    ) -> List[Dimension]:
        """
        Resolve dimensions available for a metric by looking up
        the semantic models that provide its measures.
        """
        dimensions: List[Dimension] = []
        seen: set = set()

        # Find which semantic models are relevant to this metric
        measure_names = self._extract_measure_names(metric_raw)

        for sm in semantic_models.values():
            sm_measures = {m["name"] for m in sm.get("measures", [])}
            if not sm_measures.intersection(measure_names):
                continue

            for dim_raw in sm.get("dimensions", []):
                name = dim_raw.get("name")
                if not name or name in seen:
                    continue
                seen.add(name)
                dim_type_str = dim_raw.get("type", "categorical").lower()
                dim_type = DIMENSION_TYPE_MAP.get(dim_type_str, DimensionType.UNKNOWN)
                dimensions.append(
                    Dimension(
                        name=name,
                        type=dim_type,
                        description=dim_raw.get("description"),
                        is_partition=dim_raw.get("is_partition", False),
                        expr=dim_raw.get("expr"),
                    )
                )

        # metric_time is always available as a standard MetricFlow dimension
        if "metric_time" not in seen:
            dimensions.append(
                Dimension(
                    name="metric_time",
                    type=DimensionType.TIME,
                    description="Standard MetricFlow time dimension for metric aggregation",
                    is_partition=True,
                )
            )

        return dimensions

    def _extract_measure_names(self, metric_raw: Dict[str, Any]) -> set:
        """Extract all measure names referenced by a metric definition."""
        measures = set()
        type_params = metric_raw.get("type_params", {})

        # simple metric
        measure = type_params.get("measure")
        if isinstance(measure, dict):
            if "name" in measure:
                measures.add(measure["name"])
        elif isinstance(measure, str):
            measures.add(measure)

        # ratio metric
        for key in ("numerator", "denominator"):
            part = type_params.get(key, {})
            if isinstance(part, dict) and "name" in part:
                measures.add(part["name"])

        # derived metric
        for m in type_params.get("metrics", []):
            if isinstance(m, dict) and "name" in m:
                measures.add(m["name"])

        return measures

    def _parse_time_grains(self, metric_raw: Dict[str, Any]) -> List[TimeGrain]:
        """
        Return standard MetricFlow time grains.
        MetricFlow supports all standard grains for all metrics by default.
        """
        return [
            TimeGrain(name=g, is_default=(g == "month"))
            for g in DEFAULT_TIME_GRAINS
        ]

    def _parse_filters(self, metric_raw: Dict[str, Any]) -> List[Filter]:
        """Extract filter expressions from metric definition."""
        filters = []
        raw_filter = metric_raw.get("filter")
        if raw_filter:
            filters.append(
                Filter(
                    name="default_filter",
                    description="Pre-defined filter on this metric",
                    expr=str(raw_filter).strip(),
                )
            )
        return filters

    def _metric_raw_to_definition(
        self,
        metric_raw: Dict[str, Any],
        source_file: str,
        semantic_models: Dict[str, Any],
    ) -> MetricDefinition:
        """Convert a raw YAML metric dict to a canonical MetricDefinition."""
        name = metric_raw["name"]
        metric_type = METRIC_TYPE_MAP.get(
            metric_raw.get("type", "").lower(), MetricType.UNKNOWN
        )

        return MetricDefinition(
            id=f"dbt_core::{name}",
            name=name,
            description=metric_raw.get("description") or metric_raw.get("label"),
            type=metric_type,
            measures=list(self._extract_measure_names(metric_raw)),
            filters=self._parse_filters(metric_raw),
            dimensions=self._parse_dimensions_from_semantic_models(
                metric_raw, semantic_models
            ),
            time_grains=self._parse_time_grains(metric_raw),
            requires_metric_time=True,  # MetricFlow metrics require time dimension
            source_connector=self.connector_id,
            definition_version=self._version,
            source_file=source_file,
            access_level=AccessLevel.PUBLIC,
        )

    def _load_all_metrics(self) -> Dict[str, MetricDefinition]:
        """Parse all metric definitions from the project. Cached after first load."""
        if self._metrics_cache is not None:
            return self._metrics_cache

        semantic_models = self._load_semantic_models()
        metrics: Dict[str, MetricDefinition] = {}

        for file_info in _load_yaml_files(self._project_path):
            data = file_info["data"]
            for metric_raw in data.get("metrics", []):
                if "name" not in metric_raw:
                    continue
                try:
                    definition = self._metric_raw_to_definition(
                        metric_raw,
                        source_file=str(file_info["path"]),
                        semantic_models=semantic_models,
                    )
                    metrics[definition.name] = definition
                except Exception:
                    pass  # skip malformed metrics silently

        self._metrics_cache = metrics
        return metrics

    # ------------------------------------------------------------------
    # SemanticLayerConnector interface
    # ------------------------------------------------------------------

    def list_metrics(self, user_context: UserContext) -> List[MetricDefinition]:
        return list(self._load_all_metrics().values())

    def get_metric(
        self, metric_name: str, user_context: UserContext
    ) -> Optional[MetricDefinition]:
        return self._load_all_metrics().get(metric_name)

    def get_dimensions(
        self, metric_name: str, user_context: UserContext
    ) -> List[Dimension]:
        metric = self.get_metric(metric_name, user_context)
        return metric.dimensions if metric else []

    def get_time_grains(
        self, metric_name: str, user_context: UserContext
    ) -> List[TimeGrain]:
        metric = self.get_metric(metric_name, user_context)
        return metric.time_grains if metric else []

    def query(self, query: MetricQuery, user_context: UserContext) -> MetricResult:
        """
        dbt Core connector cannot execute queries — no data platform connection.
        This is a known limitation documented in the connector.

        Use dbt Cloud connector for live query execution.
        """
        raise NotImplementedError(
            "DbtCoreConnector does not support query execution. "
            "It provides metric definitions only. "
            "Use DbtCloudConnector for live metric queries."
        )

    def invalidate_cache(self) -> None:
        """Force reload of metric definitions on next access. Useful in dev."""
        self._metrics_cache = None
        self._semantic_models_cache = None
