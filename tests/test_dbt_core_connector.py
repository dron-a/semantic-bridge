"""
dbt Core connector tests.
Run with: python tests/test_dbt_core_connector.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

from connectors.dbt_core import DbtCoreConnector
from protocol.models import MetricType, DimensionType, UserContext

SAMPLE_PROJECT = Path(__file__).parents[1] / "examples" / "jaffle_shop"
USER = UserContext(user_id="test_user", roles=["analyst"])


def test_health_check():
    conn = DbtCoreConnector(str(SAMPLE_PROJECT))
    assert conn.health_check(), "health_check() failed — dbt_project.yml not found"
    print("✓ health_check() passes on valid project")


def test_list_metrics_returns_all():
    conn = DbtCoreConnector(str(SAMPLE_PROJECT))
    metrics = conn.list_metrics(USER)
    names = [m.name for m in metrics]
    assert len(metrics) == 6, f"Expected 6 metrics, got {len(metrics)}: {names}"
    assert "revenue" in names
    assert "churn_rate" in names
    assert "average_order_value" in names
    assert "gross_profit" in names
    print(f"✓ list_metrics() returned {len(metrics)} metrics: {names}")


def test_metric_types_parsed_correctly():
    conn = DbtCoreConnector(str(SAMPLE_PROJECT))
    metrics = {m.name: m for m in conn.list_metrics(USER)}

    assert metrics["revenue"].type == MetricType.SIMPLE
    assert metrics["average_order_value"].type == MetricType.RATIO
    assert metrics["gross_profit"].type == MetricType.DERIVED
    assert metrics["churn_rate"].type == MetricType.RATIO
    print("✓ Metric types parsed correctly (simple, ratio, derived)")


def test_dimensions_resolved_from_semantic_models():
    conn = DbtCoreConnector(str(SAMPLE_PROJECT))
    revenue = conn.get_metric("revenue", USER)

    assert revenue is not None
    dim_names = [d.name for d in revenue.dimensions]
    assert "region" in dim_names, f"Expected 'region' in dims: {dim_names}"
    assert "metric_time" in dim_names, f"Expected 'metric_time' in dims: {dim_names}"
    print(f"✓ Dimensions resolved for 'revenue': {dim_names}")


def test_time_dimensions_typed_correctly():
    conn = DbtCoreConnector(str(SAMPLE_PROJECT))
    revenue = conn.get_metric("revenue", USER)
    dim_map = {d.name: d for d in revenue.dimensions}

    assert dim_map["metric_time"].type == DimensionType.TIME
    assert dim_map["region"].type == DimensionType.CATEGORICAL
    print("✓ Dimension types correct (TIME, CATEGORICAL)")


def test_time_grains_present():
    conn = DbtCoreConnector(str(SAMPLE_PROJECT))
    revenue = conn.get_metric("revenue", USER)
    grains = [g.name for g in revenue.time_grains]

    assert "day" in grains
    assert "month" in grains
    assert "quarter" in grains
    print(f"✓ Time grains present: {grains}")


def test_definition_version_populated():
    conn = DbtCoreConnector(str(SAMPLE_PROJECT))
    revenue = conn.get_metric("revenue", USER)
    assert revenue.definition_version not in (None, "", "unknown"), \
        "definition_version must be populated for answer grounding"
    print(f"✓ definition_version populated: '{revenue.definition_version}'")


def test_source_connector_is_dbt_core():
    conn = DbtCoreConnector(str(SAMPLE_PROJECT))
    metrics = conn.list_metrics(USER)
    for m in metrics:
        assert m.source_connector == "dbt_core"
    print("✓ source_connector = 'dbt_core' on all metrics")


def test_get_metric_returns_none_for_unknown():
    conn = DbtCoreConnector(str(SAMPLE_PROJECT))
    result = conn.get_metric("nonexistent_metric", USER)
    assert result is None
    print("✓ get_metric() returns None for unknown metric name")


def test_source_file_populated():
    conn = DbtCoreConnector(str(SAMPLE_PROJECT))
    revenue = conn.get_metric("revenue", USER)
    assert revenue.source_file is not None
    assert "metrics.yml" in revenue.source_file
    print(f"✓ source_file populated: '{revenue.source_file}'")


def test_filter_parsed_for_revenue():
    conn = DbtCoreConnector(str(SAMPLE_PROJECT))
    revenue = conn.get_metric("revenue", USER)
    assert len(revenue.filters) > 0, "Expected filter on revenue metric"
    print(f"✓ Filter parsed for revenue: '{revenue.filters[0].expr[:50]}...'")


def test_cache_invalidation():
    conn = DbtCoreConnector(str(SAMPLE_PROJECT))
    conn.list_metrics(USER)  # populate cache
    conn.invalidate_cache()
    assert conn._metrics_cache is None
    metrics = conn.list_metrics(USER)  # reload
    assert len(metrics) == 6
    print("✓ Cache invalidation and reload works correctly")


if __name__ == "__main__":
    tests = [
        test_health_check,
        test_list_metrics_returns_all,
        test_metric_types_parsed_correctly,
        test_dimensions_resolved_from_semantic_models,
        test_time_dimensions_typed_correctly,
        test_time_grains_present,
        test_definition_version_populated,
        test_source_connector_is_dbt_core,
        test_get_metric_returns_none_for_unknown,
        test_source_file_populated,
        test_filter_parsed_for_revenue,
        test_cache_invalidation,
    ]

    print("Running dbt Core connector tests...\n")
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"✗ {t.__name__}: {e}")
        except Exception as e:
            print(f"✗ {t.__name__} ERROR: {type(e).__name__}: {e}")

    print(f"\n{passed}/{len(tests)} tests passed")
    if passed < len(tests):
        sys.exit(1)
