"""
Protocol validation tests.
Run with: python tests/test_protocol.py
"""
import sys
sys.path.insert(0, '/home/claude/semantic-bridge')

from protocol.models import (
    AccessDecision, AccessVerdict, Dimension, DimensionType,
    MetricDefinition, MetricType, TimeGrain, UserContext,
)
from protocol.access_control import AccessController


# ---------------------------------------------------------------------------
# Minimal concrete AccessController for testing the masking logic
# ---------------------------------------------------------------------------

class TestAccessController(AccessController):
    """
    Test controller: denies 'secret_metric', 
    partially allows 'revenue' (masks 'internal_cost_center'),
    allows everything else.
    """

    @property
    def controller_id(self) -> str:
        return "test"

    def evaluate(self, user_context: UserContext, metric: MetricDefinition) -> AccessDecision:
        if metric.name == "secret_metric":
            return AccessDecision(
                verdict=AccessVerdict.DENIED,
                metric_name=metric.name,
                user_id=user_context.user_id,
                reason="Confidential metric",
            )
        if metric.name == "revenue" and "analyst" in user_context.roles:
            return AccessDecision(
                verdict=AccessVerdict.PARTIAL,
                metric_name=metric.name,
                user_id=user_context.user_id,
                masked_dimensions=["internal_cost_center"],
                reason="Analysts cannot slice by internal cost center",
            )
        return AccessDecision(
            verdict=AccessVerdict.ALLOWED,
            metric_name=metric.name,
            user_id=user_context.user_id,
        )

    def audit_log(self, user_context, question, decisions):
        pass  # no-op for tests


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def make_revenue_metric():
    return MetricDefinition(
        id="rev_001",
        name="revenue",
        description="Total recognized revenue",
        type=MetricType.SIMPLE,
        dimensions=[
            Dimension(name="region", type=DimensionType.CATEGORICAL),
            Dimension(name="internal_cost_center", type=DimensionType.CATEGORICAL),
            Dimension(name="metric_time", type=DimensionType.TIME),
        ],
        time_grains=[TimeGrain(name="month", is_default=True)],
        source_connector="dbt_core",
        definition_version="abc1234",
    )


def make_secret_metric():
    return MetricDefinition(
        id="sec_001",
        name="secret_metric",
        description="Confidential",
        type=MetricType.SIMPLE,
        dimensions=[],
        time_grains=[],
        source_connector="dbt_core",
        definition_version="abc1234",
    )


def make_churn_metric():
    return MetricDefinition(
        id="churn_001",
        name="churn_rate",
        description="Monthly customer churn rate",
        type=MetricType.RATIO,
        dimensions=[Dimension(name="region", type=DimensionType.CATEGORICAL)],
        time_grains=[TimeGrain(name="month", is_default=True)],
        source_connector="dbt_core",
        definition_version="abc1234",
    )


def test_denied_metric_invisible():
    ctrl = TestAccessController()
    user = UserContext(user_id="u1", roles=["analyst"])
    metrics = [make_revenue_metric(), make_secret_metric(), make_churn_metric()]
    filtered = ctrl.filter_metrics(user, metrics)
    names = [m.name for m in filtered]
    assert "secret_metric" not in names, "DENIED metric must be invisible"
    assert "revenue" in names
    assert "churn_rate" in names
    print("✓ DENIED metric is invisible after filter_metrics()")


def test_partial_dimension_masking():
    ctrl = TestAccessController()
    user = UserContext(user_id="u1", roles=["analyst"])
    metrics = [make_revenue_metric()]
    filtered = ctrl.filter_metrics(user, metrics)
    revenue = filtered[0]
    dim_names = [d.name for d in revenue.dimensions]
    assert "internal_cost_center" not in dim_names, "Masked dimension must be stripped"
    assert "region" in dim_names, "Non-masked dimension must remain"
    assert "metric_time" in dim_names, "Time dimension must remain"
    print("✓ Masked dimension stripped in PARTIAL access")


def test_original_metric_not_mutated():
    ctrl = TestAccessController()
    user = UserContext(user_id="u1", roles=["analyst"])
    original = make_revenue_metric()
    original_dim_count = len(original.dimensions)
    ctrl.filter_metrics(user, [original])
    assert len(original.dimensions) == original_dim_count, \
        "Original MetricDefinition must not be mutated"
    print("✓ Original MetricDefinition not mutated after masking")


def test_allowed_metric_passes_through_intact():
    ctrl = TestAccessController()
    user = UserContext(user_id="u1", roles=["admin"])  # no analyst role
    metrics = [make_revenue_metric()]
    filtered = ctrl.filter_metrics(user, metrics)
    assert len(filtered) == 1
    assert len(filtered[0].dimensions) == 3, "All dimensions intact for allowed metric"
    print("✓ ALLOWED metric passes through with all dimensions intact")


def test_citation_summary_with_out_of_scope():
    from protocol.models import GroundedAnswer
    m = make_revenue_metric()
    ga = GroundedAnswer(
        answer="Revenue was $4.2M.",
        metrics_used=[m],
        definition_versions=["abc1234"],
        out_of_scope_flags=["net_revenue_retention is not a governed metric"],
        resolution_confidence=0.7,
        user_id="u1",
        adapter_used="claude",
    )
    assert ga.has_out_of_scope()
    summary = ga.citation_summary()
    assert "revenue" in summary
    assert "net_revenue_retention" in summary
    print("✓ GroundedAnswer citation_summary includes out-of-scope flags")


if __name__ == "__main__":
    tests = [
        test_denied_metric_invisible,
        test_partial_dimension_masking,
        test_original_metric_not_mutated,
        test_allowed_metric_passes_through_intact,
        test_citation_summary_with_out_of_scope,
    ]
    print("Running protocol validation tests...\n")
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except AssertionError as e:
            print(f"✗ {t.__name__}: {e}")
        except Exception as e:
            print(f"✗ {t.__name__} ERROR: {e}")
    print(f"\n{passed}/{len(tests)} tests passed")
    if passed < len(tests):
        sys.exit(1)
