"""
Week 3 tests — access controllers and bridge engine.
Run with: python tests/test_week3.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

from protocol.models import (
    AccessVerdict, Dimension, DimensionType, GroundedAnswer,
    MetricDefinition, MetricType, TimeGrain, UserContext,
)
from access.passthrough import PassthroughAccessController
from access.static_yaml import StaticYamlAccessController
from connectors.dbt_core import DbtCoreConnector

SAMPLE_PROJECT = Path(__file__).parents[1] / "examples" / "jaffle_shop"
ACCESS_CONFIG = SAMPLE_PROJECT / "access_control.yml"


# ---------------------------------------------------------------------------
# Passthrough tests
# ---------------------------------------------------------------------------

def test_passthrough_allows_all():
    ctrl = PassthroughAccessController()
    user = UserContext(user_id="u1", roles=["intern"])

    conn = DbtCoreConnector(str(SAMPLE_PROJECT))
    all_metrics = conn.list_metrics(user)
    filtered = ctrl.filter_metrics(user, all_metrics)

    assert len(filtered) == len(all_metrics), \
        "Passthrough must allow all metrics"
    print(f"✓ Passthrough allows all {len(filtered)} metrics")


def test_passthrough_controller_id():
    ctrl = PassthroughAccessController()
    assert ctrl.controller_id == "passthrough"
    print("✓ Passthrough controller_id correct")


# ---------------------------------------------------------------------------
# Static YAML access controller tests
# ---------------------------------------------------------------------------

def test_static_yaml_loads_config():
    ctrl = StaticYamlAccessController(str(ACCESS_CONFIG))
    assert ctrl._config is not None
    print("✓ StaticYamlAccessController loads config")


def test_static_yaml_denies_confidential_to_analyst():
    ctrl = StaticYamlAccessController(str(ACCESS_CONFIG))
    conn = DbtCoreConnector(str(SAMPLE_PROJECT))
    user = UserContext(user_id="analyst@co.com", roles=["analyst"])

    all_metrics = conn.list_metrics(user)
    filtered = ctrl.filter_metrics(user, all_metrics)
    names = [m.name for m in filtered]

    assert "gross_profit" not in names, \
        "gross_profit is confidential — analyst should not see it"
    print("✓ Confidential metric denied to analyst")


def test_static_yaml_allows_confidential_to_allowlist_user():
    ctrl = StaticYamlAccessController(str(ACCESS_CONFIG))
    conn = DbtCoreConnector(str(SAMPLE_PROJECT))
    user = UserContext(user_id="cfo@jaffle.com", roles=["finance"])

    all_metrics = conn.list_metrics(user)
    filtered = ctrl.filter_metrics(user, all_metrics)
    names = [m.name for m in filtered]

    assert "gross_profit" in names, \
        "gross_profit should be visible to cfo@jaffle.com"
    print("✓ Confidential metric allowed to explicit allowlist user")


def test_static_yaml_partial_masks_dimension():
    ctrl = StaticYamlAccessController(str(ACCESS_CONFIG))
    conn = DbtCoreConnector(str(SAMPLE_PROJECT))
    user = UserContext(user_id="analyst@co.com", roles=["analyst"])

    all_metrics = conn.list_metrics(user)
    filtered = ctrl.filter_metrics(user, all_metrics)
    metric_map = {m.name: m for m in filtered}

    revenue = metric_map.get("revenue")
    assert revenue is not None, "revenue should be visible to analyst"

    dim_names = [d.name for d in revenue.dimensions]
    assert "internal_cost_center" not in dim_names, \
        "internal_cost_center should be masked for analyst"
    assert "region" in dim_names, "region should still be visible"
    print(f"✓ Analyst revenue dimensions after masking: {dim_names}")


def test_static_yaml_default_policy_allows_unmatched():
    ctrl = StaticYamlAccessController(str(ACCESS_CONFIG))
    conn = DbtCoreConnector(str(SAMPLE_PROJECT))
    # order_count has no rule — default policy is allow
    user = UserContext(user_id="anyone@co.com", roles=["marketing"])

    all_metrics = conn.list_metrics(user)
    filtered = ctrl.filter_metrics(user, all_metrics)
    names = [m.name for m in filtered]

    assert "order_count" in names, \
        "order_count has no rule — default allow policy should apply"
    print("✓ Default allow policy applies to metrics without rules")


def test_static_yaml_reload():
    ctrl = StaticYamlAccessController(str(ACCESS_CONFIG))
    original_version = ctrl._config.get("version")
    ctrl.reload()
    assert ctrl._config.get("version") == original_version
    print("✓ Config reload works correctly")


# ---------------------------------------------------------------------------
# Bridge engine tests (no LLM — tests wiring and rule enforcement)
# ---------------------------------------------------------------------------

class MockAdapter:
    """Minimal mock adapter for testing engine wiring without API calls."""

    adapter_id = "mock"

    def resolve_metrics(self, question, available_metrics, user_context):
        # Return first 2 metrics to simulate resolution
        return available_metrics[:2]

    def build_query(self, question, resolved_metrics, user_context):
        from protocol.models import MetricQuery
        return MetricQuery(metric_names=[m.name for m in resolved_metrics])

    def generate_answer(self, question, resolved_metrics, query_results, user_context):
        return GroundedAnswer(
            answer=f"Mock answer for: {question}",
            metrics_used=resolved_metrics,
            definition_versions=[m.definition_version for m in resolved_metrics],
            resolution_confidence=0.9,
            user_id=user_context.user_id,
            adapter_used="mock",
        )


def test_engine_initializes():
    from bridge.engine import BridgeEngine, BridgeConfig
    engine = BridgeEngine(
        connector=DbtCoreConnector(str(SAMPLE_PROJECT)),
        adapter=MockAdapter(),
        access_controller=PassthroughAccessController(),
        config=BridgeConfig(require_health_check=True),
    )
    assert engine.connector_id == "dbt_core"
    assert engine.adapter_id == "mock"
    print("✓ BridgeEngine initializes correctly")


def test_engine_ask_returns_grounded_answer():
    from bridge.engine import BridgeEngine, BridgeConfig
    engine = BridgeEngine(
        connector=DbtCoreConnector(str(SAMPLE_PROJECT)),
        adapter=MockAdapter(),
        access_controller=PassthroughAccessController(),
        config=BridgeConfig(require_health_check=True, log_questions=False),
    )
    user = UserContext(user_id="u1", roles=["analyst"])
    answer = engine.ask("What is our revenue?", user)

    assert isinstance(answer, GroundedAnswer)
    assert answer.user_id == "u1"
    assert answer.adapter_used == "mock"
    assert len(answer.metrics_used) > 0
    print("✓ Engine.ask() returns GroundedAnswer")


def test_engine_rule1_llm_never_gets_unfiltered_list():
    """
    Rule 1: LLM (adapter) must never receive metrics denied by access control.
    We verify by using a restrictive controller and checking what the mock adapter saw.
    """
    from bridge.engine import BridgeEngine, BridgeConfig

    seen_metrics = []

    class SpyAdapter(MockAdapter):
        def resolve_metrics(self, question, available_metrics, user_context):
            seen_metrics.extend(available_metrics)
            return available_metrics[:1]

    engine = BridgeEngine(
        connector=DbtCoreConnector(str(SAMPLE_PROJECT)),
        adapter=SpyAdapter(),
        access_controller=StaticYamlAccessController(str(ACCESS_CONFIG)),
        config=BridgeConfig(require_health_check=True, log_questions=False),
    )
    user = UserContext(user_id="analyst@co.com", roles=["analyst"])
    engine.ask("What is our gross profit?", user)

    seen_names = [m.name for m in seen_metrics]
    assert "gross_profit" not in seen_names, \
        "RULE 1 VIOLATED: LLM adapter received confidential metric"
    print(f"✓ RULE 1: LLM never received denied metric. Saw: {seen_names}")


def test_engine_rule4_provenance_always_populated():
    """Rule 4: definition_versions must always be populated."""
    from bridge.engine import BridgeEngine, BridgeConfig

    class ForgetfulAdapter(MockAdapter):
        """Simulates an adapter that forgets to populate provenance."""
        def generate_answer(self, question, resolved_metrics, query_results, user_context):
            return GroundedAnswer(
                answer="Answer without provenance",
                metrics_used=[],        # forgot
                definition_versions=[], # forgot
                resolution_confidence=0.5,
                user_id=user_context.user_id,
                adapter_used="forgetful",
            )

    engine = BridgeEngine(
        connector=DbtCoreConnector(str(SAMPLE_PROJECT)),
        adapter=ForgetfulAdapter(),
        access_controller=PassthroughAccessController(),
        config=BridgeConfig(require_health_check=True, log_questions=False),
    )
    user = UserContext(user_id="u1", roles=["analyst"])
    answer = engine.ask("What is revenue?", user)

    assert len(answer.definition_versions) > 0, \
        "RULE 4 VIOLATED: engine did not fill in definition_versions"
    print(f"✓ RULE 4: Engine enforced provenance: {answer.definition_versions}")


def test_engine_list_available_metrics_respects_access():
    from bridge.engine import BridgeEngine, BridgeConfig
    engine = BridgeEngine(
        connector=DbtCoreConnector(str(SAMPLE_PROJECT)),
        adapter=MockAdapter(),
        access_controller=StaticYamlAccessController(str(ACCESS_CONFIG)),
        config=BridgeConfig(require_health_check=True, log_questions=False),
    )
    user = UserContext(user_id="analyst@co.com", roles=["analyst"])
    visible = engine.list_available_metrics(user)
    names = [m.name for m in visible]

    assert "gross_profit" not in names
    assert "revenue" in names
    print(f"✓ list_available_metrics respects access control: {names}")


if __name__ == "__main__":
    tests = [
        test_passthrough_allows_all,
        test_passthrough_controller_id,
        test_static_yaml_loads_config,
        test_static_yaml_denies_confidential_to_analyst,
        test_static_yaml_allows_confidential_to_allowlist_user,
        test_static_yaml_partial_masks_dimension,
        test_static_yaml_default_policy_allows_unmatched,
        test_static_yaml_reload,
        test_engine_initializes,
        test_engine_ask_returns_grounded_answer,
        test_engine_rule1_llm_never_gets_unfiltered_list,
        test_engine_rule4_provenance_always_populated,
        test_engine_list_available_metrics_respects_access,
    ]

    print("Running Week 3 tests...\n")
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
