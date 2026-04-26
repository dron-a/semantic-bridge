# folder structure

---
semantic-bridge/
│
├── protocol/                    # The contracts — pure Python ABCs
│   ├── connector.py             # SemanticLayerConnector ABC
│   ├── llm_adapter.py           # LLMAdapter ABC
│   ├── access_control.py        # AccessController ABC
│   └── models.py                # MetricDefinition, GroundedAnswer etc.
│
├── connectors/                  # Reference implementations
│   ├── dbt_core/
│   ├── dbt_cloud/
│   └── cube/
│
├── adapters/                    # LLM reference implementations
│   ├── claude/
│   └── openai/
│
├── access/                      # Access control implementations
│   ├── passthrough.py
│   └── static_yaml.py
│
├── bridge/                      # Core orchestration
│   └── engine.py                # Wires protocol together
│
├── cli.py                       # Reference CLI
├── examples/                    # Sample dbt project + demo scripts
└── docs/                        # Protocol spec documentation