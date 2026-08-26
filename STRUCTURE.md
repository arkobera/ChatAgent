```
ChatAgent/
│
├── Backend/
│   ├── config.py
│   ├── custom_types.py
│   ├── data_loader.py
│   ├── main.py
│   ├── vector_db.py
│   │
│   ├── risk/                    ← NEW
│   │   ├── __init__.py
│   │   ├── features.py
│   │   ├── return_model.py
│   │   ├── graph_model.py
│   │   └── risk_engine.py
│   │
│   ├── graph/                   ← NEW
│   │   ├── __init__.py
│   │   ├── builder.py
│   │   └── analysis.py
│   │
│   └── evidence/                ← NEW
│       ├── __init__.py
│       └── retrieval.py
│
├── Eval/
│   ├── ...existing...
│   │
│   └── risk/                    ← NEW
│       ├── evaluate.py
│       ├── metrics.py
│       └── cost_analysis.py
│
├── Frontend/
│   ├── Authentication/
│   ├── uploads/
│   └── streamlit_app.py         ← MODIFY
│
├── tests/
│   ├── test_auth.py
│   ├── test_risk.py             ← NEW
│   └── test_graph.py            ← NEW
│
└── ...
```