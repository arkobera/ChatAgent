"""ReturnGuard Streamlit dashboard for evaluation, network analysis, and investigation."""

from __future__ import annotations

import html
import json
import os
import sys
from pathlib import Path

import altair as alt
import networkx as nx
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from supabase import Client, create_client

from Authentication.auth import Auth, AuthScreen


FRONTEND_DIRECTORY = Path(__file__).resolve().parent
PROJECT_ROOT = FRONTEND_DIRECTORY.parents[0]
RISK_MODULE_DIR = PROJECT_ROOT / "Backend" / "risk"
if str(FRONTEND_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(FRONTEND_DIRECTORY))
# The current engine uses sibling imports. Accommodate that in the frontend
# rather than modifying the protected model/engine implementation.
if str(RISK_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(RISK_MODULE_DIR))

from Backend.risk.graph_model import GraphRiskModel, build_graph_snapshot  # noqa: E402 
from Backend.risk.return_model import ReturnRiskModel  # noqa: E402
from Backend.risk.risk_engine import RiskEngine  # noqa: E402


load_dotenv()
st.set_page_config(
    page_title="Return Risk Management",
    page_icon=":material/shield:",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_ENV = os.getenv("APP_ENV", "LOCAL").strip().upper()
if APP_ENV not in {"LOCAL", "PROD"}:
    raise RuntimeError("APP_ENV must be either 'LOCAL' or 'PROD'.")
IS_PRODUCTION = APP_ENV == "PROD"


def service_setting(name: str, default: str | None = None) -> str | None:
    """Read the existing environment-specific setting convention."""
    return os.getenv(f"{name}_{APP_ENV}") or os.getenv(name, default)


DATA_DIRECTORY = Path(service_setting("DATA_STORAGE_PATH", str(PROJECT_ROOT / "dataHub"))) #type: ignore

supabase: Client | None = None
if IS_PRODUCTION:
    supabase = create_client(  # type: ignore[arg-type]
        service_setting("SUPABASE_URL"), service_setting("SUPABASE_KEY") #type: ignore
    )


def require_authentication() -> None:
    """Preserve the existing protected-app behavior."""
    if not st.session_state.get("user_email"):
        AuthScreen().auth_screen()
        st.stop()


def risk_badge(level: str) -> None:
    """Display the semantic risk status with native Streamlit styling."""
    st.badge(level, color={"LOW": "green", "REVIEW": "orange", "HIGH": "red"}.get(level, "gray")) #type: ignore


@st.cache_data(ttl="15m", max_entries=2)
def load_datasets() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the model-ready returns and graph-ready orders DataFrames."""
    returns_path = DATA_DIRECTORY / "transformed_data.csv"
    orders_path = DATA_DIRECTORY / "data_v2.csv"
    if not returns_path.is_file() or not orders_path.is_file():
        raise FileNotFoundError(
            f"Expected transformed_data.csv and data_v2.csv under {DATA_DIRECTORY}."
        )
    return pd.read_csv(returns_path), pd.read_csv(orders_path)


@st.cache_resource(show_spinner="Loading trained risk models...")
def load_risk_engine() -> RiskEngine:
    """Load actual local model bundles, with the existing W&B factory as fallback."""
    return_bundle = DATA_DIRECTORY / "return_risk_model.joblib"
    graph_bundle = DATA_DIRECTORY / "graph_risk_model.joblib"
    if return_bundle.is_file() and graph_bundle.is_file():
        return RiskEngine(ReturnRiskModel.load(return_bundle), GraphRiskModel.load(graph_bundle))
    return RiskEngine.from_wandb()


def frames_for_split(split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return matching windows so graph scoring does not mix dataset splits."""
    returns_df, orders_df = load_datasets()
    if split == "All data":
        return returns_df, orders_df
    return (
        returns_df.loc[returns_df["split"] == split].copy(),
        orders_df.loc[orders_df["split"] == split].copy(),
    )


@st.cache_data(ttl="10m", max_entries=8, show_spinner="Scoring returns...")
def score_split(split: str) -> tuple[pd.DataFrame, dict[str, float | int | None]]:
    """Produce actual combined decisions and evaluation metrics for one split."""
    returns_df, orders_df = frames_for_split(split)
    engine = load_risk_engine()
    return engine.assess(returns_df, orders_df), engine.evaluate(returns_df, orders_df)


@st.cache_resource(show_spinner="Building customer relationship graph...")
def load_customer_graph() -> nx.Graph:
    """Build the same actual NetworkX customer graph used by GraphRiskModel."""
    _, orders_df = load_datasets()
    graph, _ = build_graph_snapshot(orders_df)
    return graph


@st.cache_data(ttl="10m", max_entries=1, show_spinner="Computing network scores...")
def all_network_scores() -> pd.DataFrame:
    """Get actual graph-risk probabilities for all customers."""
    _, orders_df = load_datasets()
    customers = orders_df.loc[:, ["customer_id"]].drop_duplicates()
    return load_risk_engine().graph_model.predict_proba(orders_df, customers)


@st.cache_data(ttl="1h", max_entries=1)
def load_graph_json() -> dict:
    """Load the pre-built heterogeneous entity graph from JSON."""
    graph_path = DATA_DIRECTORY / "return_abuse_graph_v2.json"
    if not graph_path.is_file():
        raise FileNotFoundError(f"Graph JSON not found at {graph_path}.")
    with open(graph_path) as f:
        return json.load(f)


@st.cache_data(ttl="1h", max_entries=1)
def load_evaluation_results() -> dict:
    """Load combined RiskEngine evaluation metrics from JSON."""
    result_path = DATA_DIRECTORY / "result.json"
    if not result_path.is_file():
        raise FileNotFoundError(f"Evaluation results not found at {result_path}.")
    with open(result_path) as f:
        return json.load(f)


def query_llm(prompt: str) -> str:
    """Call Groq LLM directly with the given prompt."""
    api_key = os.getenv("GROQ_API")
    if not api_key:
        raise RuntimeError("GROQ_API environment variable is not set.")
    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1")
    response = client.chat.completions.create(
        model="openai/gpt-oss-120b",
        max_tokens=1024,
        temperature=0.3,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a return-risk analyst for an enterprise fraud detection system. "
                    "Explain the following risk assessment results in clear, concise natural language. "
                    "Highlight key risk factors, network evidence, and the recommended action. "
                    "Do not invent facts — use only the provided data."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content or "No response generated."


def render_network_svg(graph_data: dict, customer_id: str, score_map: dict[str, float], hop_count: int = 2) -> None:
    """Render the selected customer's neighborhood from the JSON entity graph as SVG."""
    node_map = {n["id"]: n for n in graph_data["nodes"]}

    if customer_id not in node_map:
        st.warning("The selected customer is not present in the graph.")
        return

    adj: dict[str, list[dict]] = {}
    for e in graph_data["edges"]:
        adj.setdefault(e["source"], []).append(e)
        adj.setdefault(e["target"], []).append(e)

    visited: dict[str, int] = {customer_id: 0}
    queue = [customer_id]
    for hop in range(hop_count):
        next_queue = []
        for nid in queue:
            for e in adj.get(nid, []):
                neighbor = e["target"] if e["source"] == nid else e["source"]
                if neighbor not in visited:
                    visited[neighbor] = hop + 1
                    next_queue.append(neighbor)
        queue = next_queue

    local_nodes = sorted(
        visited,
        key=lambda n: (n != customer_id, visited[n], -sum(e["weight"] for e in adj.get(n, [])), str(n)),
    )[:36]
    local_edges = [
        e for e in graph_data["edges"]
        if e["source"] in local_nodes and e["target"] in local_nodes
    ]

    if len(local_nodes) == 1:
        st.info("This customer has no shared-entity connections.")
        return

    import random
    random.seed(hash(customer_id) % 2**32)
    pos = {n: (random.uniform(-1, 1), random.uniform(-1, 1)) for n in local_nodes}
    for _ in range(50):
        for n in local_nodes:
            fx, fy = 0.0, 0.0
            for m in local_nodes:
                if n == m:
                    continue
                dx = pos[n][0] - pos[m][0]
                dy = pos[n][1] - pos[m][1]
                d = max((dx**2 + dy**2) ** 0.5, 0.01)
                fx += dx / d * 0.01
                fy += dy / d * 0.01
            for e in local_edges:
                if e["source"] == n:
                    other = e["target"]
                elif e["target"] == n:
                    other = e["source"]
                else:
                    continue
                if other not in pos:
                    continue
                dx = pos[n][0] - pos[other][0]
                dy = pos[n][1] - pos[other][1]
                d = max((dx**2 + dy**2) ** 0.5, 0.01)
                fx -= dx / d * 0.005 * e.get("weight", 1)
                fy -= dy / d * 0.005 * e.get("weight", 1)
            mag = max((fx**2 + fy**2) ** 0.5, 0.001)
            pos[n] = (pos[n][0] - fx / mag * 0.05, pos[n][1] - fy / mag * 0.05)

    width, height, pad = 820, 450, 40
    xs = [p[0] for p in pos.values()]
    ys = [p[1] for p in pos.values()]
    x_span = max(max(xs) - min(xs), 0.1)
    y_span = max(max(ys) - min(ys), 0.1)

    def point(node: str) -> tuple[float, float]:
        x, y = pos[node]
        return pad + (x - min(xs)) / x_span * (width - 2 * pad), pad + (y - min(ys)) / y_span * (height - 2 * pad)

    NODE_COLORS = {
        "customer": ("#b81120", "#ffdad7"),
        "device": ("#5c5e66", "#e0e2e6"),
        "ip": ("#4a90d9", "#d0e4f7"),
        "address": ("#7cb342", "#dcedc8"),
        "payment": ("#ffa726", "#fff3e0"),
    }

    edge_strs: list[str] = []
    for e in local_edges:
        if e["source"] not in pos or e["target"] not in pos:
            continue
        x1, y1 = point(e["source"])
        x2, y2 = point(e["target"])
        flagged = score_map.get(e["source"], 0.0) >= 0.7 and score_map.get(e["target"], 0.0) >= 0.7
        color = "#ffb3ae" if flagged else "#d8dade"
        dash = ' stroke-dasharray="5 5"' if flagged else ""
        sw = 1 + min(e.get("weight", 1), 3)
        edge_strs.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{color}" stroke-width="{sw}"{dash}/>')

    node_strs: list[str] = []
    for nid in local_nodes:
        if nid not in pos:
            continue
        x, y = point(nid)
        ntype = node_map.get(nid, {}).get("type", "customer")
        score = score_map.get(nid, 0.0)
        if nid == customer_id:
            fill, stroke, radius = "#ffdad7", "#b81120", 15
        elif ntype == "customer" and score >= 0.7:
            fill, stroke, radius = "#ffb3ae", "#ba1a1a", 10
        else:
            stroke, fill, radius = *NODE_COLORS.get(ntype, ("#5c5e66", "#e0e2e6")), 8
        label = html.escape(nid)
        text = f'<text x="{x:.1f}" y="{y + radius + 15:.1f}" text-anchor="middle" font-size="11">{label}</text>' if nid == customer_id else ""
        node_strs.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{fill}" stroke="{stroke}" stroke-width="2"/>{text}')

    legend_y = 10
    legend = (
        f'<g transform="translate(16 {legend_y})" font-family="Inter, sans-serif" font-size="12" fill="#3d4046">'
        f'<rect width="200" height="100" rx="4" fill="#ffffff" stroke="#e0e2e6"/>'
        f'<circle cx="15" cy="17" r="5" fill="#b81120"/><text x="28" y="21">Customer</text>'
        f'<circle cx="15" cy="37" r="5" fill="#5c5e66"/><text x="28" y="41">Device</text>'
        f'<circle cx="15" cy="57" r="5" fill="#4a90d9"/><text x="28" y="61">IP address</text>'
        f'<circle cx="15" cy="77" r="5" fill="#7cb342"/><text x="28" y="81">Address / Payment</text>'
        f'<line x1="7" y1="95" x2="23" y2="95" stroke="#ffb3ae" stroke-width="2" stroke-dasharray="4 3"/><text x="28" y="99">High-risk connection</text>'
        f'</g>'
    )

    svg = f"""
    <div style="width:100%;">
    <svg viewBox="0 0 {width} {height}" width="100%" height="{height}" role="img" aria-label="Customer network for {html.escape(customer_id)}" xmlns="http://www.w3.org/2000/svg">
      <defs><pattern id="grid" width="20" height="20" patternUnits="userSpaceOnUse"><circle cx="1" cy="1" r="1" fill="#e4bdba"/></pattern></defs>
      <rect width="100%" height="100%" fill="url(#grid)"/>
      <g>{''.join(edge_strs)}</g>
      <g font-family="Inter, sans-serif" font-size="12" fill="#191c1f">{''.join(node_strs)}</g>
      {legend}
    </svg>
    </div>
    """
    # st.html(svg)
    st.markdown(svg, unsafe_allow_html=True)


def render_overview() -> None:
    """Overview screen: actual engine metrics, trends, and alerts."""
    st.title("Return Risk Management")
    st.caption("System overview and high-level risk metrics.")
    split = st.selectbox("Dataset window", ["All data", "train", "validation", "test"], key="overview_split")
    decisions, metrics = score_split(split)
    returns_df, _ = frames_for_split(split)
    display = returns_df[["return_id", "customer_id", "return_date", "refund_amount", "return_reason"]].merge(decisions, on=["return_id", "customer_id"], how="inner", validate="one_to_one")
    daily = display.assign(return_date=pd.to_datetime(display["return_date"])).groupby("return_date", as_index=False).agg(total_returns=("return_id", "count"), average_risk=("overall_risk", "mean"))
    flagged = display.loc[display["risk_level"] != "LOW"]

    cards = st.columns(4)
    cards[0].metric("TOTAL RETURNS", f"{len(display):,}", border=True, chart_data=daily["total_returns"].tolist(), chart_type="bar")
    cards[1].metric("RISK SCORE (AVG)", f"{display['overall_risk'].mean() * 100:.1f}/100", border=True, chart_data=(daily["average_risk"] * 100).tolist(), chart_type="line", delta_color="red")
    cards[2].metric("FLAGGED TRANSACTIONS", f"{len(flagged):,}", delta=f"{float(metrics['recall'] or 0):.1%} recall", delta_color="off", border=True)
    cards[3].metric("POTENTIAL LOSS", f"${flagged['refund_amount'].sum():,.0f}", delta="flagged refund value", delta_color="off", border=True)

    eval_results = load_evaluation_results()
    with st.container(border=True):
        st.subheader("Model performance")
        st.caption("Combined RiskEngine evaluation metrics (behavioral + graph models).")
        m1, m2, m3, m4, m5, m6 = st.columns(6)
        m1.metric("Accuracy", f"{eval_results['accuracy']:.1%}")
        m2.metric("Precision", f"{eval_results['precision']:.1%}")
        m3.metric("Recall", f"{eval_results['recall']:.1%}")
        m4.metric("F1 score", f"{eval_results['f1']:.1%}")
        m5.metric("ROC-AUC", f"{eval_results['roc_auc']:.3f}")
        m6.metric("PR-AUC", f"{eval_results['pr_auc']:.3f}")
        st.caption(f"Threshold: {eval_results['evaluation_threshold']} — Total samples: {eval_results['true_negatives'] + eval_results['false_positives'] + eval_results['false_negatives'] + eval_results['true_positives']:,}")
        cm_df = pd.DataFrame(
            [
                {"": "Predicted legitimate", "Actual legitimate": eval_results["true_negatives"], "Actual abusive": eval_results["false_negatives"]},
                {"": "Predicted abusive", "Actual legitimate": eval_results["false_positives"], "Actual abusive": eval_results["true_positives"]},
            ]
        )
        st.dataframe(cm_df, hide_index=True, use_container_width=False)

    with st.container(border=True):
        st.subheader("Return volume & risk trend")
        st.caption("Actual return volume and combined average risk by return date.")
        base = alt.Chart(daily).encode(x=alt.X("return_date:T", title="Return date"))
        bars = base.mark_bar(color="#d8dade").encode(y=alt.Y("total_returns:Q", title="Returns"))
        line = base.mark_line(color="#b81120", point=True).encode(y=alt.Y("average_risk:Q", title="Average combined risk", scale=alt.Scale(domain=[0, 1])), tooltip=["return_date:T", "total_returns:Q", alt.Tooltip("average_risk:Q", format=".1%")])
        st.altair_chart(alt.layer(bars, line).resolve_scale(y="independent"))
    with st.container(border=True):
        st.subheader("High risk alerts")
        st.caption("Highest actual combined risk scores in the selected dataset window.")
        st.dataframe(display.nlargest(8, "overall_risk")[["return_id", "customer_id", "return_date", "refund_amount", "return_reason", "overall_risk", "risk_level", "recommended_action"]], hide_index=True, column_config={"refund_amount": st.column_config.NumberColumn("Refund amount", format="$%.2f"), "overall_risk": st.column_config.ProgressColumn("Risk score", min_value=0, max_value=1, format="%.1f%%")})


def render_network() -> None:
    """Network screen: actual graph, customer selector, and graph evidence."""
    st.title("Return entity graph")
    st.caption("Visualizing customer relationships built from shared devices, IPs, addresses, and payments.")
    returns_df, orders_df = load_datasets()
    graph_data = load_graph_json()
    customer_ids = sorted({n["id"] for n in graph_data["nodes"] if n["type"] == "customer"})
    customer_id = st.selectbox("Customer", customer_ids, key="network_customer")
    scores = all_network_scores()
    score_map = dict(zip(scores["customer_id"].astype(str), scores["network_risk_probability"]))
    evidence = load_risk_engine().graph_model.get_customer_risk_explanation(customer_id, orders_df) #type: ignore
    left, right = st.columns([3, 1], vertical_alignment="top")
    with left:
        with st.container(border=True):
            st.subheader(f"Customer topology: `{customer_id}`")
            st.caption("Two-hop neighborhood in the entity graph.")
            render_network_svg(graph_data, customer_id, score_map) #type: ignore
    with right:
        if "error" in evidence:
            st.error(evidence["error"])
            return
        customer_returns = returns_df.loc[returns_df["customer_id"].astype(str) == customer_id]
        network_risk = float(evidence["network_risk_probability"] or 0.0)
        st.badge("HIGH RISK ENTITY" if evidence["is_high_risk"] else "NETWORK ENTITY", color="red" if evidence["is_high_risk"] else "blue")
        st.subheader(f"Customer: `{customer_id}`")
        m1, m2 = st.columns(2)
        m1.metric("Network risk", f"{network_risk:.1%}", border=True)
        m2.metric("Refund value", f"${customer_returns['refund_amount'].sum():,.0f}", border=True)
        with st.container(border=True):
            st.markdown("**Risk factors**")
            st.write(f"Shared-device customers: `{evidence['sharing']['shared_device_customers']}`")
            st.write(f"Shared-IP customers: `{evidence['sharing']['shared_ip_customers']}`")
            st.write(f"One-hop connections: `{evidence['neighborhood']['one_hop_customer_count']}`")
            st.write(f"Community size: `{evidence['community']['community_size']}`")
        graph = load_customer_graph()
        rows = [{"customer_id": node, "shared entities": graph[customer_id][node].get("n_shared", 0), "network risk": score_map.get(str(node), 0.0)} for node in sorted(graph.neighbors(customer_id), key=lambda node: graph[customer_id][node].get("n_shared", 0), reverse=True)[:5]] if customer_id in graph else []
        st.markdown("**Connected customers**")
        st.dataframe(pd.DataFrame(rows), hide_index=True, column_config={"network risk": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.1f%%")})


def render_investigation() -> None:
    """Investigation screen: select a return, run real inference, request LLM explanation."""
    st.title("Inference engine")
    st.caption("Real-time evaluation with behavioral, graph, and combined risk models.")
    returns_df, orders_df = load_datasets()
    graph_data = load_graph_json()
    preview = ["return_id", "customer_id", "return_date", "refund_amount", "return_reason", "abuse_label"]
    with st.container(border=True):
        st.subheader("Anomaly detection queue")
        st.dataframe(returns_df[preview].head(20), hide_index=True, column_config={"refund_amount": st.column_config.NumberColumn("Refund amount", format="$%.2f")})
    customer_id = st.selectbox("Customer", sorted(returns_df["customer_id"].dropna().astype(str).unique()), key="investigation_customer")
    candidates = returns_df.loc[returns_df["customer_id"].astype(str) == customer_id]
    return_id = st.selectbox("Return", candidates["return_id"].astype(str).tolist(), key="investigation_return")
    if st.button("Run risk assessment", icon=":material/play_arrow:", type="primary", key="run_assessment"):
        selected = candidates.loc[candidates["return_id"].astype(str) == return_id].copy()
        try:
            with st.spinner("Running behavioral and network inference..."):
                engine = load_risk_engine()
                result = engine.assess(selected, orders_df).iloc[0].to_dict()
                evidence = engine.graph_model.get_customer_risk_explanation(customer_id, orders_df) #type: ignore
            st.session_state["investigation_result"] = result
            st.session_state["investigation_evidence"] = evidence
            st.session_state.pop("investigation_explanation", None)
        except Exception as error:
            st.error(f"Risk assessment failed: {error}")
    result = st.session_state.get("investigation_result")
    evidence = st.session_state.get("investigation_evidence")
    if not result or result.get("return_id") != return_id:
        st.caption("Choose a return and run the assessment to view its evidence.")
        return
    scores = all_network_scores()
    score_map = dict(zip(scores["customer_id"].astype(str), scores["network_risk_probability"]))
    left, right = st.columns([3, 2], vertical_alignment="top")
    with left:
        with st.container(border=True):
            st.subheader(f"Local topology: `{customer_id}`")
            render_network_svg(graph_data, customer_id, score_map) #type: ignore
    with right:
        with st.container(border=True):
            st.metric("Combined risk", f"{float(result['overall_risk']):.1%}", border=True)
            risk_badge(str(result["risk_level"]))
            st.write(f"Recommended action: **{result['recommended_action']}**")
            st.progress(float(result["overall_risk"]))
        with st.container(border=True):
            st.subheader("Model signals")
            st.metric("Behavioral return risk", f"{float(result['return_risk']):.1%}")
            st.metric("Network risk", f"{float(result['network_risk']):.1%}")
            if evidence and "error" not in evidence:
                st.write(f"Shared IP customers: `{evidence['sharing']['shared_ip_customers']}`")
                st.write(f"One-hop network: `{evidence['neighborhood']['one_hop_customer_count']}`")
    with st.container(border=True):
        st.subheader("AI synthesis")
        st.caption("Generates a natural language explanation from the graph model evidence.")
        if st.button("Generate explanation", icon=":material/auto_awesome:", key="generate_explanation"):
            prompt = (
                "Analyze this return-risk decision for the customer below. "
                "Use only the structured result and graph evidence provided. "
                "Clearly state the risk level, key contributing factors, and recommended action.\n\n"
                f"Risk result:\n{json.dumps(result, default=str)}\n\n"
                f"Network evidence:\n{json.dumps(evidence, default=str)}"
            )
            try:
                with st.spinner("Generating explanation..."):
                    st.session_state["investigation_explanation"] = query_llm(prompt)
            except Exception as error:
                st.error(f"Could not generate explanation: {error}")
        explanation = st.session_state.get("investigation_explanation")
        if explanation:
            with st.chat_message("assistant", avatar=":material/auto_awesome:"):
                st.write(explanation)


def render_sidebar() -> None:
    """Render the sidebar with branding and sign out."""
    with st.sidebar:
        st.markdown("## :red[ReturnGuard]")
        st.caption("ENTERPRISE ANALYTICS")
        st.space("large")
        st.caption(f"Signed in as {st.session_state.user_email}")
        if st.button("Sign out", icon=":material/logout:", key="sign_out"):
            error = Auth().sign_out()
            if error:
                st.error(f"Sign out failed: {error}")
            else:
                st.session_state.pop("user_email", None)
                st.rerun()


def main() -> None:
    require_authentication()
    render_sidebar()
    tabs = st.tabs([
        ":material/dashboard: Overview",
        ":material/account_tree: Network analysis",
        ":material/psychology: AI inference",
    ])
    try:
        with tabs[0]:
            render_overview()
        with tabs[1]:
            render_network()
        with tabs[2]:
            render_investigation()
    except Exception as error:
        st.error(f"Unable to load ReturnGuard data or models: {error}")
        st.caption("Check DATA_STORAGE_PATH, trained model bundles, and required credentials.")


if __name__ == "__main__":
    main()
