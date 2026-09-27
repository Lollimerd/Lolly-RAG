import os
import tempfile
import streamlit as st
from streamlit_markdown import st_markdown
from pyvis.network import Network
from utils.ui_utils import (
    get_entity_counts,
    get_graph_sample,
    display_container_name,
    search_nodes,
)

st.set_page_config(
    page_title="Neo4j Explorer — Lolly-RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

NODE_COLORS = {
    "Document": "#3B82F6",
    "DocumentChunk": "#10B981",
    "AppUser": "#8B5CF6",
    "Session": "#F59E0B",
    "Message": "#EC4899",
}

NODE_ICONS = {
    "Document": "📄",
    "DocumentChunk": "🧩",
    "AppUser": "👤",
    "Session": "💬",
    "Message": "✉️",
}

REL_ICONS = {
    "HAS_CHUNK": "🔗",
    "HAS_SESSION": "📂",
    "HAS_MESSAGE": "💬",
    "LAST_MESSAGE": "📌",
}

def format_tooltip(node_type: str, properties: dict) -> str:
    """Args: node_type: Type of node, properties: Property mapping."""
    priority = ["filename", "id", "description", "source", "topic", "type", "chunk_index"]
    lines = [f"Type: {node_type}"]
    for k in sorted(properties.keys(), key=lambda x: (x not in priority, x)):
        v = properties[k]
        if v is not None:
            clean = str(v).replace("**", "").replace("__", "").replace("`", "")
            lines.append(f"{k}: {clean[:97] + '...' if len(clean) > 100 else clean}")
    return "\n".join(lines) + "\n"

def create_pyvis_graph(graph_data: dict, height: str = "600px") -> str:
    """Args: graph_data: Graph dictionary, height: Viewport height."""
    net = Network(height=height, width="100%", bgcolor="#0F172A", font_color="white", directed=True, select_menu=True, filter_menu=True, cdn_resources="in_line")
    net.set_options(
        '{"nodes":{"borderWidth":2,"borderWidthSelected":4,"font":{"size":12,"face":"Arial","color":"#E2E8F0"}},'
        '"edges":{"color":{"inherit":true,"opacity":0.6},"smooth":{"type":"continuous"},"arrows":{"to":{"enabled":true,"scaleFactor":0.5}}},'
        '"physics":{"forceAtlas2Based":{"gravitationalConstant":-50,"centralGravity":0.01,"springLength":100,"springConstant":0.08},'
        '"solver":"forceAtlas2Based","stabilization":{"enabled":true,"iterations":200}},'
        '"interaction":{"navigationButtons":true,"keyboard":{"enabled":true}}}'
    )
    for node in graph_data.get("nodes", []):
        ntype = node.get("type", "Unknown")
        net.add_node(
            str(node["id"]), label=f"{NODE_ICONS.get(ntype, '●')} {node['label'][:25]}",
            title=format_tooltip(ntype, node.get("properties", {})), color=NODE_COLORS.get(ntype, "#888888"),
            size=32 if ntype == "Document" else 24, shape="dot",
        )
    for edge in graph_data.get("edges", []):
        lbl = edge.get("label", "RELATED")
        net.add_edge(str(edge["from"]), str(edge["to"]), title=lbl, label=lbl, color="#475569")
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".html", mode="w", encoding="utf-8") as f:
            tpath = f.name
        net.save_graph(tpath)
        with open(tpath, "r", encoding="utf-8") as f:
            html = f.read()
        os.unlink(tpath)
        return html
    except Exception as e:
        st.error(f"Error creating graph visualization: {e}", icon=":material/error:")
        return f"<div>Error creating graph: {str(e)}</div>"

st.html(
    """
    <div style="text-align: center; padding: 12px 0 8px;">
        <h1 style="font-size: 2.2rem; font-weight: 700; background: linear-gradient(135deg, #60A5FA 0%, #A78BFA 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 4px;">
            Neo4j Knowledge Graph Explorer
        </h1>
        <p style="color: #94A3B8; font-size: 0.95rem; margin: 0;">
            Explore entities, vector embeddings, and multi-hop relationships in your Neo4j database.
        </p>
    </div>
    """
)

with st.sidebar:
    display_container_name()
    st.subheader("Graph filters", help="Filter nodes and relationships")
    st_markdown("##### Focus on node")
    search_term = st.text_input("Search node", placeholder="Search filename, topic, or ID...", label_visibility="collapsed")
    focus_node_id = ""

    if search_term:
        results = search_nodes(search_term)
        if results:
            options = {f"{r['type']}: {r['label'][:50]}": r["id"] for r in results}
            selected_option = st.selectbox("Select node to focus", options=list(options.keys()))
            if selected_option:
                focus_node_id = options[selected_option]
        else:
            st.caption("No matching nodes found.")

    all_node_types = ["Document", "DocumentChunk", "AppUser", "Session", "Message"]
    all_rel_types = ["HAS_CHUNK", "HAS_SESSION", "HAS_MESSAGE", "LAST_MESSAGE"]

    selected_nodes = st.multiselect("Node types", options=all_node_types, default=all_node_types, help="Select entity types to display")
    selected_rels = st.multiselect("Relationship types", options=all_rel_types, default=all_rel_types, help="Select relationship types to display")
    node_limit = st.slider("Node limit", min_value=10, max_value=200, value=50, step=10, help="Maximum number of nodes to render")

    if focus_node_id:
        st.info("Showing neighborhood of selected focus node.", icon=":material/filter_center_focus:")
        if st.button("Clear focus", icon=":material/close:"):
            st.rerun()

    st.button("Refresh graph", icon=":material/refresh:", width="stretch")

st.subheader("Database entities", help="Current Neo4j node metrics")
try:
    counts = get_entity_counts()
    node_counts = counts.get("nodes", {})
    rel_counts = counts.get("relationships", {})

    for col, (ntype, nlabel) in zip(
        st.columns(5),
        [("Document", "Documents"), ("DocumentChunk", "Chunks"), ("AppUser", "Users"), ("Session", "Sessions"), ("Message", "Messages")],
    ):
        with col:
            with st.container(border=True):
                st.metric(f"{NODE_ICONS[ntype]} {nlabel}", f"{node_counts.get(ntype, 0):,}")

    st.subheader("Relationships", help="Current Neo4j edge metrics")
    for col, (rel, label) in zip(
        st.columns(4),
        [("HAS_CHUNK", "Has Chunk"), ("HAS_SESSION", "Has Session"), ("HAS_MESSAGE", "Has Message"), ("LAST_MESSAGE", "Last Message")],
    ):
        with col:
            with st.container(border=True):
                st.metric(f"{REL_ICONS[rel]} {label}", f"{rel_counts.get(rel, 0):,}")
except Exception as e:
    st.error(f"Could not fetch entity counts: {e}", icon=":material/error:")

st.subheader("Interactive knowledge graph", help="Interactive network visualization")
for col, (node_type, color) in zip(st.columns(5), NODE_COLORS.items()):
    with col:
        st.html(f'<span style="color:{color}">●</span> <strong>{NODE_ICONS.get(node_type, "")} {node_type}</strong>')

try:
    with st.spinner("Loading knowledge graph..."):
        graph_data = get_graph_sample(
            node_types=selected_nodes, rel_types=selected_rels, limit=node_limit, focus_node_id=focus_node_id,
        )
        if graph_data["nodes"]:
            st.caption(f":material/hub: Displaying **{len(graph_data['nodes'])}** nodes and **{len(graph_data['edges'])}** relationships")
            html_content = create_pyvis_graph(graph_data, height="650px")
            st.iframe(html_content, height=700, width="stretch")
        else:
            st.warning("No graph data found. Try uploading documents first using Document Explorer.", icon=":material/info:")
            if st.button("Go to Document Explorer", icon=":material/folder:"):
                st.switch_page("pages/doc_ingestion.py")
except Exception as e:
    st.error(f"Could not load knowledge graph: {e}", icon=":material/error:")
    st.exception(e)

st.subheader("Quick actions")
act_col1, act_col2 = st.columns(2)
with act_col1:
    if st.button("Refresh all", icon=":material/refresh:", width="stretch"):
        st.rerun()
with act_col2:
    if st.button("Go to Document Explorer", icon=":material/folder:", key="nav_doc_exp", width="stretch"):
        st.switch_page("pages/doc_ingestion.py")
