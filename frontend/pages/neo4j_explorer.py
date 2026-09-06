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

# ---------------------------------------------------------------------------
# Page configuration (MUST be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Neo4j Explorer — Lolly-RAG",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Color scheme for node types
NODE_COLORS = {
    "Document": "#3B82F6",       # Blue
    "DocumentChunk": "#10B981",  # Green
    "AppUser": "#8B5CF6",        # Purple
    "Session": "#F59E0B",        # Amber
    "Message": "#EC4899",        # Pink
}

# Icons for entity types
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
    """Format node properties into a clean tooltip."""
    tooltip = f"Type: {node_type}\n"

    priority_fields = ["filename", "id", "description", "source", "topic", "type", "chunk_index"]
    sorted_keys = sorted(properties.keys(), key=lambda k: (k not in priority_fields, k))

    for key in sorted_keys:
        value = properties[key]
        if value is None:
            continue

        str_val = str(value)
        str_val = str_val.replace("**", "").replace("__", "").replace("`", "")

        if len(str_val) > 100:
            str_val = str_val[:97] + "..."

        tooltip += f"{key}: {str_val}\n"

    return tooltip


def create_pyvis_graph(graph_data: dict, height: str = "600px") -> str:
    """Create an interactive Pyvis network graph from Neo4j data."""
    net = Network(
        height=height,
        width="100%",
        bgcolor="#0F172A",
        font_color="white",
        directed=True,
        select_menu=True,
        filter_menu=True,
        cdn_resources="in_line",
    )

    net.set_options("""
    {
        "nodes": {
            "borderWidth": 2,
            "borderWidthSelected": 4,
            "font": {"size": 12, "face": "Arial", "color": "#E2E8F0"}
        },
        "edges": {
            "color": {"inherit": true, "opacity": 0.6},
            "smooth": {"type": "continuous"},
            "arrows": {"to": {"enabled": true, "scaleFactor": 0.5}}
        },
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.01,
                "springLength": 100,
                "springConstant": 0.08
            },
            "solver": "forceAtlas2Based",
            "stabilization": {"enabled": true, "iterations": 200}
        },
        "interaction": {
            "navigationButtons": true,
            "keyboard": {"enabled": true}
        }
    }
    """)

    for node in graph_data.get("nodes", []):
        node_type = node.get("type", "Unknown")
        color = NODE_COLORS.get(node_type, "#888888")
        icon = NODE_ICONS.get(node_type, "●")
        node_id = str(node["id"])
        tooltip = format_tooltip(node_type, node.get("properties", {}))

        net.add_node(
            node_id,
            label=f"{icon} {node['label'][:25]}",
            title=tooltip,
            color=color,
            size=32 if node_type == "Document" else 24,
            shape="dot",
        )

    for edge in graph_data.get("edges", []):
        rel_type = edge.get("label", "RELATED")
        net.add_edge(
            str(edge["from"]),
            str(edge["to"]),
            title=rel_type,
            label=rel_type,
            color="#475569",
        )

    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".html", mode="w", encoding="utf-8"
        ) as f:
            temp_path = f.name

        net.save_graph(temp_path)

        with open(temp_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        os.unlink(temp_path)
        return html_content
    except Exception as e:
        st.error(f"Error creating graph visualization: {e}", icon=":material/error:")
        return f"<div>Error creating graph: {str(e)}</div>"


# ---------------------------------------------------------------------------
# Direct Script Execution (Streamlit Best Practices)
# ---------------------------------------------------------------------------
st.html(
    """
    <div style="text-align: center; padding: 12px 0 8px;">
        <h1 style="
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #60A5FA 0%, #A78BFA 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 4px;
        ">Neo4j Knowledge Graph Explorer</h1>
        <p style="color: #94A3B8; font-size: 0.95rem; margin: 0;">
            Explore entities, vector embeddings, and multi-hop relationships in your Neo4j database.
        </p>
    </div>
    """
)

# Display container status & filters in sidebar
with st.sidebar:
    display_container_name()

    st.subheader("Graph filters", help="Filter nodes and relationships")

    # Focus Node Search
    st_markdown("##### Focus on node")
    search_term = st.text_input(
        "Search node",
        placeholder="Search filename, topic, or ID...",
        label_visibility="collapsed",
    )
    focus_node_id = ""

    if search_term:
        results = search_nodes(search_term)
        if results:
            options = {f"{r['type']}: {r['label'][:50]}": r["id"] for r in results}
            selected_option = st.selectbox(
                "Select node to focus",
                options=list(options.keys()),
            )
            if selected_option:
                focus_node_id = options[selected_option]
        else:
            st.caption("No matching nodes found.")

    all_node_types = ["Document", "DocumentChunk", "AppUser", "Session", "Message"]
    all_rel_types = ["HAS_CHUNK", "HAS_SESSION", "HAS_MESSAGE", "LAST_MESSAGE"]

    selected_nodes = st.multiselect(
        "Node types",
        options=all_node_types,
        default=all_node_types,
        help="Select entity types to display",
    )

    selected_rels = st.multiselect(
        "Relationship types",
        options=all_rel_types,
        default=all_rel_types,
        help="Select relationship types to display",
    )

    node_limit = st.slider(
        "Node limit",
        min_value=10,
        max_value=200,
        value=50,
        step=10,
        help="Maximum number of nodes to render",
    )

    if focus_node_id:
        st.info("Showing neighborhood of selected focus node.", icon=":material/filter_center_focus:")
        if st.button("Clear focus", icon=":material/close:"):
            st.rerun()

    refresh_btn = st.button("Refresh graph", icon=":material/refresh:", width="stretch")

# Entity Counts Section
st.subheader("Database entities", help="Current Neo4j node metrics")

try:
    counts = get_entity_counts()
    node_counts = counts.get("nodes", {})
    rel_counts = counts.get("relationships", {})

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        with st.container(border=True):
            st.metric(
                label=f"{NODE_ICONS['Document']} Documents",
                value=f"{node_counts.get('Document', 0):,}",
            )

    with col2:
        with st.container(border=True):
            st.metric(
                label=f"{NODE_ICONS['DocumentChunk']} Chunks",
                value=f"{node_counts.get('DocumentChunk', 0):,}",
            )

    with col3:
        with st.container(border=True):
            st.metric(
                label=f"{NODE_ICONS['AppUser']} Users",
                value=f"{node_counts.get('AppUser', 0):,}",
            )

    with col4:
        with st.container(border=True):
            st.metric(
                label=f"{NODE_ICONS['Session']} Sessions",
                value=f"{node_counts.get('Session', 0):,}",
            )

    with col5:
        with st.container(border=True):
            st.metric(
                label=f"{NODE_ICONS['Message']} Messages",
                value=f"{node_counts.get('Message', 0):,}",
            )

    # Relationship counts
    st.subheader("Relationships", help="Current Neo4j edge metrics")

    rcol1, rcol2, rcol3, rcol4 = st.columns(4)

    with rcol1:
        with st.container(border=True):
            st.metric(
                label=f"{REL_ICONS['HAS_CHUNK']} Has Chunk",
                value=f"{rel_counts.get('HAS_CHUNK', 0):,}",
            )

    with rcol2:
        with st.container(border=True):
            st.metric(
                label=f"{REL_ICONS['HAS_SESSION']} Has Session",
                value=f"{rel_counts.get('HAS_SESSION', 0):,}",
            )

    with rcol3:
        with st.container(border=True):
            st.metric(
                label=f"{REL_ICONS['HAS_MESSAGE']} Has Message",
                value=f"{rel_counts.get('HAS_MESSAGE', 0):,}",
            )

    with rcol4:
        with st.container(border=True):
            st.metric(
                label=f"{REL_ICONS['LAST_MESSAGE']} Last Message",
                value=f"{rel_counts.get('LAST_MESSAGE', 0):,}",
            )

except Exception as e:
    st.error(f"Could not fetch entity counts: {e}", icon=":material/error:")

# Knowledge Graph Visualization
st.subheader("Interactive knowledge graph", help="Interactive network visualization")

# Legend
legend_cols = st.columns(5)
for i, (node_type, color) in enumerate(NODE_COLORS.items()):
    with legend_cols[i % 5]:
        st.html(
            f'<span style="color:{color}">●</span> <strong>{NODE_ICONS.get(node_type, "")} {node_type}</strong>'
        )

try:
    with st.spinner("Loading knowledge graph..."):
        graph_data = get_graph_sample(
            node_types=selected_nodes,
            rel_types=selected_rels,
            limit=node_limit,
            focus_node_id=focus_node_id,
        )

        if graph_data["nodes"]:
            st.caption(
                f":material/hub: Displaying **{len(graph_data['nodes'])}** nodes and **{len(graph_data['edges'])}** relationships"
            )
            html_content = create_pyvis_graph(graph_data, height="650px")
            st.iframe(html_content, height=700, width="stretch")
        else:
            st.warning(
                "No graph data found. Try uploading documents first using Document Explorer.",
                icon=":material/info:",
            )
            if st.button("Go to Document Explorer", icon=":material/folder:"):
                st.switch_page("pages/doc_ingestion.py")

except Exception as e:
    st.error(f"Could not load knowledge graph: {e}", icon=":material/error:")
    st.exception(e)

# Quick Actions
st.subheader("Quick actions")

act_col1, act_col2 = st.columns(2)

with act_col1:
    if st.button("Refresh all", icon=":material/refresh:", width="stretch"):
        st.rerun()

with act_col2:
    if st.button("Go to Document Explorer", icon=":material/folder:", key="nav_doc_exp", width="stretch"):
        st.switch_page("pages/doc_ingestion.py")
