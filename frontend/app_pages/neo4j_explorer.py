import os
import streamlit as st
from pyvis.network import Network
from utils.ui_utils import (
    display_container_name,
    get_entity_counts,
    get_graph_sample,
    search_nodes,
)

# Color scheme for node types
NODE_COLORS = {
    "Question": "#3B82F6",  # Blue
    "Answer": "#10B981",    # Emerald
    "Tag": "#F59E0B",       # Amber
    "User": "#8B5CF6",      # Purple
    "ImportLog": "#64748B",  # Slate
}

NODE_ICONS = {
    "Question": "❓",
    "Answer": "💬",
    "Tag": "🏷️",
    "User": "👤",
    "ImportLog": "📦",
}

REL_ICONS = {
    "TAGGED": "🔗",
    "ANSWERS": "💡",
    "PROVIDED": "✍️",
    "ASKED": "🙋",
}


def format_tooltip(node_type: str, properties: dict) -> str:
    """Format node properties into a clean tooltip."""
    tooltip = f"Type: {node_type}\n"
    priority_fields = ["title", "name", "display_name", "id"]
    sorted_keys = sorted(properties.keys(), key=lambda k: (k not in priority_fields, k))

    for key in sorted_keys:
        value = properties[key]
        if value is None:
            continue
        str_val = str(value).replace("**", "").replace("__", "").replace("`", "")
        if len(str_val) > 100:
            str_val = str_val[:97] + "..."
        tooltip += f"{key}: {str_val}\n"

    return tooltip


def create_pyvis_graph(graph_data: dict, height: str = "600px") -> str:
    """Create an interactive Pyvis network graph entirely in memory."""
    net = Network(
        height=height,
        width="100%",
        bgcolor="#0F172A",
        font_color="#F8FAFC",
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
            "font": {"size": 12, "face": "Arial", "color": "#F8FAFC"}
        },
        "edges": {
            "color": {"inherit": true},
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
        color = NODE_COLORS.get(node_type, "#64748B")
        icon = NODE_ICONS.get(node_type, "●")
        node_id = str(node["id"])
        tooltip = format_tooltip(node_type, node.get("properties", {}))

        net.add_node(
            node_id,
            label=f"{icon} {node['label'][:25]}",
            title=tooltip,
            color=color,
            size=30 if node_type == "Question" else 25,
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
        # Pyvis in-memory generation
        return net.generate_html()
    except Exception as e:
        return f"<div>Error creating graph visualization: {e}</div>"


# ---------------------------------------------------------------------------
# Page Layout
# ---------------------------------------------------------------------------
st.header("Neo4j Graph Explorer")
st.caption("Inspect entities, relationships, tags, and cluster neighborhoods in your graph database.")

with st.sidebar:
    st.subheader("Graph filters", help="Filter node and relationship types in the visualization")

    search_term = st.text_input(
        "Focus on node (title/name)", placeholder="e.g. python, fastapi", key="explorer_search_node"
    )
    focus_node_id = ""

    if search_term:
        results = search_nodes(search_term)
        if results:
            options = {f"{r['type']}: {r['label'][:50]}": r["id"] for r in results}
            selected_option = st.selectbox(
                "Select node to focus", options=list(options.keys())
            )
            if selected_option:
                focus_node_id = options[selected_option]
        else:
            st.caption("No matching nodes found.")

    st.divider()

    all_node_types = ["Question", "Answer", "Tag", "User"]
    all_rel_types = ["TAGGED", "ANSWERS", "PROVIDED", "ASKED"]

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
        help="Maximum number of nodes to render in the physics graph",
    )

    if focus_node_id:
        st.info("Showing neighborhood of selected node.", icon=":material/center_focus_strong:")

    if st.button("Refresh graph", icon=":material/refresh:"):
        st.rerun()

# --- Entity & Relationship Counts ---
st.subheader("Database statistics")

try:
    counts = get_entity_counts()
    node_counts = counts.get("nodes", {})
    rel_counts = counts.get("relationships", {})

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        with st.container(border=True):
            st.metric(label="Questions", value=f"{node_counts.get('Question', 0):,}")
    with c2:
        with st.container(border=True):
            st.metric(label="Answers", value=f"{node_counts.get('Answer', 0):,}")
    with c3:
        with st.container(border=True):
            st.metric(label="Tags", value=f"{node_counts.get('Tag', 0):,}")
    with c4:
        with st.container(border=True):
            st.metric(label="Users", value=f"{node_counts.get('User', 0):,}")
    with c5:
        with st.container(border=True):
            st.metric(label="Import records", value=f"{node_counts.get('ImportLog', 0):,}")

    r1, r2, r3, r4 = st.columns(4)
    with r1:
        with st.container(border=True):
            st.metric(label="TAGGED links", value=f"{rel_counts.get('TAGGED', 0):,}")
    with r2:
        with st.container(border=True):
            st.metric(label="ANSWERS links", value=f"{rel_counts.get('ANSWERS', 0):,}")
    with r3:
        with st.container(border=True):
            st.metric(label="PROVIDED links", value=f"{rel_counts.get('PROVIDED', 0):,}")
    with r4:
        with st.container(border=True):
            st.metric(label="ASKED links", value=f"{rel_counts.get('ASKED', 0):,}")

except Exception as e:
    st.error(f"Could not fetch entity counts: {e}", icon=":material/error:")

st.divider()

# --- Interactive Graph Rendering ---
st.subheader("Interactive knowledge graph")

# Legend row
leg_cols = st.columns(5)
for i, (node_type, color) in enumerate(NODE_COLORS.items()):
    with leg_cols[i % 5]:
        st.markdown(
            f'<span style="color:{color}">●</span> **{NODE_ICONS.get(node_type, "")} {node_type}**',
            unsafe_allow_html=True,
        )

try:
    with st.spinner("Rendering interactive graph..."):
        graph_data = get_graph_sample(
            node_types=selected_nodes,
            rel_types=selected_rels,
            limit=node_limit,
            focus_node_id=focus_node_id,
        )

        if graph_data.get("nodes"):
            st.caption(f"Displaying **{len(graph_data['nodes'])}** nodes and **{len(graph_data['edges'])}** relationships")
            html_content = create_pyvis_graph(graph_data, height="650px")
            st.iframe(html_content, height=700)
        else:
            st.warning("No graph data found matching current filter criteria.", icon=":material/info:")

except Exception as e:
    st.error(f"Could not load knowledge graph: {e}", icon=":material/error:")
