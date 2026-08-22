import os
import streamlit as st
import streamlit.components.v1 as components
from pyvis.network import Network
import tempfile
from utils.ui_utils import (
    get_entity_counts,
    get_graph_sample,
    display_container_name,
    search_nodes,
)

# Color scheme for node types
NODE_COLORS = {
    "Question": "#3498db",  # Blue
    "Answer": "#2ecc71",  # Green
    "Tag": "#e67e22",  # Orange
    "User": "#9b59b6",  # Purple
    "ImportLog": "#95a5a6",  # Gray
}

# Icons for entity types
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

    # Prioritize certain fields
    priority_fields = ["title", "name", "display_name", "id"]
    sorted_keys = sorted(properties.keys(), key=lambda k: (k not in priority_fields, k))

    for key in sorted_keys:
        value = properties[key]
        if value is None:
            continue

        str_val = str(value)
        # Basic markdown stripping
        str_val = str_val.replace("**", "").replace("__", "").replace("`", "")

        # Handle long text
        if len(str_val) > 100:
            str_val = str_val[:97] + "..."

        tooltip += f"{key}: {str_val}\n"

    return tooltip


def create_pyvis_graph(graph_data: dict, height: str = "600px") -> str:
    """Create an interactive Pyvis network graph from Neo4j data."""
    # Create network with inline resources
    net = Network(
        height=height,
        width="100%",
        bgcolor="#1a1a2e",
        font_color="white",
        directed=True,
        select_menu=True,
        filter_menu=True,
        cdn_resources="in_line",
    )

    # Configure physics for better layout
    net.set_options("""
    {
        "nodes": {
            "borderWidth": 2,
            "borderWidthSelected": 4,
            "font": {"size": 12, "face": "Arial"}
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

    # Add nodes with colors based on type
    for node in graph_data.get("nodes", []):
        node_type = node.get("type", "Unknown")
        color = NODE_COLORS.get(node_type, "#888888")
        icon = NODE_ICONS.get(node_type, "●")

        # Ensure ID is a string relative to avoiding JS issues with large ints
        node_id = str(node["id"])

        # Generate formatted tooltip
        tooltip = format_tooltip(node_type, node.get("properties", {}))

        net.add_node(
            node_id,
            label=f"{icon} {node['label'][:25]}",
            title=tooltip,
            color=color,
            size=30 if node_type == "Question" else 25,
            shape="dot",
        )

    # Add edges
    for edge in graph_data.get("edges", []):
        rel_type = edge.get("label", "RELATED")
        net.add_edge(
            str(edge["from"]),
            str(edge["to"]),
            title=rel_type,
            label=rel_type,
            color="#555555",
        )

    # Generate HTML using a safer temp file approach
    try:
        # Create a temp file
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".html", mode="w", encoding="utf-8"
        ) as f:
            temp_path = f.name

        # Write graph to the temp file (this handles opening/closing internally)
        net.save_graph(temp_path)

        # Read the content back
        with open(temp_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        # Clean up
        os.unlink(temp_path)

        return html_content
    except Exception as e:
        st.error(f"Error creating graph visualization: {e}")
        return f"<div>Error creating graph: {str(e)}</div>"


def render_page():
    st.set_page_config(page_title="Neo4j Explorer", page_icon="🔍", layout="wide")

    st.header("🔍 Neo4j Knowledge Graph Explorer")
    st.caption("Explore entities and relationships in your Neo4j database")

    # Display container status in sidebar
    with st.sidebar:
        display_container_name()
        st.divider()

        # Filters
        st.subheader("🎛️ Graph Filters")

        # Focus Node Search
        st.markdown("##### 🎯 Focus on Node")
        search_term = st.text_input(
            "Search Node (Title/Name)", placeholder="e.g. python"
        )
        focus_node_id = ""

        if search_term:
            results = search_nodes(search_term)
            if results:
                options = {f"{r['type']}: {r['label'][:50]}": r["id"] for r in results}
                selected_option = st.selectbox(
                    "Select Node to Focus", options=list(options.keys())
                )
                if selected_option:
                    focus_node_id = options[selected_option]
            else:
                st.caption("No nodes found.")

        st.divider()

        all_node_types = ["Question", "Answer", "Tag", "User"]
        all_rel_types = ["TAGGED", "ANSWERS", "PROVIDED", "ASKED"]

        selected_nodes = st.multiselect(
            "Node Types",
            options=all_node_types,
            default=all_node_types,
            help="Select which entity types to display",
        )

        selected_rels = st.multiselect(
            "Relationship Types",
            options=all_rel_types,
            default=all_rel_types,
            help="Select which relationship types to display",
        )

        node_limit = st.slider(
            "Node Limit",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            help="Maximum number of nodes to display",
        )

        if focus_node_id:
            st.info("Showing neighborhood of selected node.")
            if st.button("❌ Clear Focus"):
                st.rerun()

        refresh_btn = st.button("🔄 Refresh Graph", use_container_width=True)

    # Entity Counts Section
    st.subheader("📊 Database Entities")

    try:
        counts = get_entity_counts()
        node_counts = counts.get("nodes", {})
        rel_counts = counts.get("relationships", {})

        # Node counts
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(
                label=f"{NODE_ICONS['Question']} Questions",
                value=f"{node_counts.get('Question', 0):,}",
            )

        with col2:
            st.metric(
                label=f"{NODE_ICONS['Answer']} Answers",
                value=f"{node_counts.get('Answer', 0):,}",
            )

        with col3:
            st.metric(
                label=f"{NODE_ICONS['Tag']} Tags",
                value=f"{node_counts.get('Tag', 0):,}",
            )

        with col4:
            st.metric(
                label=f"{NODE_ICONS['User']} Users",
                value=f"{node_counts.get('User', 0):,}",
            )

        with col5:
            st.metric(
                label=f"{NODE_ICONS['ImportLog']} Imports",
                value=f"{node_counts.get('ImportLog', 0):,}",
            )

        # Relationship counts
        st.subheader("🔗 Relationships")

        rcol1, rcol2, rcol3, rcol4 = st.columns(4)

        with rcol1:
            st.metric(
                label=f"{REL_ICONS['TAGGED']} Tagged",
                value=f"{rel_counts.get('TAGGED', 0):,}",
            )

        with rcol2:
            st.metric(
                label=f"{REL_ICONS['ANSWERS']} Answers",
                value=f"{rel_counts.get('ANSWERS', 0):,}",
            )

        with rcol3:
            st.metric(
                label=f"{REL_ICONS['PROVIDED']} Provided",
                value=f"{rel_counts.get('PROVIDED', 0):,}",
            )

        with rcol4:
            st.metric(
                label=f"{REL_ICONS['ASKED']} Asked",
                value=f"{rel_counts.get('ASKED', 0):,}",
            )

    except Exception as e:
        st.error(f"Could not fetch entity counts: {e}")

    st.divider()

    # Knowledge Graph Visualization
    st.subheader("🌐 Interactive Knowledge Graph")

    # Legend
    legend_cols = st.columns(5)
    for i, (node_type, color) in enumerate(NODE_COLORS.items()):
        with legend_cols[i % 5]:
            st.markdown(
                f'<span style="color:{color}">●</span> **{NODE_ICONS.get(node_type, "")} {node_type}**',
                unsafe_allow_html=True,
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
                st.info(
                    f"Displaying {len(graph_data['nodes'])} nodes and {len(graph_data['edges'])} relationships"
                )

                # Generate and display the graph
                html_content = create_pyvis_graph(graph_data, height="650px")
                st.iframe(html_content, height=700)
            else:
                st.warning(
                    "No graph data found. Try importing some data first using the Loader page."
                )
                if st.button("📥 Go to Loader", use_container_width=False):
                    st.switch_page("pages/loader.py")

    except Exception as e:
        st.error(f"Could not load knowledge graph: {e}")
        st.exception(e)

    # Quick Actions
    st.divider()
    st.subheader("🚀 Quick Actions")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 Refresh All", use_container_width=True):
            st.rerun()

    with col2:
        if st.button("📊 Go to Dashboard", use_container_width=True):
            st.switch_page("pages/dashboard.py")

    with col3:
        if st.button("📥 Go to Loader", use_container_width=True):
            st.switch_page("pages/loader.py")


render_page()
