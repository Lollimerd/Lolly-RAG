"""Database analytics and graph visualization helpers."""

import logging
from typing import Any, Dict, List

from setup.init_config import get_graph_instance

logger = logging.getLogger(__name__)

# Cypher Queries for Analytics and Visualizations
_DATABASE_SUMMARY_QUERY = """
MATCH (d:Document) WITH count(d) AS total_documents
MATCH (dc:DocumentChunk) WITH total_documents, count(dc) AS total_chunks
MATCH (u:AppUser) WITH total_documents, total_chunks, count(u) AS total_users
MATCH (s:Session) WITH total_documents, total_chunks, total_users, count(s) AS total_sessions
MATCH (m:Message) WITH total_documents, total_chunks, total_users, total_sessions, count(m) AS total_messages
RETURN total_documents, total_chunks, total_users, total_sessions, total_messages
"""

_ENTITY_NODES_QUERY = """
CALL {
    MATCH (d:Document) RETURN 'Document' AS label, count(d) AS count UNION ALL
    MATCH (dc:DocumentChunk) RETURN 'DocumentChunk' AS label, count(dc) AS count UNION ALL
    MATCH (u:AppUser) RETURN 'AppUser' AS label, count(u) AS count UNION ALL
    MATCH (s:Session) RETURN 'Session' AS label, count(s) AS count UNION ALL
    MATCH (m:Message) RETURN 'Message' AS label, count(m) AS count
}
RETURN label, count
"""

_ENTITY_RELS_QUERY = """
CALL {
    MATCH ()-[r:HAS_CHUNK]->() RETURN 'HAS_CHUNK' AS type, count(r) AS count UNION ALL
    MATCH ()-[r:HAS_SESSION]->() RETURN 'HAS_SESSION' AS type, count(r) AS count UNION ALL
    MATCH ()-[r:HAS_MESSAGE]->() RETURN 'HAS_MESSAGE' AS type, count(r) AS count UNION ALL
    MATCH ()-[r:LAST_MESSAGE]->() RETURN 'LAST_MESSAGE' AS type, count(r) AS count
}
RETURN type, count
"""

_SEARCH_NODES_QUERY = """
MATCH (n)
WHERE (n.filename IS NOT NULL AND toLower(n.filename) CONTAINS toLower($term))
   OR (n.description IS NOT NULL AND toLower(n.description) CONTAINS toLower($term))
   OR (n.content IS NOT NULL AND toLower(n.content) CONTAINS toLower($term))
   OR (n.source IS NOT NULL AND toLower(n.source) CONTAINS toLower($term))
   OR (n.topic IS NOT NULL AND toLower(n.topic) CONTAINS toLower($term))
   OR (n.id IS NOT NULL AND toLower(toString(n.id)) CONTAINS toLower($term))
RETURN elementId(n) AS id,
       labels(n)[0] AS type,
       COALESCE(n.filename, n.topic, n.source, substring(n.content, 0, 40), n.id, labels(n)[0]) AS label
LIMIT $limit
"""

_NODE_CASE = """
CASE labels(n)[0]
    WHEN 'Document' THEN COALESCE(n.filename, 'Doc ' + elementId(n))
    WHEN 'DocumentChunk' THEN 'Chunk ' + COALESCE(toString(n.chunk_index), elementId(n))
    WHEN 'AppUser' THEN 'User ' + COALESCE(n.id, elementId(n))
    WHEN 'Session' THEN COALESCE(n.topic, 'Session ' + elementId(n))
    WHEN 'Message' THEN COALESCE(substring(n.content, 0, 30), 'Message ' + elementId(n))
    ELSE elementId(n)
END
"""
_M_CASE = _NODE_CASE.replace("n.", "m.").replace("labels(n)", "labels(m)")

def get_database_summary() -> dict:
    """Args: None."""
    result = get_graph_instance().query(_DATABASE_SUMMARY_QUERY)
    if result:
        return result[0]
    return {
        "total_documents": 0,
        "total_chunks": 0,
        "total_users": 0,
        "total_sessions": 0,
        "total_messages": 0,
    }

def get_import_history(limit: int = 20) -> list:
    """Args: limit: Number of records."""
    return []

def get_entity_counts() -> dict:
    """Args: None."""
    driver = get_graph_instance()
    node_res = driver.query(_ENTITY_NODES_QUERY)
    rel_res = driver.query(_ENTITY_RELS_QUERY)
    return {
        "nodes": {r["label"]: r["count"] for r in (node_res or [])},
        "relationships": {r["type"]: r["count"] for r in (rel_res or [])},
    }

def search_nodes(search_term: str, limit: int = 10) -> list:
    """Args: search_term: Query text, limit: Maximum matches."""
    result = get_graph_instance().query(
        _SEARCH_NODES_QUERY,
        {
            "term": search_term,
            "limit": limit,
        },
    )
    return result or []

def get_graph_sample(node_types: List[str], rel_types: List[str], limit: int = 50, focus_node_id: str = "") -> dict:
    """Args: node_types: Node filters, rel_types: Relationship filters, limit: Max elements, focus_node_id: Optional root node ID."""
    node_types = node_types or ["Document", "DocumentChunk", "AppUser", "Session", "Message"]
    rel_types = rel_types or ["HAS_CHUNK", "HAS_SESSION", "HAS_MESSAGE", "LAST_MESSAGE"]
    driver = get_graph_instance()

    if focus_node_id:
        query = f"""
        MATCH path = (root)-[r*1..2]-(m)
        WHERE elementId(root) = $focus_node_id
          AND ALL(n IN nodes(path) WHERE labels(n)[0] IN $node_types)
          AND ALL(rel IN relationships(path) WHERE type(rel) IN $rel_types)
        WITH relationships(path) AS rels
        UNWIND rels AS r
        WITH startNode(r) AS n, r, endNode(r) AS m
        LIMIT $limit
        RETURN elementId(n) AS source_id,
               labels(n)[0] AS source_label,
               properties(n) AS source_props,
               {_NODE_CASE} AS source_name,
               elementId(m) AS target_id,
               labels(m)[0] AS target_label,
               properties(m) AS target_props,
               {_M_CASE} AS target_name,
               type(r) AS rel_type
        """
        params = {
            "node_types": node_types,
            "rel_types": rel_types,
            "limit": limit * 3,
            "focus_node_id": focus_node_id,
        }
    else:
        query = f"""
        MATCH (n)-[r]->(m)
        WHERE (labels(n)[0] IN $node_types OR labels(m)[0] IN $node_types)
          AND type(r) IN $rel_types
        WITH n, r, m
        LIMIT $limit
        RETURN elementId(n) AS source_id,
               labels(n)[0] AS source_label,
               properties(n) AS source_props,
               {_NODE_CASE} AS source_name,
               elementId(m) AS target_id,
               labels(m)[0] AS target_label,
               properties(m) AS target_props,
               {_M_CASE} AS target_name,
               type(r) AS rel_type
        """
        params = {
            "node_types": node_types,
            "rel_types": rel_types,
            "limit": limit * 2,
        }

    results = driver.query(query, params) or []
    nodes, edges, node_ids = [], [], set()
    for r in results:
        for side in ("source", "target"):
            nid = r[f"{side}_id"]
            if nid not in node_ids:
                name = r[f"{side}_name"]
                props = r.get(f"{side}_props", {})
                for k, v in list(props.items()):
                    if "date" in k.lower() or "time" in k.lower() or hasattr(v, "isoformat"):
                        props[k] = str(v)
                nodes.append({
                    "id": nid,
                    "label": name[:30] if name else str(nid),
                    "type": r[f"{side}_label"],
                    "title": name,
                    "properties": props,
                })
                node_ids.add(nid)
        edges.append({
            "from": r["source_id"],
            "to": r["target_id"],
            "label": r["rel_type"],
            "title": r["rel_type"],
        })
        if len(nodes) >= limit * 1.5:
            break

    return {"nodes": nodes, "edges": edges}
