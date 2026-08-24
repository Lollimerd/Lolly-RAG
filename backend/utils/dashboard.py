from typing import List, Dict, Any, Optional
from setup.init_config import get_graph_instance
import logging

logger = logging.getLogger(__name__)

def get_database_summary():
    """Get summary statistics from the database for documents, users, sessions, and messages."""
    summary_query = """
    MATCH (d:Document)
    WITH count(d) as total_documents
    MATCH (dc:DocumentChunk)
    WITH total_documents, count(dc) as total_chunks
    MATCH (u:AppUser)
    WITH total_documents, total_chunks, count(u) as total_users
    MATCH (s:Session)
    WITH total_documents, total_chunks, total_users, count(s) as total_sessions
    MATCH (m:Message)
    WITH total_documents, total_chunks, total_users, total_sessions, count(m) as total_messages
    RETURN total_documents, total_chunks, total_users, total_sessions, total_messages
    """
    driver = get_graph_instance()
    result = driver.query(summary_query)
    if result and len(result) > 0:
        return result[0]
    return {
        "total_documents": 0,
        "total_chunks": 0,
        "total_users": 0,
        "total_sessions": 0,
        "total_messages": 0,
    }

def get_import_history(limit: int = 20):
    """Placeholder for legacy import history queries."""
    return []

def get_entity_counts():
    """Get counts for all entity types and relationships in the database."""
    node_counts_query = """
    CALL {
        MATCH (d:Document) RETURN 'Document' as label, count(d) as count
        UNION ALL
        MATCH (dc:DocumentChunk) RETURN 'DocumentChunk' as label, count(dc) as count
        UNION ALL
        MATCH (u:AppUser) RETURN 'AppUser' as label, count(u) as count
        UNION ALL
        MATCH (s:Session) RETURN 'Session' as label, count(s) as count
        UNION ALL
        MATCH (m:Message) RETURN 'Message' as label, count(m) as count
    }
    RETURN label, count
    """

    rel_counts_query = """
    CALL {
        MATCH ()-[r:HAS_CHUNK]->() RETURN 'HAS_CHUNK' as type, count(r) as count
        UNION ALL
        MATCH ()-[r:HAS_SESSION]->() RETURN 'HAS_SESSION' as type, count(r) as count
        UNION ALL
        MATCH ()-[r:HAS_MESSAGE]->() RETURN 'HAS_MESSAGE' as type, count(r) as count
        UNION ALL
        MATCH ()-[r:LAST_MESSAGE]->() RETURN 'LAST_MESSAGE' as type, count(r) as count
    }
    RETURN type, count
    """
    driver = get_graph_instance()
    node_results = driver.query(node_counts_query)
    rel_results = driver.query(rel_counts_query)

    nodes = {r["label"]: r["count"] for r in node_results} if node_results else {}
    relationships = {r["type"]: r["count"] for r in rel_results} if rel_results else {}

    return {"nodes": nodes, "relationships": relationships}

def search_nodes(search_term: str, limit: int = 10):
    """Search for nodes by filename, description, content, source, id, or topic."""
    query = """
    MATCH (n)
    WHERE (n.filename IS NOT NULL AND toLower(n.filename) CONTAINS toLower($term))
       OR (n.description IS NOT NULL AND toLower(n.description) CONTAINS toLower($term))
       OR (n.content IS NOT NULL AND toLower(n.content) CONTAINS toLower($term))
       OR (n.source IS NOT NULL AND toLower(n.source) CONTAINS toLower($term))
       OR (n.topic IS NOT NULL AND toLower(n.topic) CONTAINS toLower($term))
       OR (n.id IS NOT NULL AND toLower(toString(n.id)) CONTAINS toLower($term))
    RETURN elementId(n) as id, labels(n)[0] as type, 
           COALESCE(n.filename, n.topic, n.source, substring(n.content, 0, 40), n.id, labels(n)[0]) as label
    LIMIT $limit
    """
    driver = get_graph_instance()
    result = driver.query(query, {"term": search_term, "limit": limit})
    return result if result else []

def get_graph_sample(
    node_types: List[str],
    rel_types: List[str],
    limit: int = 50,
    focus_node_id: str = "",
):
    """Fetch a sample of nodes and relationships for visualization."""
    all_node_types = ["Document", "DocumentChunk", "AppUser", "Session", "Message"]
    all_rel_types = ["HAS_CHUNK", "HAS_SESSION", "HAS_MESSAGE", "LAST_MESSAGE"]

    # Assign defaults if explicitly given empty lists
    if not node_types:
        node_types = all_node_types
    if not rel_types:
        rel_types = all_rel_types

    nodes = []
    edges = []
    node_ids = set()
    
    driver = get_graph_instance()

    if focus_node_id:
        query = """
        MATCH path = (root)-[r*1..2]-(m)
        WHERE elementId(root) = $focus_node_id
        AND ALL(n IN nodes(path) WHERE labels(n)[0] IN $node_types)
        AND ALL(rel IN relationships(path) WHERE type(rel) IN $rel_types)
        WITH relationships(path) as rels
        UNWIND rels as r
        WITH startNode(r) as n, r, endNode(r) as m
        LIMIT $limit
        RETURN 
            elementId(n) as source_id, 
            labels(n)[0] as source_label,
            properties(n) as source_props,
            CASE labels(n)[0]
                WHEN 'Document' THEN COALESCE(n.filename, 'Doc ' + elementId(n))
                WHEN 'DocumentChunk' THEN 'Chunk ' + COALESCE(toString(n.chunk_index), elementId(n))
                WHEN 'AppUser' THEN 'User ' + COALESCE(n.id, elementId(n))
                WHEN 'Session' THEN COALESCE(n.topic, 'Session ' + elementId(n))
                WHEN 'Message' THEN COALESCE(substring(n.content, 0, 30), 'Message ' + elementId(n))
                ELSE elementId(n)
            END as source_name,
            elementId(m) as target_id,
            labels(m)[0] as target_label,
            properties(m) as target_props,
            CASE labels(m)[0]
                WHEN 'Document' THEN COALESCE(m.filename, 'Doc ' + elementId(m))
                WHEN 'DocumentChunk' THEN 'Chunk ' + COALESCE(toString(m.chunk_index), elementId(m))
                WHEN 'AppUser' THEN 'User ' + COALESCE(m.id, elementId(m))
                WHEN 'Session' THEN COALESCE(m.topic, 'Session ' + elementId(m))
                WHEN 'Message' THEN COALESCE(substring(m.content, 0, 30), 'Message ' + elementId(m))
                ELSE elementId(m)
            END as target_name,
            type(r) as rel_type
        """
        params = {
            "node_types": node_types,
            "rel_types": rel_types,
            "limit": limit * 3,
            "focus_node_id": focus_node_id,
        }
    else:
        query = """
        MATCH (n)-[r]->(m)
        WHERE (labels(n)[0] IN $node_types OR labels(m)[0] IN $node_types)
          AND type(r) IN $rel_types
        WITH n, r, m
        LIMIT $limit
        RETURN 
            elementId(n) as source_id, 
            labels(n)[0] as source_label,
            properties(n) as source_props,
            CASE labels(n)[0]
                WHEN 'Document' THEN COALESCE(n.filename, 'Doc ' + elementId(n))
                WHEN 'DocumentChunk' THEN 'Chunk ' + COALESCE(toString(n.chunk_index), elementId(n))
                WHEN 'AppUser' THEN 'User ' + COALESCE(n.id, elementId(n))
                WHEN 'Session' THEN COALESCE(n.topic, 'Session ' + elementId(n))
                WHEN 'Message' THEN COALESCE(substring(n.content, 0, 30), 'Message ' + elementId(n))
                ELSE elementId(n)
            END as source_name,
            elementId(m) as target_id,
            labels(m)[0] as target_label,
            properties(m) as target_props,
            CASE labels(m)[0]
                WHEN 'Document' THEN COALESCE(m.filename, 'Doc ' + elementId(m))
                WHEN 'DocumentChunk' THEN 'Chunk ' + COALESCE(toString(m.chunk_index), elementId(m))
                WHEN 'AppUser' THEN 'User ' + COALESCE(m.id, elementId(m))
                WHEN 'Session' THEN COALESCE(m.topic, 'Session ' + elementId(m))
                WHEN 'Message' THEN COALESCE(substring(m.content, 0, 30), 'Message ' + elementId(m))
                ELSE elementId(m)
            END as target_name,
            type(r) as rel_type
        """
        params = {"node_types": node_types, "rel_types": rel_types, "limit": limit * 2}

    results = driver.query(query, params)

    if results:
        for r in results:
            if r["source_id"] not in node_ids:
                nodes.append({
                    "id": r["source_id"],
                    "label": r["source_name"][:30] if r["source_name"] else str(r["source_id"]),
                    "type": r["source_label"],
                    "title": r["source_name"],
                    "properties": r.get("source_props", {}),
                })
                node_ids.add(r["source_id"])

            if r["target_id"] not in node_ids:
                nodes.append({
                    "id": r["target_id"],
                    "label": r["target_name"][:30] if r["target_name"] else str(r["target_id"]),
                    "type": r["target_label"],
                    "title": r["target_name"],
                    "properties": r.get("target_props", {}),
                })
                node_ids.add(r["target_id"])

            edges.append({
                "from": r["source_id"],
                "to": r["target_id"],
                "label": r["rel_type"],
                "title": r["rel_type"],
            })

            if len(nodes) >= limit * 1.5:
                break

    # Convert datetime properties inside node properties to strings
    for node in nodes:
        props = node.get("properties", {})
        for k, v in list(props.items()):
            if "date" in k.lower() or "time" in k.lower() or hasattr(v, "iso_format") or hasattr(v, "isoformat"):
                props[k] = str(v)

    return {"nodes": nodes, "edges": edges}
