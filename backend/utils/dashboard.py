from typing import List, Dict, Any, Optional
from setup.init_config import get_graph_instance
import logging

logger = logging.getLogger(__name__)

def get_database_summary():
    """Get summary statistics from the database."""
    summary_query = """
    MATCH (q:Question)
    WITH count(q) as total_questions
    MATCH (t:Tag)
    WITH total_questions, count(t) as total_tags
    MATCH (a:Answer)
    WITH total_questions, total_tags, count(a) as total_answers
    MATCH (u:User)
    WITH total_questions, total_tags, total_answers, count(u) as total_users
    MATCH (log:ImportLog)
    WITH total_questions, total_tags, total_answers, total_users, count(log) as total_imports
    MATCH (log:ImportLog)
    WITH total_questions, total_tags, total_answers, total_users, total_imports, 
         max(log.timestamp) as last_import
    RETURN total_questions, total_tags, total_answers, total_users, total_imports, last_import
    """
    driver = get_graph_instance()
    result = driver.query(summary_query)
    if result and len(result) > 0:
        res = result[0]
        # Neo4j specific time might need str conversion
        if res.get("last_import"):
            res["last_import"] = str(res["last_import"])
        return res
    return {
        "total_questions": 0,
        "total_tags": 0,
        "total_answers": 0,
        "total_users": 0,
        "total_imports": 0,
        "last_import": None,
    }

def get_import_history(limit: int = 20):
    """Get recent import history from ImportLog nodes."""
    history_query = """
    MATCH (log:ImportLog)
    RETURN log.id as id, log.timestamp as timestamp, log.total_questions as questions,
           log.total_tags as tags, log.total_pages as pages, log.tags_list as tags_list,
           coalesce(log.site, 'stackoverflow') as site
    ORDER BY log.timestamp DESC
    LIMIT $limit
    """
    driver = get_graph_instance()
    result = driver.query(history_query, {"limit": limit})
    if result:
        for r in result:
            if r.get("timestamp"):
                r["timestamp"] = str(r["timestamp"])
        return result
    return []

def get_entity_counts():
    """Get counts for all entity types and relationships in the database."""
    node_counts_query = """
    CALL {
        MATCH (q:Question) RETURN 'Question' as label, count(q) as count
        UNION ALL
        MATCH (a:Answer) RETURN 'Answer' as label, count(a) as count
        UNION ALL
        MATCH (t:Tag) RETURN 'Tag' as label, count(t) as count
        UNION ALL
        MATCH (u:User) RETURN 'User' as label, count(u) as count
        UNION ALL
        MATCH (i:ImportLog) RETURN 'ImportLog' as label, count(i) as count
    }
    RETURN label, count
    """

    rel_counts_query = """
    CALL {
        MATCH ()-[r:TAGGED]->() RETURN 'TAGGED' as type, count(r) as count
        UNION ALL
        MATCH ()-[r:ANSWERS]->() RETURN 'ANSWERS' as type, count(r) as count
        UNION ALL
        MATCH ()-[r:PROVIDED]->() RETURN 'PROVIDED' as type, count(r) as count
        UNION ALL
        MATCH ()-[r:ASKED]->() RETURN 'ASKED' as type, count(r) as count
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
    """Search for nodes by title, name, or display_name."""
    query = """
    MATCH (n)
    WHERE n.title CONTAINS $term OR n.name CONTAINS $term OR n.display_name CONTAINS $term
    RETURN elementId(n) as id, labels(n)[0] as type, 
           COALESCE(n.title, n.name, n.display_name) as label
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
    all_node_types = ["Question", "Answer", "Tag", "User"]
    all_rel_types = ["TAGGED", "ANSWERS", "PROVIDED", "ASKED"]

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
        And ALL(rel IN relationships(path) WHERE type(rel) IN $rel_types)
        WITH relationships(path) as rels
        UNWIND rels as r
        WITH startNode(r) as n, r, endNode(r) as m
        LIMIT $limit
        RETURN 
            elementId(n) as source_id, 
            labels(n)[0] as source_label,
            properties(n) as source_props,
            CASE labels(n)[0]
                WHEN 'Question' THEN COALESCE(n.title, 'Question ' + elementId(n))
                WHEN 'Answer' THEN 'Answer ' + elementId(n)
                WHEN 'Tag' THEN n.name
                WHEN 'User' THEN COALESCE(n.display_name, 'User ' + elementId(n))
                ELSE elementId(n)
            END as source_name,
            elementId(m) as target_id,
            labels(m)[0] as target_label,
            properties(m) as target_props,
            CASE labels(m)[0]
                WHEN 'Question' THEN COALESCE(m.title, 'Question ' + elementId(m))
                WHEN 'Answer' THEN 'Answer ' + elementId(m)
                WHEN 'Tag' THEN n.name
                WHEN 'User' THEN COALESCE(n.display_name, 'User ' + elementId(n))
                ELSE elementId(m)
            END as target_name,
            type(r) as rel_type
        """
        params = {
            "node_types": node_types,
            "rel_types": rel_types,
            "limit": limit * 3,
            "focus_node_id": str(focus_node_id),
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
                WHEN 'Question' THEN COALESCE(n.title, 'Question ' + elementId(n))
                WHEN 'Answer' THEN 'Answer ' + elementId(n)
                WHEN 'Tag' THEN n.name
                WHEN 'User' THEN COALESCE(n.display_name, 'User ' + elementId(n))
                ELSE elementId(n)
            END as source_name,
            elementId(m) as target_id,
            labels(m)[0] as target_label,
            properties(m) as target_props,
            CASE labels(m)[0]
                WHEN 'Question' THEN COALESCE(m.title, 'Question ' + elementId(m))
                WHEN 'Answer' THEN 'Answer ' + elementId(m)
                WHEN 'Tag' THEN m.name
                WHEN 'User' THEN COALESCE(m.display_name, 'User ' + elementId(m))
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

    # Important: Neo4j datetime properties might not be directly serializable natively
    # So we should convert datetime propertes inside node properties to strings
    for node in nodes:
        props = node.get("properties", {})
        for k, v in list(props.items()):
            if "date" in k.lower() or "time" in k.lower() or hasattr(v, "iso_format") or hasattr(v, "isoformat"):
                props[k] = str(v)

    return {"nodes": nodes, "edges": edges}
