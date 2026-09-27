"""Memory management for chat sessions and users in Neo4j."""

import logging
from datetime import datetime

from langchain_neo4j import Neo4jChatMessageHistory

from setup.init_config import NEO4J_URL, NEO4J_USERNAME, NEO4J_PASSWORD, get_graph_instance

logger = logging.getLogger(__name__)

# Cypher Queries for Sessions, Messages, and Users
_MERGE_SESSION = "MERGE (s:Session {id: $session_id})"

_ADD_USER_MESSAGE = """
MATCH (s:Session {id: $session_id})
OPTIONAL MATCH (s)-[old_rel:LAST_MESSAGE]->(old_msg:Message)
DELETE old_rel
WITH s
CREATE (m:Message {content: $content, type: 'user', created_at: $ts})
CREATE (s)-[:LAST_MESSAGE]->(m)
CREATE (s)-[:HAS_MESSAGE]->(m)
"""

_FALLBACK_USER_MESSAGE = """
MATCH (s:Session {id: $session_id})-[:LAST_MESSAGE]->(m:Message)
WHERE NOT EXISTS((s)-[:HAS_MESSAGE]->(m))
SET m.created_at = $ts, m.type = 'user', m.content = $content
MERGE (s)-[:HAS_MESSAGE]->(m)
"""

_ADD_AI_MESSAGE = """
MATCH (s:Session {id: $session_id})
OPTIONAL MATCH (s)-[old_rel:LAST_MESSAGE]->(old_msg:Message)
DELETE old_rel
WITH s
CREATE (m:Message {content: $content, type: 'assistant', thought: $thought, created_at: $ts})
CREATE (s)-[:LAST_MESSAGE]->(m)
CREATE (s)-[:HAS_MESSAGE]->(m)
"""

_FALLBACK_AI_MESSAGE = """
MATCH (s:Session {id: $session_id})-[:LAST_MESSAGE]->(m:Message)
WHERE NOT EXISTS((s)-[:HAS_MESSAGE]->(m))
SET m.thought = $thought, m.created_at = $ts, m.type = 'assistant', m.content = $content
MERGE (s)-[:HAS_MESSAGE]->(m)
"""

_REPAIR_MESSAGES = """
MATCH (s:Session)-[:LAST_MESSAGE]->(m:Message)
WHERE NOT EXISTS((s)-[:HAS_MESSAGE]->(m))
MERGE (s)-[:HAS_MESSAGE]->(m)
RETURN count(m) AS repaired_count
"""

_LINK_SESSION_TOPIC = """
MERGE (u:AppUser {id: $user_id})
MERGE (s:Session {id: $session_id})
ON CREATE SET s.topic = $topic
MERGE (u)-[:HAS_SESSION]->(s)
"""

_LINK_SESSION_NO_TOPIC = """
MERGE (u:AppUser {id: $user_id})
MERGE (s:Session {id: $session_id})
MERGE (u)-[:HAS_SESSION]->(s)
"""

_GET_USER_SESSIONS = """
MATCH (u:AppUser {id: $user_id})-[:HAS_SESSION]->(s:Session)
OPTIONAL MATCH (s)-[:HAS_MESSAGE]->(m:Message)
WITH s, m
ORDER BY coalesce(m.created_at, elementId(m)) DESC
WITH s, head(collect(m)) AS last_msg
RETURN s.id AS session_id,
       last_msg.content AS last_message,
       coalesce(last_msg.created_at, "0") AS last_activity
ORDER BY last_activity ASC
LIMIT 100
"""

_GET_ALL_USERS = """
MATCH (u:AppUser)
RETURN u.id AS user_id
LIMIT 1000
"""

_DELETE_SESSION = """
MATCH (s:Session {id: $session_id})
OPTIONAL MATCH (s)-[*1..50]-(m:Message)
DETACH DELETE m, s
"""

_DELETE_USER = """
MATCH (u:AppUser {id: $user_id})
OPTIONAL MATCH (u)-[:HAS_SESSION]->(s:Session)
OPTIONAL MATCH (s)-[*1..50]-(m:Message)
DETACH DELETE m, s, u
"""

_UPDATE_LAST_AI_MESSAGE = """
MATCH (s:Session {id: $session_id})-[:LAST_MESSAGE]->(m:Message)
SET m.content = $new_content
RETURN elementId(m) AS message_id
"""

def get_chat_history(session_id: str):
    """Args: session_id: Session identifier."""
    try:
        return Neo4jChatMessageHistory(
            session_id=session_id,
            url=NEO4J_URL,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
        )
    except Exception as e:
        logger.error("Error getting chat history for session %s: %s", session_id, e)
        class EmptyHistory:
            messages = []
        return EmptyHistory()

def _ts() -> str:
    """Args: None."""
    return datetime.now().isoformat()

def add_user_message_to_session(session_id: str, content: str) -> None:
    """Args: session_id: Session ID, content: Message content."""
    graph = get_graph_instance()
    try:
        graph.query(_MERGE_SESSION, params={"session_id": session_id})
        graph.query(
            _ADD_USER_MESSAGE,
            params={
                "session_id": session_id,
                "content": content,
                "ts": _ts(),
            },
        )
        logger.info("Added user message to session %s", session_id)
    except Exception as e:
        logger.error("Error adding user message to session %s: %s", session_id, e)
        try:
            get_chat_history(session_id).add_user_message(content)
            graph.query(
                _FALLBACK_USER_MESSAGE,
                params={
                    "session_id": session_id,
                    "content": content,
                    "ts": _ts(),
                },
            )
            logger.info("Fallback succeeded for session %s", session_id)
        except Exception as fe:
            logger.error("Fallback also failed for session %s: %s", session_id, fe)

def add_ai_message_to_session(session_id: str, content: str, thought: str) -> None:
    """Args: session_id: Session ID, content: Message content, thought: Reasoning trace."""
    graph = get_graph_instance()
    try:
        graph.query(_MERGE_SESSION, params={"session_id": session_id})
        graph.query(
            _ADD_AI_MESSAGE,
            params={
                "session_id": session_id,
                "content": content,
                "thought": thought or None,
                "ts": _ts(),
            },
        )
        logger.info("Added AI message to session %s (thought len: %d)", session_id, len(thought) if thought else 0)
    except Exception as e:
        logger.error("Error adding AI message to session %s: %s", session_id, e)
        try:
            get_chat_history(session_id).add_ai_message(content)
            graph.query(
                _FALLBACK_AI_MESSAGE,
                params={
                    "session_id": session_id,
                    "content": content,
                    "thought": thought or None,
                    "ts": _ts(),
                },
            )
            logger.info("Fallback succeeded for session %s", session_id)
        except Exception as fe:
            logger.error("Fallback also failed for session %s: %s", session_id, fe)

def repair_missing_has_message_relationships() -> int:
    """Args: None."""
    try:
        result = get_graph_instance().query(_REPAIR_MESSAGES)
        repaired = result[0]["repaired_count"] if result else 0
        if repaired > 0:
            logger.info("✅ Repaired %d missing HAS_MESSAGE relationships", repaired)
        return repaired
    except Exception as e:
        logger.error("Error repairing HAS_MESSAGE relationships: %s", e)
        return 0

def link_session_to_user(session_id: str, user_id: str, topic: str) -> None:
    """Args: session_id: Session ID, user_id: User ID, topic: Session topic."""
    if not user_id:
        return
    try:
        graph = get_graph_instance()
        if topic:
            cypher = _LINK_SESSION_TOPIC
            params = {
                "user_id": user_id,
                "session_id": session_id,
                "topic": topic,
            }
        else:
            cypher = _LINK_SESSION_NO_TOPIC
            params = {
                "user_id": user_id,
                "session_id": session_id,
            }
        graph.query(cypher, params=params)
        logger.debug("Linked session %s to user %s%s", session_id, user_id, f" (topic: {topic})" if topic else "")
    except Exception as e:
        logger.error("Error linking session to user: %s", e)

def get_user_sessions(user_id: str) -> list:
    """Args: user_id: User ID."""
    try:
        return get_graph_instance().query(_GET_USER_SESSIONS, params={"user_id": user_id})
    except Exception as e:
        logger.error("Error getting sessions for user %s: %s", user_id, e)
        return []

def get_all_users() -> list:
    """Args: None."""
    try:
        result = get_graph_instance().query(_GET_ALL_USERS)
        return [r["user_id"] for r in result]
    except Exception as e:
        logger.error("Error getting all users: %s", e)
        return []

def delete_session(session_id: str) -> None:
    """Args: session_id: Session ID."""
    try:
        get_graph_instance().query(_DELETE_SESSION, params={"session_id": session_id})
        logger.info("Session %s deleted", session_id)
    except Exception as e:
        logger.error("Error deleting session %s: %s", session_id, e)

def delete_user(user_id: str) -> None:
    """Args: user_id: User ID."""
    try:
        get_graph_instance().query(_DELETE_USER, params={"user_id": user_id})
        logger.info("User %s and all data deleted", user_id)
    except Exception as e:
        logger.error("Error deleting user %s: %s", user_id, e)

def update_last_ai_message(session_id: str, new_content: str) -> None:
    """Args: session_id: Session ID, new_content: Replacement text."""
    try:
        get_graph_instance().query(
            _UPDATE_LAST_AI_MESSAGE,
            params={
                "session_id": session_id,
                "new_content": new_content,
            },
        )
        logger.info("Updated last AI message for session %s", session_id)
    except Exception as e:
        logger.error("Error updating last AI message: %s", e)
