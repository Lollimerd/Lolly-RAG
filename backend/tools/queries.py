# Cypher: multi-index hybrid search (vector, fulltext, text) over DocumentChunk & Document nodes

_SEARCH_FILTER_RETURN = """WITH node AS chunk, max(score) AS max_score, collect(DISTINCT match_type) AS match_types
MATCH (d:Document)-[:HAS_CHUNK]->(chunk)
WITH chunk, max_score, match_types, d, coalesce(chunk.communityId, chunk.CommunityId, d.communityId, d.CommunityId) AS raw_comm_id
WHERE ($str_community_ids IS NULL OR size($str_community_ids) = 0 OR (raw_comm_id IS NOT NULL AND (toString(raw_comm_id) IN $str_community_ids OR (raw_comm_id IS :: LIST<ANY> AND ANY(c IN raw_comm_id WHERE toString(c) IN $str_community_ids)))))
  AND ($target_file_types IS NULL OR size($target_file_types) = 0 OR toLower(d.file_type) IN $target_file_types)
  AND ($target_filename IS NULL OR $target_filename = '' OR toLower(d.filename) CONTAINS toLower($target_filename))
  AND ($target_sheet_name IS NULL OR $target_sheet_name = '' OR (chunk.source IS NOT NULL AND toLower(chunk.source) CONTAINS toLower($target_sheet_name)))
RETURN chunk.id AS chunk_id, chunk.content AS content, chunk.chunk_index AS chunk_index, chunk.source AS source,
       raw_comm_id AS community_id, d.id AS doc_id, d.filename AS filename, d.file_type AS file_type,
       d.upload_date AS upload_date, d.user_id AS user_id, d.chunk_count AS chunk_count, d.description AS description,
       d.file_hash AS file_hash, max_score AS score, match_types AS match_types
ORDER BY score DESC LIMIT $top_k"""

HYBRID_DOCUMENT_SEARCH_QUERY = f"""CALL {{
    CALL db.index.vector.queryNodes('DocumentChunk_index', $top_k, $query_embedding) YIELD node, score RETURN node, score, 'vector' AS match_type
    UNION
    CALL db.index.fulltext.queryNodes('DocumentChunk_keyword_index', $fulltext_query, {{limit: $top_k}}) YIELD node, score RETURN node, score, 'fulltext_chunk' AS match_type
    UNION
    CALL db.index.fulltext.queryNodes('Document_keyword_index', $fulltext_query, {{limit: $top_k}}) YIELD node AS doc, score MATCH (doc)-[:HAS_CHUNK]->(node:DocumentChunk) RETURN node, score * 0.95 AS score, 'fulltext_doc' AS match_type
    UNION
    MATCH (node:DocumentChunk) WHERE node.source IS NOT NULL AND (toLower(node.source) CONTAINS toLower($question_clean) OR toLower($question_clean) CONTAINS toLower(node.source)) RETURN node, 1.0 AS score, 'text_chunk_source' AS match_type
    UNION
    MATCH (doc:Document)-[:HAS_CHUNK]->(node:DocumentChunk)
    WHERE ((doc.filename IS NOT NULL AND (toLower(doc.filename) CONTAINS toLower($question_clean) OR toLower($question_clean) CONTAINS toLower(doc.filename)))
        OR (doc.description IS NOT NULL AND toLower(doc.description) CONTAINS toLower($question_clean))
        OR (doc.source IS NOT NULL AND toLower(doc.source) CONTAINS toLower($question_clean))
        OR (doc.file_type IS NOT NULL AND size(doc.file_type) > 1 AND toLower($question_clean) CONTAINS toLower(doc.file_type)))
    RETURN node, 0.9 AS score, 'text_doc_metadata' AS match_type
}}
{_SEARCH_FILTER_RETURN}"""

FALLBACK_DOCUMENT_SEARCH_QUERY = f"""CALL {{
    CALL db.index.vector.queryNodes('DocumentChunk_index', $top_k, $query_embedding) YIELD node, score RETURN node, score, 'vector' AS match_type
    UNION
    CALL db.index.fulltext.queryNodes('DocumentChunk_keyword_index', $fulltext_query, {{limit: $top_k}}) YIELD node, score RETURN node, score, 'fulltext_chunk' AS match_type
    UNION
    MATCH (node:DocumentChunk) WHERE node.source IS NOT NULL AND (toLower(node.source) CONTAINS toLower($question_clean) OR toLower($question_clean) CONTAINS toLower(node.source)) RETURN node, 1.0 AS score, 'text_chunk_source' AS match_type
    UNION
    MATCH (doc:Document)-[:HAS_CHUNK]->(node:DocumentChunk)
    WHERE ((doc.filename IS NOT NULL AND (toLower(doc.filename) CONTAINS toLower($question_clean) OR toLower($question_clean) CONTAINS toLower(doc.filename)))
        OR (doc.description IS NOT NULL AND toLower(doc.description) CONTAINS toLower($question_clean))
        OR (doc.file_type IS NOT NULL AND size(doc.file_type) > 1 AND toLower($question_clean) CONTAINS toLower(doc.file_type)))
    RETURN node, 0.9 AS score, 'text_doc_metadata' AS match_type
}}
{_SEARCH_FILTER_RETURN}"""