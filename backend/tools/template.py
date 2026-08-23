CYPHER_GENERATION_TEMPLATE = """Task: Generate an accurate Cypher query to retrieve relevant context from a StackOverflow developer knowledge graph in Neo4j.

Graph Schema:
{schema}

======================================================================
STRICT INSTRUCTIONS & RULES:
======================================================================
1. OUTPUT FORMAT:
   - Output ONLY the raw Cypher query statement.
   - Do NOT include markdown code fences (no ```cypher or ```).
   - Do NOT include any explanations, comments, greetings, or text before/after the query.
   - Never generate mutating queries (NO CREATE, MERGE, SET, DELETE, DROP, DETACH).

2. INDEX SELECTION (Always start with a Vector Search using `$question_embedding`):
   Choose the best entry index based on the user's intent:
   - 'Question_index' : For conceptual, "how-to", architectural, or broad topic questions (targets Question nodes).
   - 'Answer_index'   : For error messages, stack traces, exceptions, code fixes, or specific solutions (targets Answer nodes).
   - 'Tag_index'      : For technology overview, library exploration, or "what topics exist about X" (targets Tag nodes).
   - 'User_index'     : For finding expert contributors or answers by specific users (targets User nodes).

3. RELATIONSHIP TRAVERSAL & RESILIENCY:
   - Question & Answer relationships:
     (:User)-[:ASKED]->(:Question)
     (:User)-[:PROVIDED]->(:Answer)
     (:Answer)-[:ANSWERS]->(:Question)
     (:Question)-[:TAGGED]->(:Tag)
   - Use OPTIONAL MATCH for secondary relationships (answers, tags, users) so missing nodes do NOT eliminate valid results.
   - When filtering by community clustering between questions and answers, use null-safe comparison:
     (q.CommunityId IS NULL OR a.CommunityId IS NULL OR ANY(cid IN q.CommunityId WHERE cid IN a.CommunityId))
   - When answers are retrieved, prioritize accepted or positive-scoring answers (a.is_accepted = true OR a.score > 0).

4. MANDATORY RETURN COLUMN ALIASES (CRITICAL FOR DOWNSTREAM RERANKER):
   You MUST use these exact column aliases in the RETURN clause:
   - `question_title` : Title of the question (e.g., q.title AS question_title)
   - `question_body`  : Body content of the question (e.g., q.body AS question_body)
   - `answer_body`    : Body content of the answer (e.g., a.body AS answer_body)
   - `answer_score`   : Integer score of the answer (e.g., a.score AS answer_score)
   - `is_accepted`    : Boolean accepted flag of the answer (e.g., a.is_accepted AS is_accepted)
   - `tags`           : Distinct list of tag names (e.g., collect(DISTINCT t.name) AS tags)
   - `answered_by`    : Display name of the answering user (e.g., u.display_name AS answered_by)
   - `asked_by`       : Display name of the question author (e.g., u.display_name AS asked_by)
   - `score`          : Vector similarity score yielded from index query (e.g., score)

5. AGGREGATION & ORDERING:
   - When using aggregations (like `collect(DISTINCT t.name)`), any variable referenced in `ORDER BY` MUST be included in `RETURN`.
   - Always sort primarily by vector score: `ORDER BY score DESC` (and optionally `a.score DESC` or `q.score DESC`).
   - Limit results appropriately (default `LIMIT 50`).

======================================================================
CYPHER EXAMPLES:
======================================================================

# Example 1: Conceptual or "How-To" Question (Searches Question_index with optional accepted/top answers)
CALL db.index.vector.queryNodes('Question_index', 50, $question_embedding) YIELD node AS q, score
OPTIONAL MATCH (a:Answer)-[:ANSWERS]->(q)
  WHERE (q.CommunityId IS NULL OR a.CommunityId IS NULL OR ANY(cid IN q.CommunityId WHERE cid IN a.CommunityId))
    AND (a.is_accepted = true OR a.score > 0)
OPTIONAL MATCH (q)-[:TAGGED]->(t:Tag)
OPTIONAL MATCH (u:User)-[:PROVIDED]->(a)
RETURN q.title                  AS question_title,
       q.body                   AS question_body,
       a.body                   AS answer_body,
       a.score                  AS answer_score,
       a.is_accepted            AS is_accepted,
       collect(DISTINCT t.name) AS tags,
       u.display_name           AS answered_by,
       score
ORDER BY score DESC, a.score DESC
LIMIT 50


# Example 2: Error Message, Code Trace, Exception, or Solution Search (Searches Answer_index)
CALL db.index.vector.queryNodes('Answer_index', 50, $question_embedding) YIELD node AS a, score
MATCH (q:Question)<-[:ANSWERS]-(a)
WHERE (q.CommunityId IS NULL OR a.CommunityId IS NULL OR ANY(cid IN q.CommunityId WHERE cid IN a.CommunityId))
OPTIONAL MATCH (q)-[:TAGGED]->(t:Tag)
OPTIONAL MATCH (u:User)-[:PROVIDED]->(a)
RETURN q.title                  AS question_title,
       q.body                   AS question_body,
       a.body                   AS answer_body,
       a.score                  AS answer_score,
       a.is_accepted            AS is_accepted,
       collect(DISTINCT t.name) AS tags,
       u.display_name           AS answered_by,
       score
ORDER BY score DESC, a.score DESC
LIMIT 50


# Example 3: Tag / Specific Technology Filtered Search (e.g., Docker, Python, Neo4j)
CALL db.index.vector.queryNodes('Question_index', 50, $question_embedding) YIELD node AS q, score
MATCH (q)-[:TAGGED]->(matchedTag:Tag)
WHERE matchedTag.name IN ['docker', 'python', 'neo4j', 'fastapi']
OPTIONAL MATCH (a:Answer)-[:ANSWERS]->(q)
  WHERE (q.CommunityId IS NULL OR a.CommunityId IS NULL OR ANY(cid IN q.CommunityId WHERE cid IN a.CommunityId))
    AND (a.is_accepted = true OR a.score > 0)
OPTIONAL MATCH (q)-[:TAGGED]->(allTags:Tag)
OPTIONAL MATCH (u:User)-[:PROVIDED]->(a)
RETURN q.title                    AS question_title,
       q.body                     AS question_body,
       a.body                     AS answer_body,
       a.score                    AS answer_score,
       a.is_accepted              AS is_accepted,
       collect(DISTINCT allTags.name) AS tags,
       u.display_name             AS answered_by,
       score
ORDER BY score DESC, a.score DESC
LIMIT 50


# Example 4: Topic & Library Exploration / Tag-First Search (Searches Tag_index)
CALL db.index.vector.queryNodes('Tag_index', 20, $question_embedding) YIELD node AS t, score
MATCH (q:Question)-[:TAGGED]->(t)
OPTIONAL MATCH (a:Answer)-[:ANSWERS]->(q)
  WHERE (a.is_accepted = true OR a.score > 0)
OPTIONAL MATCH (u:User)-[:ASKED]->(q)
RETURN q.title                  AS question_title,
       q.body                   AS question_body,
       q.score                  AS question_score,
       a.body                   AS answer_body,
       a.score                  AS answer_score,
       a.is_accepted            AS is_accepted,
       t.name                   AS tag,
       u.display_name           AS asked_by,
       score
ORDER BY score DESC, q.score DESC
LIMIT 50


# Example 5: Expert User or Author Search (Searches User_index)
CALL db.index.vector.queryNodes('User_index', 20, $question_embedding) YIELD node AS u, score
OPTIONAL MATCH (u)-[:PROVIDED]->(a:Answer)-[:ANSWERS]->(q:Question)
OPTIONAL MATCH (q)-[:TAGGED]->(t:Tag)
RETURN u.display_name           AS answered_by,
       u.reputation             AS reputation,
       q.title                  AS question_title,
       q.body                   AS question_body,
       a.body                   AS answer_body,
       a.score                  AS answer_score,
       a.is_accepted            AS is_accepted,
       collect(DISTINCT t.name) AS tags,
       score
ORDER BY score DESC, u.reputation DESC
LIMIT 50

======================================================================
Question to answer:
{question}"""