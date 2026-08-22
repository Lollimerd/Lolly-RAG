

CYPHER_GENERATION_TEMPLATE = """Task: Generate a Cypher statement to query a StackOverflow knowledge graph.

Instructions:
- Use ONLY the relationship types and node properties defined in the schema below.
- Output ONLY the raw Cypher statement — no explanation, no markdown, no apologies.
- A Q&A pair is NOT required — you may retrieve Questions, Answers, Tags, or Users independently
  based on what best fits the user's question.
- Choose the most appropriate vector index from the 4 available options:
    • 'Question_index'  — for conceptual, how-to, or general topic questions (searches Question nodes)
    • 'Answer_index'    — for error messages, code snippets, or solution-focused queries (searches Answer nodes)
    • 'Tag_index'       — for topic/technology browsing or "what questions exist about X" queries (searches Tag nodes)
    • 'User_index'      — for finding expert users, contributors, or "who answered X" queries (searches User nodes)
- ALWAYS start with a vector index search using `$question_embedding`. E.g.:
    CALL db.index.vector.queryNodes('Question_index', 50, $question_embedding) YIELD node AS q, score
- Use OPTIONAL MATCH for all supplementary relationships (answers, tags, users) so that
  missing branches never eliminate otherwise valid results.
- When traversing Question-Answer pairs, filter by shared community membership using:
    ANY(cid IN q.CommunityId WHERE cid IN a.CommunityId)
  This ensures answers and questions belong to the same community cluster.
- Prefer accepted or high-scoring answers (a.is_accepted = true OR a.score > 0) when answers are part of the query.
- Limit results (LIMIT 25 is a good default).
- Return enough fields for a rich answer: question title, question body, answer body,
  answer score, whether the answer is accepted, relevant tag names, and vector search similarity score (`score`).
- CRITICAL: In Cypher, if a query uses aggregation (such as `collect(DISTINCT...)`) in the `RETURN` clause, any variable used in the `ORDER BY` clause (like `score`) MUST be explicitly included in the `RETURN` clause. Make sure to project `score` in your `RETURN` statement if you order by it!

Schema:
{schema}

Examples:

# Example 1: Find accepted answers for questions about a technical topic (e.g., Python async)
# Use Question_index when the user is asking a conceptual or how-to question
CALL db.index.vector.queryNodes('Question_index', 50, $question_embedding) YIELD node AS q, score
MATCH (a:Answer)-[:ANSWERS]->(q:Question)
WHERE ANY(cid IN q.CommunityId WHERE cid IN a.CommunityId)
  AND (a.is_accepted = true OR a.score > 0)
OPTIONAL MATCH (q)-[:TAGGED]->(t:Tag)
OPTIONAL MATCH (u:User)-[:PROVIDED]->(a)
RETURN q.title          AS question_title,
       q.body           AS question_body,
       a.body           AS answer_body,
       a.score          AS answer_score,
       a.is_accepted    AS is_accepted,
       collect(DISTINCT t.name) AS tags,
       u.display_name   AS answered_by,
       score
ORDER BY score DESC, a.score DESC
LIMIT 50


# Example 2: Search by answer content — for error messages, stack traces, or code fix queries
# Use Answer_index when the user pastes an error or asks about a specific solution/snippet
CALL db.index.vector.queryNodes('Answer_index', 50, $question_embedding) YIELD node AS a, score
MATCH (q:Question)<-[:ANSWERS]-(a)
WHERE ANY(cid IN q.CommunityId WHERE cid IN a.CommunityId)
  AND (a.is_accepted = true OR a.score > 0)
OPTIONAL MATCH (q)-[:TAGGED]->(t:Tag)
OPTIONAL MATCH (u:User)-[:PROVIDED]->(a)
RETURN q.title          AS question_title,
       q.body           AS question_body,
       a.body           AS answer_body,
       a.score          AS answer_score,
       a.is_accepted    AS is_accepted,
       collect(DISTINCT t.name) AS tags,
       u.display_name   AS answered_by,
       score
ORDER BY score DESC, a.score DESC
LIMIT 50

# Example 3: Filter by a specific tag or technology name for targeted topic search
# Useful when the user explicitly mentions a technology (e.g., "in Docker", "using Python")
CALL db.index.vector.queryNodes('Question_index', 50, $question_embedding) YIELD node AS q, score
MATCH (a:Answer)-[:ANSWERS]->(q:Question)
MATCH (q)-[:TAGGED]->(t:Tag)
WHERE ANY(cid IN q.CommunityId WHERE cid IN a.CommunityId)
  AND (a.is_accepted = true OR a.score > 0)
  AND t.name IN ['docker', 'python', 'linux']
OPTIONAL MATCH (q)-[:TAGGED]->(allTags:Tag)
OPTIONAL MATCH (u:User)-[:PROVIDED]->(a)
RETURN q.title          AS question_title,
       q.body           AS question_body,
       a.body           AS answer_body,
       a.score          AS answer_score,
       a.is_accepted    AS is_accepted,
       collect(DISTINCT allTags.name) AS tags,
       u.display_name   AS answered_by,
       score
ORDER BY score DESC, a.score DESC
LIMIT 50

# Example 4: Tag-first traversal — browse questions by topic WITHOUT requiring an answer match
# Use Tag_index when the user asks "what topics/questions exist about X" or needs a broad overview
# Questions without answers are still returned (OPTIONAL MATCH on answers)
CALL db.index.vector.queryNodes('Tag_index', 20, $question_embedding) YIELD node AS t, score
MATCH (q:Question)-[:TAGGED]->(t)
OPTIONAL MATCH (a:Answer)-[:ANSWERS]->(q)
OPTIONAL MATCH (u:User)-[:ASKED]->(q)
RETURN q.title          AS question_title,
       q.body           AS question_body,
       q.score          AS question_score,
       a.body           AS answer_body,
       a.score          AS answer_score,
       a.is_accepted    AS is_accepted,
       t.name           AS tag,
       u.display_name   AS asked_by,
       score
ORDER BY score DESC, q.score DESC
LIMIT 50

# Example 5: User-first traversal — find content from expert/high-reputation users
# Use User_index when the user asks "who answered X" or wants to find contributions from specific users
CALL db.index.vector.queryNodes('User_index', 20, $question_embedding) YIELD node AS u, score
OPTIONAL MATCH (u)-[:PROVIDED]->(a:Answer)-[:ANSWERS]->(q:Question)
OPTIONAL MATCH (u)-[:ASKED]->(askedQ:Question)
OPTIONAL MATCH (q)-[:TAGGED]->(t:Tag)
RETURN u.display_name   AS user_name,
       u.reputation     AS reputation,
       q.title          AS question_title,
       a.body           AS answer_body,
       a.score          AS answer_score,
       a.is_accepted    AS is_accepted,
       collect(DISTINCT t.name) AS tags,
       score
ORDER BY score DESC, u.reputation DESC
LIMIT 50

The question is:
{question}"""