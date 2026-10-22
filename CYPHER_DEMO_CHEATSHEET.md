# Cypher Live Demo Cheatsheet and Examination Defense Guide

Student Name: Rizki Piji Fathoni
Student NIM: 24031554029
Course: Graph Data Science (Midterm Examination)
Target Reference Visitor ID: 36281
Database Instance: neo4j (Bolt port 7687)

---

## 1. Quick Live Demo Operational Runbook

When presenting or demonstrating live in class, follow these exact operational steps:

1. **Start the Neo4j DBMS**:
   - Open Neo4j Desktop.
   - Locate your database instance (e.g., `UTS GDS RetailRocket`).
   - Click the green **Start** button and wait until the status displays green **Active**.
   - Note: The database is completely persistent on local disk. All 153,836 Visitor nodes, 76,078 Item nodes, 265,509 interaction edges, and the 5 `SIMILAR` edges are already permanently stored. You do NOT need to re-import anything.

2. **Open Neo4j Browser UI**:
   - In Neo4j Desktop, click the blue **Open** button to launch the Neo4j Browser.
   - Verify connection status in the left sidebar (connected as `neo4j` to `neo4j://localhost:7687`).

3. **Presenting the Jupyter Notebook**:
   - Open `notebooks/GDS_Midterm_Rizki_Piji.ipynb`.
   - All cell outputs, evaluation tables, and high-resolution NetworkX plots are pre-rendered.
   - Walk through the pipeline from Section 0 (Business Understanding) to Section 4 (Conclusion).

4. **Executing Interactive Queries in Neo4j Browser**:
   - Whenever the examiner asks to verify queries live, copy and paste the corresponding Cypher statement below into the top command line prompt (`$`) and press `Ctrl + Enter`.

---

## 2. Definitive Cypher Query Collection

### Query 1: Total Graph Ingestion and Component Summary (Soal 1.3 & 2.1.3)
Verify total node labels and counts:
```cypher
MATCH (n)
RETURN labels(n)[0] AS nodeType, count(n) AS totalCount
ORDER BY totalCount DESC;
```
Expected output:
- `Visitor`: 153,836
- `Item`: 76,078

Verify total relationship types and counts:
```cypher
MATCH ()-[r]->()
RETURN type(r) AS relationshipType, count(r) AS totalCount
ORDER BY totalCount DESC;
```
Expected output:
- `VIEWED`: 256,371
- `ADDED_TO_CART`: 7,047
- `BOUGHT`: 2,091
- `SIMILAR`: 5

---

### Query 2: Retrieve Top-5 Similar Visitors for Anchor Visitor 36281 (Soal 2.1.7a)
Execute interactive query on the persisted graph:
```cypher
MATCH (u1:Visitor {visitorId: 36281})-[s:SIMILAR]->(u2:Visitor)
RETURN u1.visitorId AS sourceVisitorId, 
       u2.visitorId AS similarVisitorId, 
       s.similarity AS similarityScore
ORDER BY similarityScore DESC
LIMIT 5;
```
Expected tabular output:
- Similar Visitor `176795`: `0.722315`
- Similar Visitor `1282391`: `0.481543`
- Similar Visitor `572116`: `0.481543`
- Similar Visitor `526884`: `0.340503`
- Similar Visitor `1258614`: `0.323029`

---

### Query 3: Write-Back of Similarity Edges into Graph Database (Soal 2.1.7b)
Demonstrating how the similarity edges were created:
```cypher
UNWIND [
  {sourceId: 36281, targetId: 176795, score: 0.722315},
  {sourceId: 36281, targetId: 1282391, score: 0.481543},
  {sourceId: 36281, targetId: 572116, score: 0.481543},
  {sourceId: 36281, targetId: 526884, score: 0.340503},
  {sourceId: 36281, targetId: 1258614, score: 0.323029}
] AS row
MATCH (u1:Visitor {visitorId: row.sourceId})
MATCH (u2:Visitor {visitorId: row.targetId})
MERGE (u1)-[s:SIMILAR]->(u2)
SET s.similarity = row.score;
```

---

### Query 4: Condition BEFORE Similarity Computation (Soal 2.1.8 - Pure Bipartite Subgraph)
Visualizing pure bipartite structure (visitors only connected to items):
```cypher
MATCH (v:Visitor)-[r]->(i:Item)
WHERE v.visitorId IN [36281, 176795, 1282391, 572116]
RETURN v, r, i
LIMIT 25;
```
Visual result in Neo4j Browser:
- Two distinct node colors (blue for Visitor, orange for Item).
- No direct user-to-user links exist.

---

### Query 5: Condition AFTER Similarity Computation (Soal 2.1.8 - Augmented Graph)
Visualizing augmented graph with direct peer shortcut edges:
```cypher
MATCH (v1:Visitor {visitorId: 36281})-[s:SIMILAR]->(v2:Visitor)
OPTIONAL MATCH (v1)-[r1]->(i:Item)
OPTIONAL MATCH (v2)-[r2]->(i)
RETURN v1, s, v2, r1, r2, i
LIMIT 50;
```
Visual result in Neo4j Browser:
- Direct directed `[:SIMILAR]` edge connecting Visitor `36281` to its nearest neighbors.
- Closed triangle paths connecting shared items, forming community clusters.

---

### Query 6: Inspecting Shared Co-Interacted Items Between Anchor and Top Peer
Demonstrating why Visitor 176795 has the highest similarity score (0.7223):
```cypher
MATCH (u1:Visitor {visitorId: 36281})-[r1]->(i:Item)<-[r2]-(u2:Visitor {visitorId: 176795})
RETURN i.itemId AS sharedItemId, type(r1) AS anchorAction, type(r2) AS peerAction;
```

---

## 3. Key Quantitative Metrics Reference

| Metric Name | Value | Meaning / Interpretation |
|:---|:---|:---|
| Target Visitor ID | `36281` | Anchor user derived deterministically via NIM seed `24031554029` |
| Top-3 Advertised Items | `65580`, `297974`, `30167` | Recommender scores: 0.3405, 0.3230, 0.3230 |
| Ground Truth Hit in P2 | Item `65580` at Rank 1 | Consumer actively interacted with item 65580 during August 16-20 |
| Single User Hit@5 | `1.0000` (100%) | Successful binary discovery in top recommendations |
| Single User NDCG@5 | `0.3392` | Normalized discounted rank gain with hit at rank position 1 |
| 100-User Cohort Hit@5 | `0.0200` vs `0.0000` | KNN captures signal in 99.998% sparse catalog vs popularity baseline |
