# Graph Data Science: E-Commerce Interaction Subgraph Analysis and KNN Recommender System

## Midterm Examination Project (UTS)
- Course: Graph Data Science (Graf Sains Data)
- Student Name: Rizki Piji Fathoni
- Student ID (NIM): 24031554029
- Program: Undergraduate Program in Data Science (S1 Sains Data)
- Institution: Faculty of Mathematics and Natural Sciences, Universitas Negeri Surabaya (UNESA)

## Project Overview
This repository contains an end-to-end Graph Data Science pipeline analyzing user-item interaction dynamics on the RetailRocket e-commerce dataset. The project explores bipartite graph modeling, temporal train-test data partitioning, Vector Space Modeling (VSM), graph projection, cosine similarity write-back to Neo4j, and K-Nearest Neighbors (KNN) recommendation evaluated via Hit@5 and NDCG@5 metrics.

## Temporal Dataset Partitioning
The interaction dataset is partitioned using an industry-standard temporal train-test split (75:25 ratio):
- Training Period (Period 1): 2015-08-01 to 2015-08-15 (15 days)
- Testing Period (Period 2): 2015-08-16 to 2015-08-20 (5 days)
- Training Event Volume: 265,509 interaction events across 153,836 visitors and 76,078 items
- Testing Event Volume: 95,404 interaction events across 55,754 visitors and 39,601 items
- Cross-Period Visitor Overlap: 3,497 repeating visitors
- Cross-Period Item Overlap: 24,905 active items

## Repository Structure
```text
MidTerm-GDS/
├── .gitignore
├── README.md
├── requirements.txt
├── notebooks/
│   └── GDS_Midterm_Rizki_Piji.ipynb
├── src/
│   ├── data_loader.py
│   ├── neo4j_connector.py
│   ├── graph_vsm.py
│   └── recommender_knn.py
└── Data/
    ├── events_p1_training.csv
    └── events_p2_testing.csv
```

## Environment Setup and Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/Rizki0907/MidTerm-GDS.git
cd MidTerm-GDS
pip install -r requirements.txt
```
