import numpy as np


def generate_knn_recommendations(events_p1_df, target_user_id, top_neighbors, top_similarities, event_weights=None, top_n=5):
    if event_weights is None:
        event_weights = {"view": 1.0, "addtocart": 2.0, "transaction": 3.0}

    user_history = set(events_p1_df[events_p1_df["visitorid"] == target_user_id]["itemid"].unique())

    item_scores = {}
    for neighbor_id, similarity_score in zip(top_neighbors, top_similarities):
        neighbor_df = events_p1_df[events_p1_df["visitorid"] == neighbor_id]
        for _, row in neighbor_df.iterrows():
            item_id = int(row["itemid"])
            if item_id not in user_history:
                weight = event_weights.get(row["event"], 1.0)
                item_scores[item_id] = item_scores.get(item_id, 0.0) + (similarity_score * weight)

    ranked_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
    return ranked_items[:top_n]


def evaluate_hit_at_k(recommended_items, ground_truth_items):
    truth_set = set(ground_truth_items)
    is_hit = any(item in truth_set for item in recommended_items)
    return 1.0 if is_hit else 0.0


def evaluate_ndcg_at_k(recommended_items, ground_truth_items, k=5):
    truth_set = set(ground_truth_items)
    dcg = sum(
        (1.0 if item in truth_set else 0.0) / np.log2(rank + 1)
        for rank, item in enumerate(recommended_items[:k], 1)
    )

    ideal_hits = min(len(truth_set), k)
    if ideal_hits == 0:
        return 0.0

    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg
