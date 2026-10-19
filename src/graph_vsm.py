import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity


def build_user_item_vsm(events_dataframe, event_weights=None):
    if event_weights is None:
        event_weights = {"view": 1.0, "addtocart": 2.0, "transaction": 3.0}

    df = events_dataframe.copy()
    df["weight"] = df["event"].map(event_weights)

    unique_users = np.sort(df["visitorid"].unique())
    unique_items = np.sort(df["itemid"].unique())

    user_to_idx = {uid: i for i, uid in enumerate(unique_users)}
    item_to_idx = {iid: j for j, iid in enumerate(unique_items)}

    row_indices = df["visitorid"].map(user_to_idx).values
    col_indices = df["itemid"].map(item_to_idx).values
    data_weights = df["weight"].values

    matrix = sp.csr_matrix(
        (data_weights, (row_indices, col_indices)),
        shape=(len(unique_users), len(unique_items)),
        dtype=np.float32
    )

    return matrix, unique_users, unique_items, user_to_idx, item_to_idx


def compute_matrix_sparsity(sparse_matrix):
    total_elements = sparse_matrix.shape[0] * sparse_matrix.shape[1]
    nnz = sparse_matrix.nnz
    sparsity = (1.0 - (nnz / total_elements)) * 100.0
    return sparsity, nnz, total_elements


def compute_top_k_user_neighbors(user_vsm_matrix, unique_users, user_to_idx, target_user_id, k=5):
    target_idx = user_to_idx[target_user_id]
    target_vector = user_vsm_matrix[target_idx]

    co_item_cols = target_vector.indices
    candidate_user_indices = np.unique(user_vsm_matrix[:, co_item_cols].nonzero()[0])
    candidate_vectors = user_vsm_matrix[candidate_user_indices]

    similarities = cosine_similarity(target_vector, candidate_vectors).flatten()
    sorted_order = np.argsort(-similarities)

    top_neighbors = []
    top_scores = []
    for idx in sorted_order:
        cand_uid = unique_users[candidate_user_indices[idx]]
        if cand_uid != target_user_id and similarities[idx] > 0.0:
            top_neighbors.append(int(cand_uid))
            top_scores.append(float(similarities[idx]))
            if len(top_neighbors) == k:
                break

    return top_neighbors, top_scores
