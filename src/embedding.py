import logging
import math

import numpy as np

logger = logging.getLogger(__name__)


def get_embeddings_in_batches(
    client, texts: list[str], model: str, batch_size: int = 10
) -> list[list[float]]:
    """
    Computes embeddings for a list of texts in batches.
    """
    paper_embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        try:
            response = client.embeddings.create(model=model, input=batch)
            # Ensure we append in the correct order
            sorted_data = sorted(response.data, key=lambda x: x.index)
            paper_embs.extend([x.embedding for x in sorted_data])
            logger.info(
                f"Processed batch {i // batch_size + 1}/{(len(texts) - 1) // batch_size + 1}"
            )
        except Exception as e:
            logger.error(f"Embedding computation failed for batch starting at {i}: {e}")
            raise e
    return paper_embs


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """
    Computes the cosine similarity between two vectors.
    """
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    return dot_product / (norm1 * norm2) if norm1 and norm2 else 0.0


def compute_knn_scores(
    paper_embs: list[list[float]], zotero_embs: list[list[float]], top_k: int = 5
) -> list[float]:
    """
    Computes kNN similarity scores for paper embeddings against Zotero embeddings using NumPy vectorization.
    """
    if not paper_embs:
        return []
    if not zotero_embs:
        return [0.0] * len(paper_embs)

    p_mat = np.asarray(paper_embs, dtype=np.float32)
    z_mat = np.asarray(zotero_embs, dtype=np.float32)

    p_norm = np.linalg.norm(p_mat, axis=1, keepdims=True)
    z_norm = np.linalg.norm(z_mat, axis=1, keepdims=True)

    p_norm = np.where(p_norm == 0, 1.0, p_norm)
    z_norm = np.where(z_norm == 0, 1.0, z_norm)

    p_normalized = p_mat / p_norm
    z_normalized = z_mat / z_norm

    sim_matrix = np.matmul(p_normalized, z_normalized.T)

    k = min(top_k, sim_matrix.shape[1])
    if k <= 0:
        return [0.0] * len(paper_embs)

    top_k_sims = np.partition(sim_matrix, -k, axis=1)[:, -k:]
    return np.mean(top_k_sims, axis=1).astype(float).tolist()
