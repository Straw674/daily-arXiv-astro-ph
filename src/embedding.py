import math
import logging

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
    Computes kNN similarity scores for paper embeddings against Zotero embeddings.
    """
    scores = []
    for emb in paper_embs:
        sims = [cosine_similarity(emb, z_emb) for z_emb in zotero_embs]
        sims.sort(reverse=True)
        top = sims[:top_k]
        avg_score = sum(top) / len(top) if top else 0.0
        scores.append(avg_score)
    return scores
