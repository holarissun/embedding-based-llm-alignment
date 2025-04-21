import numpy as np
import itertools

def pair_annotate(embeddings, rewards, annotation_quality=0.1, pair_strategy='random', pairs_per_prompt=10):
    """
    Construct pairwise training data (winner vs loser) from reward-annotated embeddings.

    Args:
        embeddings: np.ndarray, shape (N, K, D)
            Embeddings of responses for each prompt.
        rewards: np.ndarray, shape (N, K, 1)
            Corresponding reward scores.
        annotation_quality: float
            Simulated annotation quality.
            > 0 : stochastic Bradley-Terry-style labeling.
            = -1: perfect annotation (based on true rewards).
        pair_strategy: str
            Strategy to generate pairs. Supported: 'random'
        pairs_per_prompt: int
            Number of pairs to sample per prompt (only used in 'random' strategy).

    Returns:
        pairwise_embeddings: np.ndarray, shape (M, 2, D)
            Embeddings of (winner, loser) pairs.
        labels: np.ndarray, shape (M,)
    """
    N, K, D = embeddings.shape
    positive_sample = []
    negative_sample = []
    labels = []

    rew_scale_factor = np.std(rewards) + 1e-8  # for BT-model prob normalization

    for i in range(N):  # iterate over prompts
        if pair_strategy == 'random':
            sampled_pairs = set()
            num_attempts = 0
            while len(sampled_pairs) < pairs_per_prompt and num_attempts < pairs_per_prompt * 5:
                j, k = np.random.choice(K, 2, replace=False)
                pair_key = tuple(sorted((j, k)))
                if pair_key not in sampled_pairs:
                    sampled_pairs.add(pair_key)
                num_attempts += 1
        else:
            raise NotImplementedError(f"Unknown pair_strategy: {pair_strategy}")

        for j, k in sampled_pairs:
            # flip order randomly to avoid position bias
            if np.random.rand() < 0.5:
                idx1, idx2 = j, k
            else:
                idx1, idx2 = k, j

            rew1 = rewards[i, idx1, 0]
            rew2 = rewards[i, idx2, 0]
            emb1 = embeddings[i, idx1]
            emb2 = embeddings[i, idx2]

            if annotation_quality < 0:
                # perfect annotator: choose the higher reward
                if rew1 > rew2:
                    positive_sample.append(emb1)
                    negative_sample.append(emb2)
                else:
                    positive_sample.append(emb2)
                    negative_sample.append(emb1)
            else:
                # noisy label via BT model
                delta_reward = (rew1 - rew2) / rew_scale_factor
                prob = 1 / (1 + np.exp(-delta_reward * annotation_quality))
                if np.random.rand() < prob:
                    positive_sample.append(emb1)
                    negative_sample.append(emb2)
                else:
                    positive_sample.append(emb2)
                    negative_sample.append(emb1)

    # Output arrays
    positive_sample = np.array(positive_sample)  # shape (M, D)
    negative_sample = np.array(negative_sample)  # shape (M, D)
    pairwise_embeddings = np.stack([positive_sample, negative_sample], axis=1)  # shape (M, 2, D)
    labels = np.ones(pairwise_embeddings.shape[0], dtype=np.int32) # dummy labels
    return pairwise_embeddings, labels
