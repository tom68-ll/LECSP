from sklearn.cluster import KMeans
import numpy as np
import torch

# embedding = [1, sequence_length, embedding_dimension]
# embeddings = [embedding1, embedding2, ..., embeddingN]


def k_means(embeddings, n_clusters, batch_size=100):
    # Convert embeddings to numpy array in batches to avoid OOM
    embeddings_array = np.array([e.cpu().numpy() for e in embeddings])
    print(embeddings_array.shape)

    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init=10)
    clusters = kmeans.fit_predict(embeddings_array)

    cluster_centers = kmeans.cluster_centers_

    # Calculate distances in batches
    num_samples = embeddings_array.shape[0]
    closest_data = np.zeros(num_samples, dtype=int)
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        batch_embeddings = embeddings_array[start:end]
        distances = np.linalg.norm(batch_embeddings[:, np.newaxis, :] - cluster_centers, axis=2)
        closest_data[start:end] = np.argmin(distances, axis=1)

    return clusters, closest_data


if __name__ == '__main__':
    np.random.seed(0)
    #example
    embeddings = [torch.rand(1, 19, 768) for _ in range(10)] 
    k_means(embeddings,n_clusters=3)