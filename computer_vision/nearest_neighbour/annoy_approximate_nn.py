"""Annoy approximate nearest neighbour.
"""
import numpy as np
from annoy import AnnoyIndex

embed_dim = 4
annoy_index = AnnoyIndex(embed_dim, 'angular')


query_vector = np.asarray([0.1, 0.2, 0.5, 0.8])

vector_db = np.asarray([
    [0.5, 0.8, 0.8, 0.2],
    [0.2, 0.3, 0.6, 0.7],
    [0.2, 0.3, 0.4, 0.9],
    [0.12, 0.3, 0.5, 0.8],
    [0.1, 0.2, 0.51, 0.81],
    [0.9, 0.9, 0.1, 0.1],
    [0.1, 0.2, 0.5, 0.8],
])


for i in range(vector_db.shape[0]):
    annoy_index.add_item(i, vector_db[i])

annoy_index.build(10)
indices, distances = annoy_index.get_nns_by_vector(query_vector, 3, include_distances=True)

print(indices, distances)
