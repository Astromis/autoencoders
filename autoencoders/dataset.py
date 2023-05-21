
import torch
import faiss
import numpy as np
from torch.utils.data import Dataset

class EmbeddingDataset(Dataset):
    """The dataset holding any embedding matrix"""

    def __init__(self, embeddings):
        self.embeddings = torch.from_numpy(embeddings)

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # 1 is a dumb value in order to be conviniet with mnist dataset
        return self.embeddings[idx], 1

class EmbeddingDatasetWithGraph(EmbeddingDataset):
    def __init__(self, embeddings, k, graph_config=False, **kwargs):
        super(EmbeddingDatasetWithGraph, self).__init__(embeddings)

        self.graph_config = graph_config 
        self.set_graph(k, embeddings)
    
    def set_graph(self, k, embeddings):
        index = faiss.IndexHNSWFlat(embeddings.shape[1], 32)

        # training is not needed

        # this is the default, higher is more accurate and slower to
        # construct
        index.hnsw.efConstruction = 40
        # to see progress
        index.verbose = False
        index.add(embeddings)

        index.hnsw.search_bounded_queue = False
        index.hnsw.efSearch = 256
        D, I = index.search(embeddings, k+1)
        self.dist_mat_indices = I[:,1:k+1]
        #data_temp = self.data.view(len(self.data), -1).clone()
        #dist_mat = torch.cdist(data_temp, data_temp)
        #dist_mat_indices = torch.topk(dist_mat, k=self.graph_config['num_nn'] + 1, dim=1, largest=False, sorted=True)
        #self.dist_mat_indices = dist_mat_indices.indices[:, 1:]

    def __getitem__(self, idx):
        bs_nn = self.graph_config['bs_nn']
        if self.graph_config['include_center']:
            x_c = self.embeddings[idx]
            x_nn = self.embeddings[
                self.dist_mat_indices[
                    idx, 
                    np.random.choice(range(self.graph_config['num_nn']), bs_nn-1, replace=self.graph_config['replace'])
                ]
            ]
            return x_c, torch.cat([x_c.unsqueeze(0), x_nn], dim=0)
        else:
            x_c = self.embeddings[idx]
            x = self.embeddings[
                self.dist_mat_indices[
                    idx, 
                    np.random.choice(range(self.graph_config['num_nn']), bs_nn, replace=self.graph_config['replace'])
                ]
            ]
            return x_c, x
