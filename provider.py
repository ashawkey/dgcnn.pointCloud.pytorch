import h5py
import numpy as np
import torch
import torch.utils.data

def load_h5(h5_filename):
    f = h5py.File(h5_filename,"r")
    data = f['data']
    label = f['label']
    return data, label


class ModelNet40(torch.utils.data.Dataset):
    def __init__(self, part, channel="last", points=2048):
        print("==> Creating dataset <"+part+">:")
        if part=="train":
            files_path = "data/modelnet40_ply_hdf5_2048/train_files.txt"
        else:
            files_path = "data/modelnet40_ply_hdf5_2048/test_files.txt"
        
        with open(files_path, "r") as f:
            files = f.readlines()
        
        data = []
        labels = []
        for f in files:
            X, Y = load_h5(f[:-1])
            data.append(X)
            labels.append(Y)

        self.data = np.vstack(data).astype(np.float32)
        self.labels = np.vstack(labels).reshape((-1)).astype(np.int64)
        self.length = self.data.shape[0]
        
        # random sample 
        self.data = self.data[:, 0:points, :]

        if channel != "last":
            self.data = np.transpose(self.data, (0,2,1)) # [B, dims, N] for pytorch conv 

        print("    data:", self.data.shape, self.data.dtype)
        print("    labels:",self.labels.shape, self.labels.dtype)

        assert(self.data.shape[0] == self.labels.shape[0])
        print("==> Dataset Created")


    def __getitem__(self, i):
        return self.data[i], self.labels[i]

    def __len__(self):
        return self.length


