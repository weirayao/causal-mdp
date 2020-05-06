import os
import random
import numpy as np
import torch

def create_fully_connected_edge_index(n_objects):
    n_relations  = n_objects * (n_objects - 1)
    edge_index = torch.zeros((2, n_relations), dtype=torch.long)
    count = 0
    for i in range(n_objects):
        for j in range(n_objects):
            if(i != j):
                edge_index[0, count] = i
                edge_index[1, count] = j
                count += 1
    return edge_index

class BallSimulationDataset(Dataset):
    def __init__(self, root, raw_folder_name, processed_folder_name, use_cuda=False):
        super().__init__()
        self.root = root
        self.raw_folder = os.path.join(root, raw_folder_name)
        self.processed_folder = os.path.join(root, processed_folder_name)
        if not os.path.exists(self.raw_folder):
            os.mkdir(self.raw_folder)
        if not os.path.exists(self.processed_folder):
            os.mkdir(self.processed_folder)
        self.filenames = [fn.split(".")[0] for fn in os.listdir(self.raw_folder) if "npz" in fn]
        self.filenames.reverse()
        self._process_raw_files()
        if use_cuda:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = 'cpu'
        
    def _process_raw_files(self):
        for filename in self.filenames:
            pt_fn = os.path.join(self.processed_folder, filename+'.pt')
            if not os.path.exists(pt_fn):
                raw_file_path = os.path.join(self.raw_folder, filename+'.npz')
                batch = np.load(raw_file_path)
                batch_x = torch.Tensor(batch["X"])
                batch_y = torch.Tensor(batch["Y"])
                n_objects = batch_x.shape[2]
                edge_index = create_fully_connected_edge_index(n_objects)
                torch.save((batch_x, batch_y, edge_index), pt_fn)
    
    def __getitem__(self, idx):
        filename = self.filenames[idx]
        pt_fn = os.path.join(self.processed_folder, filename+'.pt')
        batch = torch.load(pt_fn)
        batch = [item.to(self.device) for item in batch]
        return batch
    
    def __len__(self):
        return len(self.filenames)
    
    def shuffle(self):
        random.shuffle(self.filenames)