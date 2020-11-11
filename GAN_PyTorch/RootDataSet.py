from torch.utils.data import Dataset
import uproot
import numpy as np

class RootDataSet(Dataset):
    def __init__(self, filename):

        #    lines = f.read().split('\n')
        file = uproot.open(filename)
        events = file["PhaseSpaceTree"]
        NumEntries = events.numentries  # number of data objects (vectors)
        params = ["H1_PX", "H1_PY", "H1_PZ", "H2_PX", "H2_PY", "H2_PZ", "H3_PX", "H3_PY", "H3_PZ"]

        data = events.lazyarrays(params)
        #data["H1_PZ"] = data["H1_PZ"]/4
        #data["H2_PZ"] = data["H2_PZ"] / 4
        #data["H3_PZ"] = data["H3_PZ"] / 4
        self.data_arr = np.vstack(list(data[elem] for elem in params)).T
        self.leaves = len(params);
        # Splitting the text data and lables from each other
        #X, y = [], []
        #for line in lines:
        #    X.append(line.split(',')[0])
        #    y.append(line.split(',')[1])

        # Store them in member variables.
        # self.X = X
        # self.y = y

        #return text_pp

    def __len__(self):
        return len(self.data_arr)

    def __getitem__(self, index):
        return self.data_arr[index][:]
