import copy
import torch
from torch_geometric.data import Data, Batch, HeteroData


def parse_slice(slice_, end=0):
    start = slice_.start if slice_.start is not None else 0
    step = slice_.step if slice_.step is not None else 1
    if slice_.stop is None:
        stop = end
    elif slice_.stop == -1:
        stop = end - 1
    else:
        stop = slice_.stop
    return range(start, stop, step)

class GraphBatchStorage:
    def __init__(self, B=None, T=None, data=None):
        if data is None:
            assert B is not None and T is not None
            self.B = B
            self.T = T
            self.storage = [None for _ in range(T)]
        else:
            self.storage = data
            self.B = len(data)
            self.T = len(data[0])
        # print(f"Initializing GraphBatchStorage B: {self.B}, T: {self.T}")
        # in running an episode: batch size B is 1 and episode length T is given
        # in replay buffer: batch size is given (>1) and episode length T is given too

    def set_graph(self, t, graph_data):
        if isinstance(graph_data, HeteroData):
            assert t < self.T, IndexError("Index (t, b) out of bounds.")
            assert graph_data['channel']['batch'].max().item() + 1 == self.B
            self.storage[t] = graph_data
        else:
            raise NotImplementedError

    def __getitem__(self, index):

        if isinstance(index, int):
            return self.storage[index]
            # return Batch.from_data_list([graph for graph in self.storage[index] if graph is not None])

        elif len(index) == 2:
            index = tuple(index)

            if isinstance(index[0], int) and isinstance(index[1], int): # reading a single step
                return self.storage[index[0]][index[1]]

            if index[0] == slice(None) and isinstance(index[1], int): # reading a batch of a single steps of all episode 
                t = index[1]
                return Batch.from_data_list([graphs[t] for graphs in self.storage if graphs[t] is not None])

            elif index[1] == slice(None) and isinstance(index[0], int): # reading a batch of an episode
                return Batch.from_data_list([graph for graph in self.storage[index[0]] if graph is not None])

            elif isinstance(index[0], slice) and isinstance(index[1], slice): # reading a sample of batches
                data = []
                for i in parse_slice(index[0], self.B):
                    batch = []
                    for j in parse_slice(index[1], self.T):
                        if self.storage[i][j] is not None:
                            batch.append(self.storage[i][j])
                    data.append(batch)
                return GraphBatchStorage(data=data)

            elif isinstance(index[0], list) and isinstance(index[1], slice):
                data = []
                for b in index[0]:
                    batch = []
                    for t in parse_slice(index[1], self.T):
                        if self.storage[b][t] is not None:
                            batch.append(self.storage[b][t])
                    data.append(batch)
                return GraphBatchStorage(data=data)

            else:
                raise IndexError("Invalid index. Use [:, t] to retrieve the batch at time step t.")

        else:
            raise IndexError("Invalid index.")

    def __str__(self):
        output = "GraphBatchStorage:\n"
        for b in range(self.B):
            # Count non-None graphs in each batch
            num_graphs = sum(1 for graph in self.storage[b] if graph is not None)
            if num_graphs > 0:
                output += f"  Batch[{b}]: {num_graphs} graphs\n"
        return output

    def clone(self):
        return GraphBatchStorage(data=copy.deepcopy(self.storage))

    @property
    def shape(self):
        list_ = []
        for b in range(self.B):
            # Count non-None graphs in each batch
            num_graphs = sum(1 for graph in self.storage[b] if graph is not None)
            if num_graphs > 0:
                list_.append((b, num_graphs))
        return "(batch_id, non_empty_graphs): " + str(list_)

    def merge_all_graphs(self):
        return Batch.from_data_list(self.storage[:-1])
