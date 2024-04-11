import torchdatasets as td
from _datasets._datasets import get_dataset


class IndexedDataset(td.Dataset):
    def __init__(self, args, train=True, train_transform=False):
        super().__init__()
        self.dataset = get_dataset(args, train=train, train_transform=train_transform)
        
        self.args = args

    def __getitem__(self, index):
        # if self.args.dataset == 'snli':
        #     data = self.dataset[index]
        #     target = self.dataset[index]['label']
        #     return data, target, index
        
        data, target = self.dataset[index]
        return data, target, index

    def __len__(self):
        return len(self.dataset)

    def clean(self):
        self._cachers = []