import random
import string

from torch.utils.data.dataset import Dataset


def train_test_split(pairs, train_test_split_ratio):
    random.shuffle(pairs)
    split = int(train_test_split_ratio * len(pairs))
    train_pairs, test_pairs = pairs[:split], pairs[split:]
    return train_pairs, test_pairs


class SimpleDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class CharSortedDataset(Dataset):
    def __init__(
        self,
        N,
        min_length=1,
        max_length=26,
        population=string.ascii_lowercase,
        replace=True,
    ):
        assert min_length >= 1
        assert max_length >= min_length
        if replace:
            assert max_length <= len(population)

        def random_sequence():
            k = random.randint(min_length, max_length)
            if replace:
                return random.choices(population, k=k)
            else:
                return random.sample(population, k=k)

        self.N = N
        self.srcs = [random_sequence() for _ in range(N)]
        self.trgs = [sorted(src) for src in self.srcs]
        self.srcs = ["".join(src) for src in self.srcs]
        self.trgs = ["".join(trg) for trg in self.trgs]

    def __getitem__(self, index):
        return self.srcs[index], self.trgs[index]

    def __len__(self):
        return self.N
