import numpy as np


class LabelEncoder:
    """
    Label encode a series of labels
    """

    def __init__(self, data) -> None:
        self.__training_data = data
        self.index_to_token = list(set(data))
        self.token_to_index = {
            token: index for index, token in enumerate(self.index_to_token)
        }

    def __len__(self):
        return len(self.token_to_index)

    @property
    def encoded_data(self):
        return np.array([self.token_to_index[token] for token in self.__training_data])

    def encode(self, data):
        return np.array([self.token_to_index[token] for token in data])

    def decode(self, data):
        if isinstance(data, int) or isinstance(data, np.int64):
            return self.index_to_token[data]
        else:
            return np.array([self.index_to_token[index] for index in data])
