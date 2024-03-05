from os import stat
import typing

import numpy as np

import typing


class BagOfWords:
    """
    A type of encoder, makes
    """

    def __init__(self, data: typing.Iterable) -> None:
        """
        Generate the bag of words
        :param data: an array of words, or an iterable containing arrays of words
        """
        data = np.array(self.__linearize_array(data))
        self.index_to_words = np.unique(data)
        self.words_to_index = {w: i for i, w in enumerate(self.index_to_words)}

    @classmethod
    def __linearize_array(cls, text):
        x = []
        for item in text:
            if isinstance(item, str):
                x.append(item)
            else:
                x.extend(cls.__linearize_array(item))
        return x

    def __call__(self, text: typing.Iterable[str]) -> np.array:
        return self.get_counts(text)

    def __len__(self) -> int:
        return len(self.index_to_words)

    def encode_data(
        self: "BagOfWords",
        text: typing.Union[typing.Iterable[str], typing.Iterable[typing.Iterable[str]]],
    ) -> np.array:
        """
        Compute the encodings of words in a new input tokenized string
        """
        x = []
        for item in text:
            if isinstance(item, str):
                if item in self.words_to_index:
                    x.append(self.words_to_index[item])
            else:
                x.append(self.encode_data(item))
        return x

    def decode_data(self: "BagOfWords", encoded_text: typing.Iterable[int]):
        if isinstance(encoded_text, int) or isinstance(encoded_text, np.int64):
            return self.index_to_words[encoded_text]
        else:
            return list(map(self.decode_data, encoded_text))

    def get_counts(
        self: "BagOfWords",
        text: typing.Union[typing.Iterable[str], typing.Iterable[typing.Iterable[str]]],
    ):
        """
        Computes the counts of words in a new input tokenized string
        """
        if len(text) == 0 or isinstance(text[0], str):
            x = np.zeros(shape=len(self))
            for word in text:
                if word in self.words_to_index:
                    x[self.words_to_index[word]] += 1
            return x
        else:
            return np.stack([self.get_counts(item) for item in text], axis=0)
