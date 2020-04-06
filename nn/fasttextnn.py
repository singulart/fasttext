import numpy as np


class FastTextNN:
    """
        Code grabbed from here: https://github.com/facebookresearch/fastText/pull/552
    """
    def __init__(self, ft_model, ft_matrix=None):
        self.ft_model = ft_model
        self.ft_words = ft_model.get_words()
        self.word_frequencies = dict(zip(*ft_model.get_words(include_freq=True)))
        self.ft_matrix = ft_matrix
        if self.ft_matrix is None:
            self.ft_matrix = np.empty((len(self.ft_words), ft_model.get_dimension()))
            for i, word in enumerate(self.ft_words):
                self.ft_matrix[i, :] = ft_model.get_word_vector(word)

    def find_nearest_neighbor(self, query_word, vectors, n=10, cossims=None):
        """
        vectors is a 2d numpy array corresponding to the vectors you want to consider

        cossims is a 1d numpy array of size len(vectors), which can be passed for efficiency
        returns the index of the closest n matches to query within vectors and the cosine similarity (cosine the angle between the vectors)

        """

        query = self.ft_model.get_word_vector(query_word)
        if cossims is None:
            cossims = np.matmul(vectors, query, out=cossims)

        norms = np.sqrt((query ** 2).sum() * (vectors ** 2).sum(axis=1))
        cossims = cossims / norms
        if query_word in self.ft_words:
            result_i = np.argpartition(-cossims, range(n + 1))[1:n + 1]
        else:
            result_i = np.argpartition(-cossims, range(n + 1))[0:n]
        return list(zip(result_i, cossims[result_i]))

    def nearest_words(self, word, n=10, word_freq=None):
        result = self.find_nearest_neighbor(word, self.ft_matrix, n=n)
        if word_freq:
            return [(self.ft_words[r[0]], r[1]) for r in result if
                    self.word_frequencies[self.ft_words[r[0]]] >= word_freq]
        else:
            return [(self.ft_words[r[0]], r[1]) for r in result]
