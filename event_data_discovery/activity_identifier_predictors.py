import random

from sklearn.base import BaseEstimator, ClassifierMixin


class RandomActivityIdentifierPredictor(BaseEstimator, ClassifierMixin):

    def fit(self, X, y):
        self.p = sum(y) / len(y)
        return self

    def predict(self, X):
        random.seed(1)
        return [int(random.random() <= self.p) for x in X]


class OnePerTS(BaseEstimator, ClassifierMixin):

    def fit(self, X, y):
        return self

    def predict(self, X):
        random.seed(1)
        y = [0] * len(X)
        ts_to_cand = dict()
        for i, x in enumerate(X):
            if x['timestamp_attribute_id'] not in ts_to_cand:
                ts_to_cand[x['timestamp_attribute_id']] = list()
            ts_to_cand[x['timestamp_attribute_id']].append(i)
        for i in ts_to_cand:
            y[random.choice(ts_to_cand[i])] = 1
        return y
