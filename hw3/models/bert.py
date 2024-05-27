if __name__ == '__main__':
    from base import BaseModel
    from utils import split_sequences
else:
    from .base import BaseModel
    from .utils import split_sequences
from transformers import pipeline
import numpy as np


class BertModel(BaseModel):
    def __init__(self):
        self.model = pipeline ('fill-mask', model = 'bert-base-uncased')
    def fit(self, X, y):
        pass
    def predict(self, X):
        pred = self.model(X)
        return pred[0]['token_str']
    def score(self, X, y):
        accuracy = 0
        for i, sample in enumerate(X):
            target = y[i]
            pred = self.predict(sample)
            if pred == target:
                accuracy += 1
            if i % 50 == 0:
                print(f"Sample {i} - Accuracy: {accuracy / (i + 1)}")
        return accuracy / len(X)

if __name__ == '__main__':
    model = BertModel()
    print(model.predict("I am so glad that I am [MASK] to complete this assignment."))

