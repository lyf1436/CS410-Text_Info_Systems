if __name__ == '__main__':
    from base import BaseModel
    from utils import split_sequences
else:
    from .base import BaseModel
    from .utils import split_sequences
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

class GPTModel(BaseModel):
    def __init__(self):
        return
    def fit(self, X, y):
        pass
    def predict(self, X):
        inputs = tokenizer(X, return_tensors="pt")
        # input_ids = inputs["input_ids"]
        # for id in input_ids[0]:
        #     word = tokenizer.decode(id)
        with torch.no_grad():
            logits = model(**inputs).logits[:, -1, :]
        pred_id = torch.argmax(logits).item()
        pred_word = tokenizer.decode(pred_id)
        return pred_word
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
    gpt_model = GPTModel()
    print(gpt_model.predict("I am so glad that"))

