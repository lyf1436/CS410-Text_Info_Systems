from .lstm import LSTMModel
from .bert import BertModel
from .gpt import GPTModel

MODEL_ZOO = {
    'lstm': LSTMModel,
    'bert': BertModel,
    'gpt': GPTModel
}