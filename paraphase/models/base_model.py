from abc import ABC, abstractmethod
import torch

class BaseParaphraser(ABC):
    def __init__(self, model_name: str, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

    @abstractmethod
    def load_model(self):
        """Load the model and tokenizer"""
        pass
    
    @abstractmethod
    def paraphrase(self, text: str, num_return_sequences: int = 1, **kwargs) -> list:
        """Generate paraphrase for a single text"""
        pass
    
    @abstractmethod
    def batch_paraphrase(self, texts: list, num_return_sequences: int = 1, **kwargs) -> list:
        """Generate paraphrases for multiple texts"""
        pass