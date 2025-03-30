from transformers import BartForConditionalGeneration, BartTokenizer
from .base_model import BaseParaphraser


class BARTParaphraser(BaseParaphraser):
    def __init__(self, model_name : str = "facebook/bart-base", device : str = None):
        super().__init__(model_name, device)
        self.load_model()

    def load_model(self):
        self.tokenizer = BartTokenizer.from_pretrained(self.model_name)
        self.model = BartForConditionalGeneration.from_pretrained(self.model_name)
        self.model.to(self.device)
    
    def paraphrase(self, text : str, num_return_sequences : int = 1, max_length : int = 250) -> list:
        inputs = self.tokenizer(text, return_tensor="pt", max_length=max_length, truncation=True)
        inputs = inputs.to(self.device)

        outputs = self.model.generate(
            **inputs,
            num_return_sequences= num_return_sequences,
            num_beams = num_return_sequences * 2,
            max_length = max_length,
            temperature = 0.7,
            top_k = 50,
            top_p = 0.95,
            do_sample = True,
        )

        paraphrases = []
        for output in outputs:
            paraphrase = self.tokenizer.decode(output, skip_special_tokens = True)
            paraphrases.append(paraphrase)

        return paraphrases
    
    def batch_paraphrase(self, texts: list, num_return_sequences: int = 1, max_length: int = 256) -> list:
        return [self.paraphrase(text, num_return_sequences, max_length) for text in texts]