from transformers import T5ForConditionalGeneration, T5Tokenizer
from .base_model import BaseParaphraser

class T5Paraphraser(BaseParaphraser):
    def __init__(self, model_name: str = "t5-base", device: str = None):
        super().__init__(model_name, device)
        self.load_model()

    def load_model(self):
        self.tokenizer = T5Tokenizer.from_pretrained(self.model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(self.model_name)
        self.model.to(self.device)

    def paraphrase(self, text: str, num_return_sequences: int = 1, max_length: int = 256) -> list:
        prefix = "paraphrase: "
        encoded = self.tokenizer(prefix + text, return_tensors="pt", max_length=max_length, truncation=True, padding=True)
        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=60,
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=2,
            num_beams=4
        )

        paraphrases = []
        for output in outputs:
            generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
            paraphrases.append(generated_text)

        return paraphrases

    def batch_paraphrase(self, texts: list, batch_size: int = 4, **kwargs):
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_results = [self.paraphrase(text, **kwargs) for text in batch]
            results.extend(batch_results)
        return results