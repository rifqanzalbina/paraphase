import re
import nltk
from typing import List

class TextPreprocessor:
    def __init__(self):
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.dowload('punkt')

    def split_sentences(self, text : str) -> List[str]:
        return nltk.sent_tokenize(text)
    
    def normalize_text(self, text : str) -> str:
        # ! Lowercase
        text = text.lower()
        # ! Remove multiple spaces
        text = re.sub(r'\s+', '', text )
        return text.strip()

    def prepare_for_model(self, text : str) -> str:
        """ Prepare text for model input"""
        text = self.normalize_text(text)
        return text

    def process_batch(self, texts : List[str]) -> List[str]:
        """ Process a batch of texts """
        processed_texts = [self.prepare_for_model(text) for text in texts]
        return processed_texts