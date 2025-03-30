import re
from typing import List

class TextCleaner:
    @staticmethod
    def remove_special_characters(text: str) -> str:
        return re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    @staticmethod
    def remove_extra_whitespace(text: str) -> str:
        return ' '.join(text.split())
    
    @staticmethod
    def remove_urls(text: str) -> str:
        return re.sub(r'http\S+|www.\S+', '', text)
    
    @staticmethod
    def remove_html_tags(text: str) -> str:
        return re.sub(r'<.*?>', '', text)
    
    def clean_text(self, text: str) -> str:
        """Apply all cleaning operations"""
        text = self.remove_urls(text)
        text = self.remove_html_tags(text)
        text = self.remove_special_characters(text)
        text = self.remove_extra_whitespace(text)
        return text.strip()
    
    def clean_batch(self, texts: List[str]) -> List[str]:
        """Clean a batch of texts"""
        return [self.clean_text(text) for text in texts]