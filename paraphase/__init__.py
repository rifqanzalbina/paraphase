from .models import T5Paraphraser, BARTParaphraser
from .utils import TextPreprocessor, TextCleaner
from .evaluator import ParaphraseEvaluator
from .config import ModelConfig

__version__ = "0.0.1"
__all__ = [
    # 'Paraphraser',  # Remove it here as well
    'T5Paraphraser',
    'BARTParaphraser',
    'TextPreprocessor',
    'TextCleaner',
    'ParaphraseEvaluator',
    'ModelConfig'
]