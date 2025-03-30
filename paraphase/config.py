from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    model_type : str = "t5" 
    model_name : str = "indonesian-nlp/t5-base-indonesian-paraphrase"
    max_length : int = 50
    num_beams : int = 5
    temperature : float = 0.7
    top_k : int = 50
    top_p : float = 0.85
    device : Optional[str] = None
    batch_size : int = 4

    @classmethod
    def default_t5_config(cls):
        return cls(
            model_type = "t5",
            model_name = "indonesian-nlp/t5-base-indonesian-paraphrase"
        )
    
    @classmethod
    def default_bart_config(cls):
        return cls(
            model_type = "bart",
            model_name = "indonesian/bart-base"
        )
    
    