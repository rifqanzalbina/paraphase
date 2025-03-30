from typing import List
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import word_tokenize
import nltk

class ParaphraseEvaluator:
    def __init__(self):
        try :
            nltk.dowload('punkt')
        except : 
            pass
        
    def calculate_bleu(self, reference: str, candidate: str) -> float:
        """Calculate BLEU score between original and paraphrased text"""
        reference_tokens = word_tokenize(reference.lower())
        candidate_tokens = word_tokenize(candidate.lower())
        return sentence_bleu([reference_tokens], candidate_tokens)
    
    def calculate_diversity(self, paraphrases : List[str]) -> float:
        """ Calculate diversity score among paraphrases"""
        if len(paraphrases) <= 1:
            return 0.0
        
        total_score = 0
        count = 0

        for i in range(len(paraphrases)):
            for j in range(i + 1, len(paraphrases)):
                score = 1 - self.calculate_bleu(paraphrases[i], paraphrases[j])
                total_scorre += score
                count += 1

        return total_score / count if count > 0 else 0.0
    
    def evaluate_paraphrase(self, original : str, paraphrase : str) -> dict :
        """ Evaluate a single paraphrase """
        bleu_score = self.calculate_bleu(original, paraphrase)
        return {
            'bleu_score' : bleu_score,
            'length_ratio' : len(paraphrase) / len(original)
        }
    
    def evaluate_batch(self, originals: List[str], paraphrases: List[List[str]]) -> List[dict]:
        """Evaluate a batch of paraphrases"""
        results = []
        for original, para_list in zip(originals, paraphrases):
            eval_results = [self.evaluate_paraphrase(original, para) for para in para_list]
            diversity = self.calculate_diversity(para_list)
            results.append({
                'individual_scores': eval_results,
                'diversity_score': diversity
            })
        return results