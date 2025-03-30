from paraphase import (
    T5Paraphraser,
    BARTParaphraser,
    TextPreprocessor,
    TextCleaner,
    ParaphraseEvaluator,
    ModelConfig
)

def main():
    # Initialize components
    config = ModelConfig.default_t5_config()
    preprocessor = TextPreprocessor()
    cleaner = TextCleaner()
    evaluator = ParaphraseEvaluator()
    
    # Initialize model
    paraphraser = T5Paraphraser(model_name=config.model_name)
    
    # Prepare text
    text = "Teknologi kecerdasan buatan sangat berkembang pesat di era modern ini."
    cleaned_text = cleaner.clean_text(text)
    processed_text = preprocessor.prepare_for_model(cleaned_text)
    
    # Generate paraphrases
    paraphrases = paraphraser.paraphrase(
        processed_text,
        num_return_sequences=3,
        max_length=config.max_length
    )
    
    # Evaluate results
    evaluation = evaluator.evaluate_batch([text], [paraphrases])
    
    # Print results
    print("Original:", text)
    print("\nParaphrases:")
    for i, para in enumerate(paraphrases, 1):
        print(f"{i}. {para}")
    
    print("\nEvaluation:")
    print(f"Diversity Score: {evaluation[0]['diversity_score']:.3f}")
    for i, score in enumerate(evaluation[0]['individual_scores'], 1):
        print(f"Paraphrase {i} BLEU Score: {score['bleu_score']:.3f}")

if __name__ == "__main__":
    main()