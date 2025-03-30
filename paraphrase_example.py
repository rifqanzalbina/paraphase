from paraphase.models import T5Paraphraser, BARTParaphraser

def main():
    t5_paraphraser = T5Paraphraser()
    text = "Saya sangat senang belajar pemrograman python."

    print("T5 Paraphrases : ")
    paraphrases = t5_paraphraser.paraphrase(text, num_return_sequences=3)
    for i, p in enumerate(paraphrases, 1):
        print(f"{i}. {p}")

    bart_paraphraser = BARTParaphraser()

    print("\nBart paraphrases : ")
    paraphrases = bart_paraphraser.paraphrase(text, num_return_sequences=3)
    for i, p in enumerate(paraphrases, 1):
        print(f"{i}. {p}")

if __name__ == "__main__":
    main()