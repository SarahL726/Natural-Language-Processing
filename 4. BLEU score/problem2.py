import re
from nltk.translate import bleu_score

def tokenize(sentence):
    res = re.sub(r'[^\w\s]', '', sentence)
    seq = res.split(' ')
    return " ".join(seq).lower()

def calculate_precision(candidate_sentence, reference_sentence, n):
    if n == 1:
        candidate_grams = set(candidate_sentence.split())
        reference_grams = set(reference_sentence.split())
    elif n == 2:
        candidate_grams = set(zip(candidate_sentence.split()[:-1], candidate_sentence.split()[1:]))
        reference_grams = set(zip(reference_sentence.split()[:-1], reference_sentence.split()[1:]))
    elif n == 3:
        candidate_grams = set(zip(candidate_sentence.split()[:-2], candidate_sentence.split()[1:-1], candidate_sentence.split()[2:]))
        reference_grams = set(zip(reference_sentence.split()[:-2], reference_sentence.split()[1:-1], reference_sentence.split()[2:]))
    
    # Count the number of candidate bigrams that are present in the reference bigrams
    correct_grams = candidate_grams.intersection(reference_grams)

    # Calculate the bigram precision as the ratio of correct bigrams to total candidate bigrams
    precision = len(correct_grams) / len(candidate_grams)
    return precision


if __name__ == "__main__":
    source = "Fodd bynnag, yr wyf yn ystyried bod iaith hiliol, iaith sy’n gwahaniaethu ar sail rhyw neu ar unrhyw sail arall, a honiadau yn erbyn Aelodau, yn peri tramgwydd."
    reference = "However, I consider that racist, sexist or other discriminatory language, and allegations against Members, are offensive."
    candidate = ["However, I consider racist language, sexist or other discrimination, and allegations against Members offensive.", 
                 "However, I regard racist language, language that discriminates on the basis of sex or on any other grounds, and allegations against Members, as offensive.",
                 "Racist Members consider that discriminatory allegations as language are the basis of offensive sexist allegations, however.",
                 "Allegations against members are offensive."]
    
    reference_token = tokenize(reference)
    candidate_tokens = []
    for c in candidate: candidate_tokens.append(tokenize(c))

    for ct in candidate_tokens:
        print(bleu_score.modified_precision(reference_token, ct, 1),
              bleu_score.modified_precision(reference_token, ct, 2),
              bleu_score.modified_precision(reference_token, ct, 3))
        print(calculate_precision(ct, reference_token, 1), 
              calculate_precision(ct, reference_token, 2), 
              calculate_precision(ct, reference_token, 3))
        print()
        

