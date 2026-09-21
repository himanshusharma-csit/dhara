import math
import re
import unicodedata
import numpy as np
from collections import Counter, defaultdict
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
import nltk as nl

# Run only the first time while setting a new environment variable or directory
# nl.download("wordnet")
# nl.download("punkt_tab")

# Optional import for ROUGE
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False


_TOKEN_PATTERN = re.compile(r"[\u0900-\u097F]+|[a-zA-Z]+|[0-9]+")


def tokenize_text(text):
    text = unicodedata.normalize("NFC", str(text).lower())
    return _TOKEN_PATTERN.findall(text)


def calculate_bleu_score(references=None, predictions=None):
    """
    Calculate corpus-level BLEU-1 through BLEU-4 scores.
    :param references: list of reference caption strings (e.g. 5 strings per image)
    :param predictions: list of predicted caption strings
    :return: (BLEU-1, BLEU-2, BLEU-3, BLEU-4) as percentages
    """
    tokenized_refs = [tokenize_text(ref) for ref in references]
    tokenized_preds = [tokenize_text(pred) for pred in predictions]
    # Each prediction shares all references as valid alternatives
    formatted_refs = [tokenized_refs for _ in tokenized_preds]

    # SmoothingFunction method1 handles cases where higher-order n-gram counts are 0
    smoothing = SmoothingFunction().method1
    bleu_1_score = corpus_bleu(formatted_refs, tokenized_preds, weights=(1, 0, 0, 0), smoothing_function=smoothing)
    bleu_2_score = corpus_bleu(formatted_refs, tokenized_preds, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
    bleu_3_score = corpus_bleu(formatted_refs, tokenized_preds, weights=(1/3, 1/3, 1/3, 0), smoothing_function=smoothing)
    bleu_4_score = corpus_bleu(formatted_refs, tokenized_preds, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)

    return (round(bleu_1_score * 100, 3),
            round(bleu_2_score * 100, 3),
            round(bleu_3_score * 100, 3),
            round(bleu_4_score * 100, 3))


def calculate_meteor_score(references=None, predictions=None):
    """
    Calculate average METEOR score across all predictions.
    :param references: list of reference caption strings
    :param predictions: list of predicted caption strings
    :return: average METEOR score as a percentage
    """
    tokenized_refs = [tokenize_text(ref) for ref in references]
    scores = []
    for pred in predictions:
        tokenized_pred = tokenize_text(pred)
        score = nl.translate.meteor_score.meteor_score(
            references=tokenized_refs,
            hypothesis=tokenized_pred
        )
        scores.append(score)
    avg_score = sum(scores) / len(scores)
    return round(avg_score * 100, 3)


def calculate_rouge_score(references=None, predictions=None):
    """
    Calculate average ROUGE-L F1 score across all predictions.
    Requires: pip install rouge-score
    :param references: list of reference caption strings
    :param predictions: list of predicted caption strings
    :return: average ROUGE-L F1 score as a percentage
    """
    if not ROUGE_AVAILABLE:
        raise ImportError("rouge_score package not found. Install with: pip install rouge-score")

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    scores = []
    for pred in predictions:
        # Score against each reference and take the best match
        best_score = 0.0
        for ref in references:
            result = scorer.score(ref, pred)
            best_score = max(best_score, result['rougeL'].fmeasure)
        scores.append(best_score)
    avg_score = sum(scores) / len(scores)
    return round(avg_score * 100, 3)


def _get_ngrams(tokens, n):
    """Extract n-grams from a token list."""
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _compute_tfidf(ngram_counts, doc_freq, num_docs):
    """Compute TF-IDF vector for a sentence given its n-gram counts."""
    tfidf = {}
    for ngram, count in ngram_counts.items():
        tf = math.log(1.0 + count)
        df = doc_freq.get(ngram, 0)
        idf = math.log(max(1.0, num_docs) / (1.0 + df))
        tfidf[ngram] = tf * idf
    return tfidf


def _cosine_similarity(vec_a, vec_b):
    """Compute cosine similarity between two sparse vectors (dicts)."""
    common_keys = set(vec_a.keys()) & set(vec_b.keys())
    if not common_keys:
        return 0.0
    dot = sum(vec_a[k] * vec_b[k] for k in common_keys)
    norm_a = math.sqrt(sum(v ** 2 for v in vec_a.values()))
    norm_b = math.sqrt(sum(v ** 2 for v in vec_b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def calculate_cider_score(references=None, predictions=None, max_n=4):
    """
    Calculate CIDEr-D score.
    :param references: list of reference caption strings
    :param predictions: list of predicted caption strings
    :param max_n: maximum n-gram order (default 4)
    :return: average CIDEr score (scaled by 10, as per convention)
    """
    # Tokenize all sentences
    tokenized_refs = [tokenize_text(ref) for ref in references]
    tokenized_preds = [tokenize_text(pred) for pred in predictions]

    # Build document frequency from references
    num_docs = len(tokenized_refs)

    cider_scores = []
    for n in range(1, max_n + 1):
        # Compute document frequency for this n-gram order across references
        doc_freq = defaultdict(int)
        ref_ngram_counts = []
        for ref_tokens in tokenized_refs:
            ngrams = _get_ngrams(ref_tokens, n)
            ngram_count = Counter(ngrams)
            ref_ngram_counts.append(ngram_count)
            for ngram in set(ngrams):
                doc_freq[ngram] += 1

        # Compute TF-IDF for each reference
        ref_tfidfs = []
        for ngram_count in ref_ngram_counts:
            tfidf = _compute_tfidf(ngram_count, doc_freq, num_docs)
            ref_tfidfs.append(tfidf)

        # Compute CIDEr_n for each prediction
        scores_n = []
        for pred_tokens in tokenized_preds:
            pred_ngrams = _get_ngrams(pred_tokens, n)
            pred_ngram_count = Counter(pred_ngrams)
            pred_tfidf = _compute_tfidf(pred_ngram_count, doc_freq, num_docs)

            # Average cosine similarity with all references
            sim_sum = 0.0
            for ref_tfidf in ref_tfidfs:
                sim_sum += _cosine_similarity(pred_tfidf, ref_tfidf)
            avg_sim = sim_sum / max(len(ref_tfidfs), 1)

            # Length penalty (Gaussian)
            avg_ref_len = np.mean([len(r) for r in tokenized_refs])
            pred_len = len(pred_tokens)
            length_diff = pred_len - avg_ref_len
            penalty = math.exp(-(length_diff ** 2) / (2 * (6 ** 2)))  # sigma=6

            scores_n.append(avg_sim * penalty)

        cider_scores.append(scores_n)

    # Average across n-gram orders and predictions
    cider_scores = np.array(cider_scores)  # shape: (max_n, num_preds)
    per_pred = cider_scores.mean(axis=0)   # average across n-gram orders
    avg_score = per_pred.mean()            # average across predictions

    # CIDEr is conventionally scaled by 10
    return round(avg_score * 10, 3)


