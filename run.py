import gradio as gr
import nltk
from nltk.util import ngrams
from nltk.tokenize import word_tokenize
from nltk.corpus import brown
from collections import defaultdict, Counter
import string
import random
import time

# Download needed NLTK data
nltk.download('punkt', quiet=True)
nltk.download('brown', quiet=True)

# --- N-gram Model Building ---
def build_ngram_models(sentences_list, n=3):
    trigram_model = defaultdict(Counter)
    bigram_model = defaultdict(Counter)
    unigram_counts = Counter()
    sentence_start = "<s>"
    sentence_end = "</s>"

    for sentence in sentences_list:
        words = [sentence_start, sentence_start] + [w.lower() for w in sentence] + [sentence_end]
        # Trigrams
        for w1, w2, w3 in ngrams(words, 3):
            trigram_model[(w1, w2)][w3] += 1
        # Bigrams
        for w1, w2 in ngrams(words, 2):
            if not (w1 == sentence_start and w2 == sentence_start):
                bigram_model[w1][w2] += 1
        # Unigrams
        unigram_counts.update(words)

    # Remove padding from unigram counts
    unigram_counts.pop(sentence_start, None)
    unigram_counts.pop(sentence_end, None)
    return trigram_model, bigram_model, unigram_counts

# Load Brown corpus and build raw counts
corpus_sentences = brown.sents()
trigram_counts, bigram_counts, unigram_counts = build_ngram_models(corpus_sentences)
V = len(unigram_counts)

# --- Precompute for Kneser-Ney ---
D = 0.75  # discount parameter

# Continuation counts: how many unique contexts each word completes
continuation_counts = Counter()
for w1, nexts in bigram_counts.items():
    for w2 in nexts:
        continuation_counts[w2] += 1
total_continuations = sum(continuation_counts.values())

# Type and token totals for contexts
trigram_types  = {ctx: len(nexts) for ctx, nexts in trigram_counts.items()}
trigram_totals = {ctx: sum(nexts.values()) for ctx, nexts in trigram_counts.items()}
bigram_types   = {w1: len(nexts) for w1, nexts in bigram_counts.items()}
bigram_totals  = {w1: sum(nexts.values()) for w1, nexts in bigram_counts.items()}

# --- Kneser-Ney Probability Functions ---
def p_continuation(w):
    return continuation_counts.get(w, 0) / total_continuations if total_continuations > 0 else 0


def p_kn_bigram(w, w_prev):
    counts = bigram_counts.get(w_prev, Counter())
    total = bigram_totals.get(w_prev, 0)
    c = counts.get(w, 0)
    if total > 0:
        lambda_bi = (D * bigram_types.get(w_prev, 0) / total)
        discounted = max(c - D, 0) / total
        return discounted + lambda_bi * p_continuation(w)
    else:
        return p_continuation(w)


def p_kn_trigram(w, w_prev2, w_prev1):
    ctx = (w_prev2, w_prev1)
    counts = trigram_counts.get(ctx, Counter())
    total = trigram_totals.get(ctx, 0)
    c = counts.get(w, 0)
    if total > 0:
        lambda_tri = (D * trigram_types.get(ctx, 0) / total)
        discounted = max(c - D, 0) / total
        return discounted + lambda_tri * p_kn_bigram(w, w_prev1)
    else:
        return p_kn_bigram(w, w_prev1)

# --- Prediction Function with Kneser-Ney ---
def predict_next_words(sentence_fragment, num_predictions=5):
    if not sentence_fragment:
        return "Please enter some text."
    try:
        words = [w.lower() for w in word_tokenize(sentence_fragment)]
    except Exception as e:
        return f"Error tokenizing input: {e}"

    num_preds = int(num_predictions)
    # If we have at least two words, use trigram smoothing
    if len(words) >= 2:
        w1, w2 = words[-2], words[-1]
        c_tri = set(trigram_counts.get((w1, w2), {}))
        c_bi  = set(bigram_counts.get(w2, {}))
        c_uni = set([w for w, _ in unigram_counts.most_common(50)])
        candidates = c_tri | c_bi | c_uni
        # Score and sort
        scored = [(w, p_kn_trigram(w, w1, w2)) for w in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        preds = [w for w, _ in scored if w != "</s>"]
        return "Predicted next words (Kneser-Ney):\n- " + "\n- ".join(preds[:num_preds])

    # If only one word, back off to bigram smoothing
    elif len(words) == 1:
        w2 = words[-1]
        c_bi  = set(bigram_counts.get(w2, {}))
        c_uni = set([w for w, _ in unigram_counts.most_common(50)])
        candidates = c_bi | c_uni
        scored = [(w, p_kn_bigram(w, w2)) for w in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        preds = [w for w, _ in scored if w != "</s>"]
        return "Predicted next words (Bigram Kneser-Ney):\n- " + "\n- ".join(preds[:num_preds])

    # Fallback to continuation unigram
    else:
        uni_sorted = sorted(unigram_counts.keys(), key=lambda w: p_continuation(w), reverse=True)
        return "Predicted next words (Unigram Continuation):\n- " + "\n- ".join(uni_sorted[:num_preds])

# --- Gradio Interface ---
iface = gr.Interface(
    fn=predict_next_words,
    inputs=[
        gr.Textbox(label="Enter Sentence Fragment", placeholder="Type the beginning of a sentence..."),
        gr.Slider(minimum=1, maximum=10, step=1, value=5, label="Number of Predictions")
    ],
    outputs=gr.Textbox(label="Predicted Next Words", lines=5),
    title="N-gram Next Word Predictor (Brown Corpus + Kneser-Ney Smoothing)",
    description="Uses Modified Kneser-Ney smoothing for tri-, bi- and unigram next-word prediction on the Brown corpus.",
    examples=[
        ["This is a"],
        ["predict the next"],
        ["in the city of"],
        ["ask not what your country"]
    ],
    allow_flagging='never'
)

if __name__ == "__main__":
    iface.launch()

