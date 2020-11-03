from typing import List, Dict
import sys
import collections
import math
import csv
import spacy
from tqdm import tqdm
from collections import Counter

def get_ngram_counter(text, n):
    """
    Returns a counter, indicating how many times each n-gram appeared in text.
    Note: this function does NOT lowercase text. If you want to lowercase, you should
    do so before calling this function.
    Input:
      text: is a string, with tokens space-separated.
    Returns:
      counter: mapping from each n-gram (a space-separated string) appearing in text,
        to the number of times it appears
    """
    ngrams = [" ".join(text[i:i+n]) for i in range(len(text)-(n-1))]  # list of str
    counter = Counter()
    counter.update(ngrams)
    return counter

def _distinct_n(sample, n):
    """
    Returns (total number of unique ngrams in story_text) / (total number of ngrams in story_text, including duplicates).
    Text is lowercased before counting ngrams.
    Returns None if there are no ngrams
    """
    # ngram_counter maps from each n-gram to how many times it appears
    ngram_counter = get_ngram_counter(sample, n)
    if sum(ngram_counter.values()) == 0:
        print("Warning: encountered a story with no {}-grams".format(n))
        print(sample)
        print("ngram_counter: ", ngram_counter)
        return 0
    return len(ngram_counter) / sum(ngram_counter.values())


def _get_ngrams(segment, max_order):
    """Extracts all n-grams upto a given maximum order from an input segment.
    Args:
        segment: text segment from which n-grams will be extracted.
        max_order: maximum length in tokens of the n-grams returned by this
        methods.
    Returns:
        The Counter containing all n-grams upto max_order in segment
        with a count of how many times each n-gram occurred.
    """
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i+order])
            ngram_counts[ngram] += 1
    return ngram_counts


def _compute_bleu(reference_corpus, translation_corpus, max_order=4, smooth=False):
    """Computes BLEU score of translated segments against one or more references.
    Args:
        reference_corpus: list of lists of references for each translation. Each
            reference should be tokenized into a list of tokens.
        translation_corpus: list of translations to score. Each translation
            should be tokenized into a list of tokens.
        max_order: Maximum n-gram order to use when computing BLEU score.
        smooth: Whether or not to apply Lin et al. 2004 smoothing.
    Returns:
        3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
            precisions and brevity penalty.
    """
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    reference_length = 0
    translation_length = 0
    for (references, translation) in zip(reference_corpus, translation_corpus):
        #print(references)
        #print(translation)
        reference_length += min(len(r) for r in references)
        translation_length += len(translation)

        merged_ref_ngram_counts = collections.Counter()
        for reference in references:
            merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
        translation_ngram_counts = _get_ngrams(translation, max_order)
        overlap = translation_ngram_counts & merged_ref_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram)-1] += overlap[ngram]
        for order in range(1, max_order+1):
            possible_matches = len(translation) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order-1] += possible_matches

    precisions = [0] * max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = ((matches_by_order[i] + 1.) /
                             (possible_matches_by_order[i] + 1.))
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (float(matches_by_order[i]) /
                                 possible_matches_by_order[i])
            else:
                precisions[i] = 0.0

    if min(precisions) > 0:
        p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    ratio = float(translation_length) / reference_length

    if ratio > 1.0:
        bp = 1.
    else:
        bp = math.exp(1 - 1. / ratio)

    bleu = geo_mean * bp

    return (bleu, precisions, bp, ratio, translation_length, reference_length)


def read(filename):
    with open(filename, 'r') as f:
        data = [line.strip() for line in f.readlines()]
    return data

def read_reference(filename):
    data = []
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for line in reader:
            data.append(line[1:])
    return data


def distinct_ngrams(inputs, n):
    output = {}
    for input in inputs:
        for i in range(len(input)-n+1):
            g = ' '.join(input[i:i+n])
            output.setdefault(g, 0)
            output[g] += 1
    if sum(output.values())==0:
        ratio = 0
    else:
        ratio = float(len(output.keys()))/ sum(output.values())

    return ratio

GT_FILE = 'data/story/test/target.csv'

def evaluate(GEN_FILE):
    nlp = spacy.load("en_core_web_sm")
    preds = read(GEN_FILE)
    gts = read_reference(GT_FILE)
    res = {'R2':0.0, 'RL': 0.0, 'B1': 0.0, 'B2': 0.0, 'B3': 0.0, 'B4': 0.0, "D1":0.0, "D2":0.0, "D3":0.0, "D4":0.0}
    gt_bleu = []
    pred_bleu = []
    rl_f = open(GEN_FILE + '.rl', 'w')
    for gt, pred in tqdm(zip(gts, preds)):

        
        pred_list = [x.text for x in nlp(pred.strip())]
        #pred_list = pred.split()

        pred_bleu.append(pred_list)
        
        gt_list = [[x.text for x in nlp(y.strip())] for y in gt]
        #gt_list = [x.split() for x in gt]
        
        gt_bleu.append(gt_list)

        


    res["D1"] += distinct_ngrams(pred_bleu, 1)
    res["D2"] += distinct_ngrams(pred_bleu, 2)
    res["D3"] += distinct_ngrams(pred_bleu, 3)
    res["D4"] += distinct_ngrams(pred_bleu, 4)

    res['B4'] = _compute_bleu(gt_bleu, pred_bleu, max_order=4)[0]
    res['B3'] = _compute_bleu(gt_bleu, pred_bleu, max_order=3)[0]
    res['B2'] = _compute_bleu(gt_bleu, pred_bleu, max_order=2)[0]
    res['B1'] = _compute_bleu(gt_bleu, pred_bleu, max_order=1)[0]
    rl_f.close()
    
    print('Bleu-1: {:.4f} | Bleu-2: {:.4f} | Bleu-3 {:.4f} | Bleu-4 {:.4f} | Dist-1 {:.4f} | Dist-2 {:.4f} | Dist-3 {:.4f} | Dist-4 {:.4f}'.format(
                        res['B1'],res['B2'],res['B3'],res['B4'],
                        res['D1'],res['D2'],res['D3'],res['D4']))

GEN_FILE = sys.argv[1]
evaluate(GEN_FILE)

