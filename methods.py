import pandas as pd
import re
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer, T5ForConditionalGeneration
import tokenizers
import json
import puz
import os
import numpy as np
import streamlit as st
import scipy

import sys
import subprocess
import copy
import json

from itertools import zip_longest
from copy import deepcopy
import regex

from solver.Crossword import Crossword
from solver.BPSolver import BPSolver
from models import setup_closedbook, setup_t5_reranker, DPRForCrossword
from solver.Utils import print_grid

from utils import puz_to_json

import load
import models
import random

dpr = models.setup_closedbook(0)

##############################
# NO MULTI-CLUE BATCHING YET #
##############################

def filter(preds, pred_scores, sz, cw_dict, get_rank):
    preds = [p.lower() for p in preds]
    zipped = list(zip(preds, pred_scores))
    zipped = [z for z in zipped if z[0] in cw_dict and len(z[0]) == sz]
    d = {}
    if get_rank:
        for i, (pred, score) in enumerate(sorted(zipped, key=lambda x: x[1], reverse=True)):
            d[pred] = [i, score]
    else:
        for pred, score in zipped:
            d[pred] = score
    return d

def basic(clue, sz, cw_dict, indic_dict, get_rank=True):
    preds, pred_scores = models.answer_clues(dpr, [clue], 999999999, output_strings=True)
    preds, pred_scores = preds[0], list(pred_scores[0])
    return filter(preds, pred_scores, sz, cw_dict, get_rank)

# returns array of possible (indicator, type, bank) tuples
def extract_indic(words, indic_dict):
    indic = []
    for lo in range(len(words)):
        for hi in range(lo, len(words)):
            if lo == 0 and hi == len(words) - 1:
                continue
            joint = ' '.join(words[lo:hi+1])
            if joint in indic_dict:
                for indic_type, freq in indic_dict[joint]:
                    indic.append((joint, indic_type, freq, words[:lo] + words[hi+1:]))
    return indic

# returns array of possible (definition, indicator, type, freq, bank) tuples
def gen_combos(clue, indic_dict, thresh=0, sorted=True):
    words = clue.split()
    combos = []
    for idx in range(1, len(words) - 1):
        pre = ' '.join(words[:idx])
        suf = ' '.join(words[idx:])
        pre_indic = extract_indic(words[:idx], indic_dict)
        suf_indic = extract_indic(words[idx:], indic_dict)
        for indic, indic_type, freq, bank in pre_indic:
            if freq >= thresh:
                combos.append((suf, indic, indic_type, freq, bank))
        for indic, indic_type, freq, bank in suf_indic:
            if freq >= thresh:
                combos.append((pre, indic, indic_type, freq, bank))
    if sorted:
        combos.sort(key=lambda x: x[3], reverse=True)
    return combos

def baseline_alternation(pred, bank):
    return False, None

def baseline_anagram(pred, bank):
    if len(bank) != 1:
        return False, None
    pred_decomp = list(pred)
    pred_decomp.sort()
    bank_decomp = list(bank[0])
    bank_decomp.sort()
    if pred_decomp == bank_decomp:
        return True, bank[0]
    return False, None

def baseline_container(pred, bank):
    return False, None

def baseline_deletion(pred, bank):
    return False, None

def baseline_hidden(pred, bank):
    return False, None

def baseline_homophone(pred, bank):
    return False, None

def baseline_insertion(pred, bank):
    return False, None

def baseline_reversal(pred, bank):
    if len(bank) != 1:
        return False, None
    if pred[::-1] == bank[0]:
        return True, bank[0]
    return False, None

def baseline_dbl_defn(clue, sz, cw_dict, thresh_rank=np.inf, thresh_score=0, get_rank=True):
    words = clue.split()
    union = {}
    for idx in range(1, len(words) - 1):
        pre = ' '.join(words[:idx])
        suf = ' '.join(words[idx:])
        preds, pred_scores = models.answer_clues(dpr, [pre, suf], 999999999, output_strings=True)
        pre_dict = filter(preds[0], pred_scores[0], sz, cw_dict, get_rank=True)
        suf_dict = filter(preds[1], pred_scores[1], sz, cw_dict, get_rank=True)
        for pred in pre_dict:
            if pred in suf_dict and \
                pre_dict[pred][0] <= thresh_rank and \
                suf_dict[pred][0] <= thresh_rank and \
                pre_dict[pred][1] >= thresh_score and \
                suf_dict[pred][1] >= thresh_score:
                if not pred in union:
                    union[pred] = []
                union[pred].append(pre_dict[pred][1])
                union[pred].append(suf_dict[pred][1])
    condensed = {}
    for pred in union:
        condensed[pred] = 1 / np.mean([1 / s for s in union[pred]])
    ranked = sorted(condensed.items(), key=lambda x: -x[1])
    final = {}
    if get_rank:
        for i, (pred, score) in enumerate(ranked):
            final[pred] = (i+1, score)
    else:
        for pred, score in ranked:
            final[pred] = score
    return final

def baseline(clue, sz, cw_dict, indic_dict, get_rank=True):    
    combos = gen_combos(clue, indic_dict)
    batch = []
    for defn, indic, indic_type, freq, bank in combos:
        batch.append(defn)
    preds, pred_scores = models.answer_clues(dpr, batch, 999999999, output_strings=True)
    filtered = []
    for i in range(len(preds)):
        filtered.append(filter(preds[i], pred_scores[i], sz, cw_dict, get_rank=False))
    union = {}
    type_funcs = {
        "alternation": baseline_alternation,
        "anagram": baseline_anagram,
        "container": baseline_container,
        "deletion": baseline_deletion,
        "hidden": baseline_hidden,
        "homophone": baseline_homophone,
        "insertion": baseline_insertion,
        "reversal": baseline_reversal
    }
    for i in range(len(combos)):
        defn, indic, indic_type, freq, bank = combos[i]
        for pred, score in filtered[i].items():
            ok, used = type_funcs[indic_type](pred, bank)
            if not pred in union:
                union[pred] = []
            union[pred].append((ok, score, defn, indic, indic_type, freq, bank, used))
    # double definitions are special
    dbl_defn = baseline_dbl_defn(clue, sz, cw_dict, thresh_rank=100, thresh_score=0, get_rank=False)
    for pred in dbl_defn:
        if not pred in union:
            union[pred] = []
        union[pred].append((True, dbl_defn[pred], None, None, None, None, None, None))
    condensed = {}
    for pred in union:
        oks = False
        scores = []
        for ok, score, defn, indic, indic_type, freq, bank, used in union[pred]:
            oks |= ok
            scores.append(score)
        avg_score = 1 / np.mean([1 / s for s in scores])
        condensed[pred] = (oks, avg_score)
    ranked = sorted(condensed.items(), key=lambda x: (-x[1][0], -x[1][1])) # for reverse
    final = {}
    if get_rank:
        for i, (pred, (oks, avg_score)) in enumerate(ranked):
            final[pred] = (i+1, avg_score)
    else:
        for pred, (oks, avg_score) in ranked:
            final[pred] = avg_score
    return final