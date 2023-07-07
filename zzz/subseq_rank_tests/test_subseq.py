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
import pickle

random.seed()

data = load.load_data()
dpr = models.setup_closedbook(0)

clues, defns, answers = [], [], []

for datapoint in data:
    clue, nondef, defn, ans, sz = load.unwrap_data(datapoint)
    if not clue or not defn or not ans:
        continue
    clues.append(clue)
    defns.append(defn)
    answers.append(ans)

use_clues = False

# how many preds are subsequences of the clue
# if the answer is a subseq of the clue, what is its rank among preds that are subseqs of the clue
# when the answer not a subseq of the clue, how often are the preds subseqs of the clue
subseq_rate, answer_subseq_rank, false_subseq_rate = [0, 0], [0, 0], [0, 0]

if use_clues:
    with open('subseq_clues.pkl', 'rb') as f:
        subseq_rate, answer_subseq_rank, false_subseq_rate = pickle.load(f)
else:
    with open('subseq_defns.pkl', 'rb') as f:
        subseq_rate, answer_subseq_rank, false_subseq_rate = pickle.load(f)

def is_subseq(clue, ans):
    clue = clue.lower().replace(' ', '')
    ans = ans.lower().replace(' ', '')
    i = 0
    for c in clue:
        if i == len(ans):
            return True
        if c == ans[i]:
            i += 1
    return False

def gen(clues, answers, file, max_answers=1000000):
    max_answers = 1000000
    preds, pred_scores = models.answer_clues(dpr, clues, max_answers, output_strings=True)

    for i, pred in enumerate(preds):
        pred = [p.lower().replace(' ', '') for p in pred if len(p) - p.count(' ') == len(answers[i]) - answers[i].count(' ')]
        subseq_preds = [p for p in pred if is_subseq(clues[i], p)]
        subseq_rate[0] += len(subseq_preds)
        subseq_rate[1] += len(pred)
        answer_subseq = is_subseq(clues[i], answers[i])
        if answer_subseq:
            answer_subseq_rank[1] += 1
            if answers[i] in subseq_preds:
                answer_subseq_rank[0] += subseq_preds.index(answers[i]) + 1
            else:
                answer_subseq_rank[0] += 1000000
        else:
            false_subseq_rate[0] += len(subseq_preds)
            false_subseq_rate[1] += len(pred)
        
    with open(file, 'wb') as f:
        pickle.dump((subseq_rate, answer_subseq_rank, false_subseq_rate), f)

while True:
    c = list(zip(clues, defns, answers))
    random.shuffle(c)
    clues, defns, answers = zip(*c)
    trunc = 10

    if use_clues:
        gen(clues[:trunc], answers[:trunc], 'subseq_clues.pkl')
    else:
        gen(defns[:trunc], answers[:trunc], 'subseq_defns.pkl')