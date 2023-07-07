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

answer_freqs = {}
clues, defns, answers = [], [], []

for datapoint in data:
    clue, nondef, defn, ans, sz = load.unwrap_data(datapoint)
    if not clue or not defn or not ans:
        continue
    clues.append(clue)
    defns.append(defn)
    answers.append(ans)
    if ans not in answer_freqs:
        answer_freqs[ans] = 0
    answer_freqs[ans] += 1

use_clues = True

if use_clues:
    try:
        with open('clues.pkl', 'rb') as f:
            pred_ranks = pickle.load(f)
    except:
        pred_ranks = []
else:
    try:
        with open('defns.pkl', 'rb') as f:
            pred_ranks = pickle.load(f)
    except:
        pred_ranks = []

def gen(clues, file, max_answers=10000):
    preds, pred_scores = models.answer_clues(dpr, clues, max_answers, output_strings=True)
    
    for i in range(len(preds)):
        combined = []
        for j in range(len(preds[i])):
            p = preds[i][j].lower().replace(' ', '')
            if p in answer_freqs:
                combined.append((pred_scores[i][j] * np.log(answer_freqs[p]), p))
            else:
                combined.append((pred_scores[i][j] * np.log(0.5), p))
        combined.sort(reverse=True)
        pred_scores[i] = [c[0] for c in combined]
        preds[i] = [c[1] for c in combined]
    
    for i, pred in enumerate(preds):
        pred = [p for p in pred if len(p) - p.count(' ') == len(answers[i]) - answers[i].count(' ')]
        if answers[i].lower().replace(' ', '') not in pred:
            pred_ranks.append(max_answers)
        else:
            pred_ranks.append(pred.index(answers[i].lower().replace(' ', '')) + 1)
            
    with open(file, 'wb') as f:
        pickle.dump(pred_ranks, f)

while True:
    c = list(zip(clues, defns, answers))
    random.shuffle(c)
    clues, defns, answers = zip(*c)
    trunc = 10

    if use_clues:
        gen(clues[:trunc], 'clues.pkl')
    else:
        gen(defns[:trunc], 'defns.pkl')