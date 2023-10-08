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

import re
import itertools

dpr = models.setup_closedbook(0)
cw_dict = load.load_words(only_ans=True)

def clean_string(s):
    return re.sub(r'[^\w\s]', '', s).lower()

import numpy as np

def similarity_score(bank, word):
    if isinstance(bank, list):
        if isinstance(bank[0], str):
            bank = ' '.join(bank)
        else:
            bank = ' '.join([b.word for b in bank])
    word = clean_string(word)
    preds, preds_scores = models.answer_clues(dpr, [bank], 999999999, output_strings=True)
    preds, preds_scores = preds[0], list(preds_scores[0])
    preds = [clean_string(pred) for pred in preds]
    # preds_scores = [-i for i in range(len(preds_scores))] # alternative scoring
    word_score = 0
    max_score = max(preds_scores)
    score_sum = np.sum(np.exp(np.array(preds_scores) - max_score))
    log_score_sum = np.log(score_sum) + max_score
    for i, pred in enumerate(preds):
        if pred == word:
            word_score = np.exp(preds_scores[i] - log_score_sum)
            break
    return word_score

class Word:
    def __init__(self, word, prob=1):
        self.word = word
        self.prob = prob
    
    def eval(self, ans):
        if self.word == ans:
            return self.prob
        else:
            return 0
    
class Operator:
    def __init__(self, bank=''):
        if isinstance(bank, str):
            self.bank0 = bank
            bank = bank.split()
            bank = [Word(b) for b in bank]
        else:
            if isinstance(bank[0], str):
                self.bank0 = ' '.join(bank)
                bank = [Word(b) for b in bank]
            else:
                self.bank0 = ' '.join([b.word for b in bank])
        self.bank = bank
        self.prob = 1
        for b in bank:
            self.prob *= b.prob
        
    def eval(self, ans):
        return 1

class Alternation(Operator):
    def __init__(self, bank):
        super().__init__(bank)
        
    def eval(self, ans):
        # TODO: recursive eval for nested operators
        merged = ''.join([b.word for b in self.bank])
        for i in range(len(merged)):
            j = 0
            for k in range(i, len(merged), 2):
                if merged[k] == ans[j]:
                    j += 1
                else:
                    break
            if j == len(ans):
                return self.prob
        return 0
    
class Anagram(Operator):
    def __init__(self, bank):
        super().__init__(bank)
        
    def eval(self, ans):
        # TODO: recursive eval for nested operators
        merged = ''.join([b.word for b in self.bank])
        if ''.join(sorted(merged)) == ''.join(sorted(ans)):
            return self.prob
        return 0
    
class Concatenation(Operator):
    def __init__(self, bank):
        super().__init__(bank)
        
    def eval(self, ans):
        # TODO: recursive eval for nested operators
        merged = ''.join([b.word for b in self.bank])
        if merged == ans:
            return self.prob
        return 0

class Container(Operator):
    def __init__(self, bank):
        super().__init__(bank)
        
    def eval(self, ans):
        # TODO
        raise NotImplementedError
    
class Deletion(Operator):
    def __init__(self, bank):
        super().__init__(bank)
        
    def eval(self, ans):
        # TODO
        raise NotImplementedError
    
class Hidden(Operator):
    def __init__(self, bank):
        super().__init__(bank)
        
    def eval(self, ans):
        # TODO: recursive eval for nested operators
        merged = ''.join([b.word for b in self.bank])
        if ans in merged: # doesn't enforce ans spans all words in bank
            return self.prob
        return 0
    
class Initialism(Operator):
    def __init__(self, bank):
        super().__init__(bank)
        
    def eval(self, ans):
        # TODO: recursive eval for nested operators
        merged = ''.join([b.word[0] for b in self.bank])
        if merged == ans:
            return self.prob
        return 0
    
class Terminalism(Operator):
    def __init__(self, bank):
        super().__init__(bank)
        
    def eval(self, ans):
        # TODO: recursive eval for nested operators
        merged = ''.join([b.word[-1] for b in self.bank])
        if merged == ans:
            return self.prob
        return 0
    
class Homophone(Operator):
    def __init__(self, bank):
        super().__init__(bank)
        
    def eval(self, ans):
        # TODO
        raise NotImplementedError

class Insertion(Operator):
    def __init__(self, bank):
        super().__init__(bank)
        
    def eval(self, ans):
        # TODO
        raise NotImplementedError

class Reversal(Operator):
    def __init__(self, bank):
        super().__init__(bank)
        
    def eval(self, ans):
        # TODO: recursive eval for nested operators
        merged = ''.join([b.word for b in self.bank])
        if merged[::-1] == ans:
            return self.prob
        return 0

class Substitution(Operator):
    def __init__(self, bank):
        super().__init__(bank)
        
    def eval(self, ans):
        # TODO: recursive eval for nested operators
        return similarity_score(self.bank0, ans)

class Definition:
    def __init__(self, defn):
        self.defn = defn
    
    def eval(self, ans):
        return similarity_score(self.defn, ans)

def eval(ops, defn, ans):
    return ops.eval(ans) * defn.eval(ans)

print("eval(Operator(), Definition(\"capturing\"), \"taking\") = ",
    eval(Operator(), Definition("capturing"), "taking"))
print("eval(Operator(), Definition(\"Provider of social introductions\"), \"toastmaster\") = ",
    eval(Operator(), Definition("Provider of social introductions"), "toastmaster"))
print("eval(Anagram(\"to a smart set\"), Definition(\"Provider of social introductions\"), \"toastmaster\") = ",
    eval(Anagram("to a smart set"), Definition("Provider of social introductions"), "toastmaster"))
print("eval(Operator(), Definition(\"Provider of social introductions\"), \"greeter\") = ",
    eval(Operator(), Definition("Provider of social introductions"), "greeter"))
print("eval(Anagram(\"to a smart set\"), Definition(\"Provider of social introductions\"), \"greeter\") = ",
    eval(Anagram("to a smart set"), Definition("Provider of social introductions"), "greeter"))