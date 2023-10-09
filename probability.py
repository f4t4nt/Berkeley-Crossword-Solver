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

def clean_string(s, remove_spaces=True):
    s = re.sub(r'[^a-zA-Z0-9\s]', '', s)
    s = s.lower()
    if remove_spaces:
        s = re.sub(r'\s+', '', s)
    return s

def synonyms(bank):
    if isinstance(bank, list):
        bank = ' '.join(bank)
    preds, preds_scores = models.answer_clues(dpr, [bank], 999999999, output_strings=True)
    preds, preds_scores = preds[0], list(preds_scores[0])
    preds = [clean_string(p, remove_spaces=False) for p in preds]
    preds_map = {}
    max_score = max(preds_scores)
    score_sum = np.sum(np.exp(np.array(preds_scores) - max_score))
    log_score_sum = np.log(score_sum) + max_score
    for i, pred in enumerate(preds):
        preds_map[pred] = np.exp(preds_scores[i] - log_score_sum)
    return preds_map

class Operator:
    def __init__(self, bank=''):
        if isinstance(bank, str):   # pure string, need to parse
            bank = bank.split()
            bank = [{b: 1} for b in bank]
        else:                       # list of Words and Operators, no change necessary
            pass
        self.bank = bank
        
    def all_banks(self, i=0):
        if i >= len(self.bank):
            yield ([], 1)
        else:
            for w, w_s in self.bank[i].items():
                for b, b_s in self.all_banks(i + 1):
                    yield ([w] + b, w_s * b_s)
        
    def eval(self):
        new_bank = []
        for b in self.bank:
            if isinstance(b, dict):
                new_bank.append(b)
            else:
                new_bank.append(b.eval())
        self.bank = new_bank
        return {}

class Alternation(Operator):
    def __init__(self, bank):
        super().__init__(bank)
        
    def eval(self):
        super().eval()
        options = {}
        for b, b_s in self.all_banks():
            merged = clean_string(''.join(b))
            for i in range(len(merged)):
                cur = merged[i]
                for j in range(i + 2, len(merged), 2):
                    cur += merged[j]
                    if len(cur) >= 3: # for now
                        if not cur in options:
                            options[cur] = 0
                        options[cur] += b_s
        return options

class Anagram(Operator):
    def __init__(self, bank):
        super().__init__(bank)
        
    def eval(self):
        super().eval()
        options = {}
        for b, b_s in self.all_banks():
            merged = clean_string(''.join(b))
            assert len(merged) <= 15 # for now
            for perm in itertools.permutations(merged):
                cur = ''.join(perm)
                if not cur in options:
                    options[cur] = 0
                options[cur] += b_s
        return options
    
class Concatenation(Operator):
    def __init__(self, bank):
        super().__init__(bank)
        
    def eval(self):
        super().eval()
        options = {}
        for b, b_s in self.all_banks():
            merged = clean_string(''.join(b))
            if not merged in options:
                options[merged] = 0
            options[merged] += b_s
        return options

class Container(Operator):
    def __init__(self, bank):
        super().__init__(bank)
        
    def eval(self):
        super().eval()
        options = {}
        for b, b_s in self.all_banks():
            assert len(b) == 2 # for now
            for i in range(len(b[0]) + 1):
                cur = b[0][:i] + b[1] + b[0][i:]
                if not cur in options:
                    options[cur] = 0
                options[cur] += b_s
            for i in range(len(b[1]) + 1):
                cur = b[1][:i] + b[0] + b[1][i:]
                if not cur in options:
                    options[cur] = 0
                options[cur] += b_s
        return options

# not really sure if this is distinct to other ops..
# https://en.wikipedia.org/wiki/Cryptic_crossword#Deletions
# https://cryptics.georgeho.org/data/clues/234180
class Deletion(Operator):
    def __init__(self, bank):
        super().__init__(bank)
        
    def eval(self):
        # TODO
        raise NotImplementedError
    
class Hidden(Operator):
    def __init__(self, bank):
        super().__init__(bank)
        
    def eval(self):
        super().eval()
        options = {}
        for b, b_s in self.all_banks():
            merged = clean_string(''.join(b))
            for i in range(len(merged)):
                for j in range(i + 1, len(merged)): # need to enforce using all words
                    cur = merged[i:j]
                    if len(cur) >= 3: # for now
                        if not cur in options:
                            options[cur] = 0
                        options[cur] += b_s
        return options
    
class Initialism(Operator):
    def __init__(self, bank):
        super().__init__(bank)
        
    def eval(self):
        super().eval()
        options = {}
        for b, b_s in self.all_banks():
            merged = clean_string(''.join([b_[0] for b_ in b]))
            if not merged in options:
                options[merged] = 0
            options[merged] += b_s
        return options
    
class Terminalism(Operator):
    def __init__(self, bank):
        super().__init__(bank)
        
    def eval(self):
        super().eval()
        options = {}
        for b, b_s in self.all_banks():
            merged = clean_string(''.join([b_[-1] for b_ in b]))
            if not merged in options:
                options[merged] = 0
            options[merged] += b_s
        return options
    
class Homophone(Operator):
    def __init__(self, bank):
        super().__init__(bank)
        
    def eval(self):
        # TODO
        raise NotImplementedError

# not really sure if this is distinct to other ops..
# https://cryptics.georgeho.org/data/clues/464087
# https://cryptics.georgeho.org/data/clues/2313
class Insertion(Operator):
    def __init__(self, bank):
        super().__init__(bank)
        
    def eval(self):
        # TODO
        raise NotImplementedError

class Reversal(Operator):
    def __init__(self, bank):
        super().__init__(bank)
        
    def eval(self):
        super().eval()
        options = {}
        for b, b_s in self.all_banks():
            merged = clean_string(''.join(b))[::-1]
            if not merged in options:
                options[merged] = 0
            options[merged] += b_s
        return options

class Substitution(Operator):
    def __init__(self, bank):
        super().__init__(bank)
        
    def eval(self):
        super().eval()
        options = {}
        for b, b_s in self.all_banks():
            merged = ' '.join(b)
            syns = synonyms(merged)
            for syn, syn_s in syns.items():
                syn = clean_string(syn)
                if not syn in options:
                    options[syn] = 0
                options[syn] += b_s * syn_s
        options = sorted(options.items(), key=lambda x: -x[1])
        options = options[:1000]
        options = {o[0]: o[1] for o in options}
        return options

class Definition:
    def __init__(self, defn):
        self.defn = defn
    
    def eval(self):
        return synonyms(self.defn)

def eval(part1, part2, ans):
    term1 = part1.eval().get(ans, 0)
    term2 = part2.eval().get(ans, 0)
    print(term1, "*", term2, "=", term1 * term2)
    return term1 * term2

# Concatenation([Substitution("A long arduous journey, especially one made on foot."), Substitution("chess piece")]), Definition("Walking"), "trekking"
print("eval(Concatenation([Substitution(\"A long arduous journey, especially one made on foot.\"), Substitution(\"chess piece\")]), Definition(\"Walking\"), \"trekking\") = ",
    eval(Concatenation([Substitution("A long arduous journey, especially one made on foot."), Substitution("chess piece")]), Definition("Walking"), "trekking"))
# Anagram("to a smart set"), Definition("Provider of social introductions"), "toastmaster"
print("eval(Anagram(\"to a smart set\"), Definition(\"Provider of social introductions\"), \"toastmaster\") = ",
    eval(Anagram("to a smart set"), Definition("Provider of social introductions"), "toastmaster"))
# Anagram("to a smart set"), Definition("Provider of social introductions"), "greeter"
print("eval(Anagram(\"to a smart set\"), Definition(\"Provider of social introductions\"), \"greeter\") = ",
    eval(Anagram("to a smart set"), Definition("Provider of social introductions"), "greeter"))
# Odd stuff of Mr. Waugh is set for someone wanting women to vote (10)
# [Odd] Alternation("stuff of Mr. Waugh is set for"), Definition("someone wanting women to vote"), "suffragist"
print("eval(Alternation(\"stuff of Mr. Waugh is set for\"), Definition(\"someone wanting women to vote\"), \"suffragist\") = ",
    eval(Alternation("stuff of Mr. Waugh is set for"), Definition("someone wanting women to vote"), "suffragist"))
# Outlaw leader managing money (7)
# Concatenation([Substitution("Outlaw"), Substitution("leader")]), Definition("managing money"), "banking"
print("eval(Concatenation([Substitution(\"Outlaw\"), Substitution(\"leader\")]), Definition(\"managing money\"), \"banking\") = ",
    eval(Concatenation([Substitution("Outlaw"), Substitution("leader")]), Definition("managing money"), "banking"))
# Country left judgeable after odd losses (8)
# Definition("Country"), Concatenation([Substitution("left"), Alternation("judgeable")]) [after odd losses], "portugal"
print("eval(Definition(\"Country\"), Concatenation([Substitution(\"left\"), Alternation(\"judgeable\")]), \"portugal\") = ",
    eval(Definition("Country"), Concatenation([Substitution("left"), Alternation("judgeable")]), "portugal"))
# Shade’s a bit circumspect, really (7)
# Definition("Shade's"), [a bit] Hidden("circumspect, really"), "spectre"
print("eval(Definition(\"Shade's\"), Hidden(\"circumspect, really\"), \"spectre\") = ",
    eval(Definition("Shade's"), Hidden("circumspect, really"), "spectre"))
# Speak about idiot making sense (6)
# Container([Substitution("Speak") [about], Substitution("idiot")]) [making], Definition("sense"), "sanity"
print("eval([Substitution(\"Speak\") [about], Substitution(\"idiot\")]), Definition(\"sense\"), \"sanity\") = ",
    eval(Container([Substitution("Speak"), Substitution("idiot")]), Definition("sense"), "sanity"))
# A bit of god-awful back trouble (3)
# [A bit of] Reversal([Hidden("god-awful")]) [back], Definition("trouble"), "ado"
print("eval([Hidden(\"god-awful\")]), Definition(\"trouble\"), \"ado\") = ",
    eval(Reversal([Hidden("god-awful")]), Definition("trouble"), "ado"))
# Quangos siphoned a certain amount off, creating scandal (6)
# Hidden("Quangos siphoned") [a certain amount off], Definition("creating scandal"), "gossip"
print("eval(Hidden(\"Quangos siphoned\"), Definition(\"creating scandal\"), \"gossip\") = ",
    eval(Hidden("Quangos siphoned"), Definition("creating scandal"), "gossip"))
# Bird is cowardly, about to fly away (5)
# Definition("Bird"), [is] Hidden([Substitution("cowardly,")]) [about to fly away], "raven"
print("eval(Definition(\"Bird\"), Hidden([Substitution(\"cowardly,\")]), \"raven\") = ",
    eval(Definition("Bird"), Hidden([Substitution("cowardly,")]), "raven"))

# Works, just anagramming "decaffeinated" is too slow
# # As is a less stimulating cup defeat, faced in a bad way (13)
# # Definition("As is a less stimulating cup"), Anagram("defeat, faced in") [a bad way], "decaffeinated"
# print("eval(Definition(\"As is a less stimulating cup\"), Anagram(\"defeat, faced in\"), \"decaffeinated\") = ",
#     eval(Definition("As is a less stimulating cup"), Anagram("defeat, faced in"), "decaffeinated"))