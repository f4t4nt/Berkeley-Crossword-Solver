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

def clean_string(s, remove_spaces=True):
    s = re.sub(r'[^a-zA-Z0-9\s]', '', s)
    s = s.lower()
    if remove_spaces:
        s = re.sub(r'\s+', '', s)
    return s

def check_chars(word, fixed, fixed_len=False):
    if fixed_len and len(word) != len(fixed):
        return False
    for i, c in enumerate(word):
        if i >= len(fixed):
            return False # just for now
        if not c in fixed[i]:
            return False
    return True

def valid_cut(cut, bank):
    for i in range(len(bank)):
        if isinstance(bank[i], str) and not check_chars(bank[i], cut[i], fixed_len=True):
            return False
    return True

def merge_cuts(cuts, size, cnt, bank):
    rv = [[dict() for _ in range(size)] for _ in range(cnt)]
    for cut in cuts:
        if not valid_cut(cut, bank):
            continue
        for i, c in enumerate(cut):
            for j, chars in enumerate(c):
                for k, v in chars.items():
                    if k in rv[i][j]:
                        rv[i][j][k] = v # constant probability for now
                    else:
                        rv[i][j][k] = v
    return rv

def get_cuts(base, cnt):
    if cnt == 1:
        return [[base]]
    else:
        cuts = []
        for i in range(len(base) - cnt + 1):
            for p in get_cuts(base[i + 1:], cnt - 1):
                cuts.append([base[:i + 1]] + p)
        return cuts

# cuts = get_cuts([{'j': 1}, {'o': 1}, {'r': 1}, {'d': 1}, {'a': 1}, {'n': 1}], 3)
# # print(cuts)
# merged = merge_cuts(cuts, 'jordan', 3, [1, 2, 3])
# for d in merged:
#     print(d)

# # slicedapplebreadcrust
# #                 board
# # cuts = get_cuts([{'s': 1}, {'l': 1}, {'i': 1}, {'c': 1}, {'e': 1}, {'d': 1}, {'a': 1}, {'p': 1}, {'p': 1}, {'l': 1}, {'e': 1}, {'b': 1}, {'r': 1}, {'e': 1}, {'a': 1}, {'d': 1}, {'c': 1, 'b': 1}, {'r': 1, 'o': 1}, {'u': 1, 'a': 1}, {'s': 1, 'r': 1}, {'t': 1, 'd': 1}], 4)
# cuts = get_cuts([{'s': 1}, {'l': 1}, {'i': 1}, {'c': 1}, {'e': 1}, {'d': 1}, {'a': 1}, {'p': 1}, {'p': 1}, {'l': 1}, {'e': 1}, {'b': 1}, {'r': 1}, {'e': 1}, {'a': 1}, {'d': 1}, {'c': 1, 'b': 1}, {'r': 1, 'o': 1}, {'u': 1, 'a': 1}, {'s': 1, 'r': 1}, {'t': 1}], 4)
# # print(cuts)
# merged = merge_cuts(cuts, len('slicedapplebreadcrust'), 4, [1, 'apple', 3, 4])
# for d in merged:
#     print(d)

dpr = models.setup_closedbook(0)
cw_dict = load.load_words(only_ans=True)

def apply_cond(word, **kwargs):
    exact_len = kwargs.get('exact_len')
    min_len = kwargs.get('min_len', 1)
    max_len = kwargs.get('max_len', 100)
    fixed = kwargs.get('fixed')
    in_dict = kwargs.get('in_dict', False)
    
    if exact_len:
        min_len = exact_len
        max_len = exact_len
    return len(word) >= min_len \
        and len(word) <= max_len \
        and (not fixed or check_chars(word, fixed)) \
        and (not in_dict or word in cw_dict)

def synonyms(bank, **kwargs):
    if isinstance(bank, list):
        bank = ' '.join(bank)
    preds, preds_scores = models.answer_clues(dpr, [bank], 2000, output_strings=True)
    preds, preds_scores = preds[0], list(preds_scores[0])
    ### Optimize
    preds = [clean_string(p) for p in preds]
    new_preds = [p for p in preds if apply_cond(p, **kwargs)]
    new_preds_scores = [preds_scores[i] for i, p in enumerate(preds) if apply_cond(p, **kwargs)]
    ###
    preds = new_preds
    preds_scores = new_preds_scores
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
        self.eval_factor = False
        self.factor = 0 # should never be 0
        
    def all_banks(self, i=0):
        if i >= len(self.bank):
            yield ([], 1)
        else:
            for w, w_s in self.bank[i].items():
                for b, b_s in self.all_banks(i + 1):
                    yield ([w] + b, w_s * b_s)
    
    def net_factor(self):
        if not self.eval_factor:
            for b in self.bank:
                if isinstance(b, Operator):
                    self.factor *= b.net_factor()
            self.eval_factor = True
        return self.factor
    
    def eval(self, **kwargs):
        if 'exact_len' in kwargs:
            kwargs['max_len'] = kwargs['exact_len']
            del kwargs['exact_len']
        if 'in_dict' in kwargs:
            del kwargs['in_dict']
            
        # new_bank = []
        # for b in self.bank:
        #     if isinstance(b, dict):
        #         new_bank.append(b)
        #     else:
        #         new_bank.append(b.eval(**kwargs))
        
        if 'fixed' in kwargs: # janky for now, need to collapse to one statement
            fixed = kwargs.get('fixed')
            cuts = get_cuts(fixed, len(self.bank))
            merged = merge_cuts(cuts, len(fixed), len(self.bank), self.bank)
            
            self.net_factor()
            new_bank = self.bank
            branch = []
            for i, b in enumerate(self.bank):
                if isinstance(b, Operator):
                    branch.append([i, b.factor])
            branch.sort(key=lambda x: x[1])
            branch = [b[0] for b in branch]
            for i in branch:
                copy_args = deepcopy(kwargs)
                copy_args['fixed'] = merged[i]
                new_bank[i] = new_bank[i].eval(**copy_args)
        else:
            self.net_factor()
            new_bank = self.bank
            branch = []
            for i, b in enumerate(self.bank):
                if isinstance(b, Operator):
                    branch.append([i, b.factor])
            branch.sort(key=lambda x: x[1])
            branch = [b[0] for b in branch]
            for i in branch:
                new_bank[i] = new_bank[i].eval(**kwargs)
                
        return {}

class Alternation(Operator):
    def __init__(self, bank):
        super().__init__(bank)
        self.factor = 100 # O(n^2)
        
    def eval(self, **kwargs):
        super().eval() # alternation removes letters
        self.options = {}
        for b, b_s in self.all_banks():
            merged = clean_string(''.join(b))
            for i in range(len(merged)):
                cur = merged[i]
                for j in range(i + 2, len(merged), 2):
                    cur += merged[j]
                    if apply_cond(cur, **kwargs):
                        if not cur in self.options:
                            self.options[cur] = 0
                        self.options[cur] += b_s
        return self.options

class Anagram(Operator):
    def __init__(self, bank):
        super().__init__(bank)
        self.factor = 1000 # O(n!)
        
    def eval(self, **kwargs):
        super().eval(**kwargs)
        self.options = {}
        in_dict = kwargs.get('in_dict', False)
        for b, b_s in self.all_banks():
            merged = clean_string(''.join(b))
            if not in_dict:
                for perm in itertools.permutations(merged):
                    cur = ''.join(perm)
                    if apply_cond(cur, **kwargs):
                        if not cur in self.options:
                            self.options[cur] = 0
                        self.options[cur] = b_s # we don't count a1a2 and a2a1 as distinct
            else:
                for word in cw_dict:
                    if len(word) == len(merged) and sorted(word) == sorted(merged) and apply_cond(word, **kwargs):
                        if not word in self.options:
                            self.options[word] = 0
                        self.options[word] = b_s # we don't count a1a2 and a2a1 as distinct
        return self.options
    
class Concatenation(Operator):
    def __init__(self, bank):
        super().__init__(bank)
        self.factor = 1 # O(1)
        
    def eval(self, **kwargs):
        super().eval(**kwargs)
        self.options = {}
        for b, b_s in self.all_banks():
            merged = clean_string(''.join(b))
            if apply_cond(merged, **kwargs):
                if not merged in self.options:
                    self.options[merged] = 0
                self.options[merged] += b_s
        return self.options

class Container(Operator):
    def __init__(self, bank):
        super().__init__(bank)
        self.factor = 10 # O(n)
        
    def eval(self, **kwargs):
        super().eval() # for now just to avoid trying to fix any characters
        self.options = {}
        for b, b_s in self.all_banks():
            assert len(b) == 2 # for now
            for _ in range(2):
                for i in range(len(b[0]) + 1):
                    cur = b[0][:i] + b[1] + b[0][i:]
                    if apply_cond(cur, **kwargs):
                        if not cur in self.options:
                            self.options[cur] = 0
                        self.options[cur] += b_s
                b = [b[1], b[0]]
        return self.options

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
        self.factor = 100 # O(n^2)
        
    def eval(self, **kwargs):
        super().eval() # hidden removes letters
        self.options = {}
        for b, b_s in self.all_banks():
            merged = clean_string(''.join(b))
            for i in range(len(merged)):
                for j in range(i + 1, len(merged)): # need to enforce using all words
                    cur = merged[i:j]
                    if apply_cond(cur, **kwargs):
                        if not cur in self.options:
                            self.options[cur] = 0
                        self.options[cur] += b_s
        return self.options
    
class Initialism(Operator):
    def __init__(self, bank):
        super().__init__(bank)
        self.factor = 1 # O(1)
        
    def eval(self, **kwargs):
        super().eval() # initialism removes letters
        self.options = {}
        for b, b_s in self.all_banks():
            merged = clean_string(''.join([b_[0] for b_ in b]))
            if apply_cond(merged, **kwargs):
                if not merged in self.options:
                    self.options[merged] = 0
                self.options[merged] += b_s
        return self.options
    
class Terminalism(Operator):
    def __init__(self, bank):
        super().__init__(bank)
        self.factor = 1 # O(1)
        
    def eval(self, **kwargs):
        super().eval() # terminalism removes letters
        self.options = {}
        for b, b_s in self.all_banks():
            merged = clean_string(''.join([b_[-1] for b_ in b]))
            if apply_cond(merged, **kwargs):
                if not merged in self.options:
                    self.options[merged] = 0
                self.options[merged] += b_s
        return self.options
    
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
        self.factor = 1 # O(1)
        
    def eval(self, **kwargs):
        super().eval(**kwargs)
        self.options = {}
        for b, b_s in self.all_banks():
            merged = clean_string(''.join(b))[::-1]
            if apply_cond(merged, **kwargs):
                if not merged in self.options:
                    self.options[merged] = 0
                self.options[merged] += b_s
        return self.options

class Substitution(Operator):
    def __init__(self, bank):
        super().__init__(bank)
        self.factor = 500 # >O(n^2) <O(n!)
        
    def eval(self, **kwargs):
        super().eval() # substitution removes letters
        self.options = {}
        for b, b_s in self.all_banks():
            merged = ' '.join(b)
            syns = synonyms(merged, **kwargs)
            for syn, syn_s in syns.items():
                # syn = clean_string(syn) # should already be clean
                if apply_cond(syn, **kwargs):
                    if not syn in self.options:
                        self.options[syn] = 0
                    self.options[syn] += b_s * syn_s
        self.options = sorted(self.options.items(), key=lambda x: -x[1])
        self.options = self.options[:2000] # for now
        self.options = {o[0]: o[1] for o in self.options}
        return self.options

class Definition:
    def __init__(self, defn):
        self.defn = defn
    
    def eval(self, **kwargs):
        return synonyms(self.defn, **kwargs)

def normalize_dict(d):
    s = sum(d.values())
    for k in d:
        d[k] /= s
    return d

def eval(part1, part2, ans):
    full = {}
    for c in "abcdefghijklmnopqrstuvwxyz":
        full[c] = 1
    fixed = [full] * len(ans)
    fixed[2] = {ans[2]: 1}
    
    # tmp = {}
    # for c in ans:
    #     tmp[c] = 1
    # fixed = [tmp] * len(ans)
    # # fixed = [{ans[i]: 1} for i in range(len(ans))]
    dict1 = part1.eval(exact_len=len(ans), fixed=fixed, in_dict=True)
    dict2 = part2.eval(exact_len=len(ans), fixed=fixed, in_dict=True)
    dict1 = normalize_dict(dict1)
    dict2 = normalize_dict(dict2)
    term1 = dict1.get(ans, 0)
    term2 = dict2.get(ans, 0)
    print(term1, "*", term2, "=", term1 * term2)
    return term1 * term2

print("eval(Definition(\"Country\"), Concatenation([Substitution(\"left\"), Alternation(\"judgeable\")]), \"portugal\") = ",
    eval(Definition("Country"), Concatenation([Substitution("left"), Alternation("judgeable")]), "portugal"))

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
print("eval(Reversal([Hidden(\"god-awful\")]), Definition(\"trouble\"), \"ado\") = ",
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