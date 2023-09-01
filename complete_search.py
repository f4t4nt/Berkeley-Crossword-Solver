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
import spacy

random.seed()

nlp = spacy.load("en_core_web_lg")
data = load.load_data()
words = load.load_words()
dpr = models.setup_closedbook(0)

max_answers = 1000000

def get_possible_defns(clue):
    words = clue.split()
    rv = []
    for i in range(1, len(words)):
        left, right = ' '.join(words[:i]), ' '.join(words[i:])
        rv.append(left)
        rv.append(right)
    return rv

def try_alternating(nondefn, ans):
    cur = ''
    pos = []
    for d in range(2):
        i = d
        skip = False
        while i < len(nondefn):
            if skip:
                while i < len(nondefn) and not nondefn[i].isalpha():
                    i += 1
                i += 1
                if i >= len(nondefn):
                    break
                skip = False
                continue
            else:
                while i < len(nondefn) and not nondefn[i].isalpha():
                    i += 1
                if i >= len(nondefn):
                    break
                pos.append(i)
            cur += nondefn[i]
            if nondefn[i].islower():
                nondefn = nondefn[:i] + nondefn[i].upper() + nondefn[i+1:]
            if len(cur) > len(ans):
                nondefn = nondefn[:pos[0]] + nondefn[pos[0]].lower() + nondefn[pos[0]+1:]
                cur = cur[1:]
                pos = pos[1:]
            if cur == ans:
                words = nondefn.split()
                return True, [words[k] for k in range(len(words)) if words[k] != words[k].lower()]
            i += 1
            skip = True
    return False, None

def try_anagram(nondefn, ans):
    words = nondefn.split()
    ans_decomp = list(ans)
    ans_decomp.sort()
    for i in range(len(words)):
        word_decomp = list(words[i])
        word_decomp.sort()
        if word_decomp == ans_decomp:
            return True, [words[i]]
    return False, None

def try_hidden(nondefn, ans):
    cur = ''
    pos = []
    i = 0
    while i < len(nondefn):
        while i < len(nondefn) and not nondefn[i].isalpha():
            i += 1
        if i >= len(nondefn):
            break
        pos.append(i)
        cur += nondefn[i]
        if nondefn[i].islower():
            nondefn = nondefn[:i] + nondefn[i].upper() + nondefn[i+1:]
        if len(cur) > len(ans):
            nondefn = nondefn[:pos[0]] + nondefn[pos[0]].lower() + nondefn[pos[0]+1:]
            cur = cur[1:]
            pos = pos[1:]
        if cur == ans:
            words = nondefn.split()
            return True, [words[k] for k in range(len(words)) if words[k] != words[k].lower()]
        i += 1
    return False, None

def try_initials(nondefn, ans):
    words = nondefn.split()
    for i in range(len(words)):
        used = []
        jw, ja = i, 0
        while jw < len(words) and ja < len(ans):
            if words[jw][0] == ans[ja]:
                used.append(words[jw][0].upper() + words[jw][1:])
                jw += 1
                ja += 1
            else:
                break
        if ja == len(ans):
            return True, used
    return False, None

def try_letter_bank(nondefn, ans):
    ans_decomp = list(set(ans))
    ans_decomp.sort()
    words = nondefn.split()
    for i in range(len(words)):
        word_decomp = list(set(words[i]))
        word_decomp.sort()
        if ans_decomp == word_decomp and len(words[i]) < len(ans):
            return True, [words[i]]
        for j in range(i + 1, len(words)):
            word_decomp = list(set(words[i] + words[j]))
            word_decomp.sort()
            if ans_decomp == word_decomp:
                return True, [words[i], words[j]]
            for k in range(j + 1, len(words)):
                word_decomp = list(set(words[i] + words[j] + words[k]))
                word_decomp.sort()
                if ans_decomp == word_decomp:
                    return True, [words[i], words[j], words[k]]
    return False, None

def try_reverse(nondefn, ans):
    words = nondefn.split()
    for i in range(len(words)):
        if words[i][::-1] == ans:
            return True, [words[i]]
    return False, None

def try_spoonerism(nondefn, ans):
    words = nondefn.split()
    for idx0 in range(len(words)):
        for idx1 in range(idx0 + 1, len(words)): # isis comes up a lot for range(idx0, len(words))
            if len(words[idx0]) == 1 or len(words[idx1]) == 1:
                continue
            if words[idx1][0] + words[idx0][1:] + words[idx0][0] + words[idx1][1:] == ans:
                return True, [words[idx0], words[idx1]]
    return False, None

def try_terminals(nondefn, ans):
    words = nondefn.split()
    for i in range(len(words)):
        used = []
        jw, ja = i, 0
        while jw < len(words) and ja < len(ans):
            if words[jw][-1] == ans[ja]:
                used.append(words[jw][:-1] + words[jw][-1].upper())
                jw += 1
                ja += 1
            else:
                break
        if ja == len(ans):
            return True, used
    return False, None

random.seed()
random.shuffle(data)

answer_set = set()
for datapoint in data:
    clue, nondefn, defn, ans, sz = load.unwrap_data(datapoint)
    if not clue or not defn or not ans:
        continue
    ans = ans.lower().replace(' ', '')
    answer_set.add(ans)
print('TOTAL SIZE OF ANSWER SET', len(answer_set))

for datapoint in data:
    clue, nondefn, defn, ans, sz = load.unwrap_data(datapoint)
    if not clue or not defn or not ans:
        continue
    ans = ans.lower().replace(' ', '')
    print('TRYING CLUE', clue, 'DEFN', defn, 'ANS', ans)
    # success, used = try_hidden(nondefn, ans)
    # if success:
    #     print('SUCCESS HIDDEN', used)
    # else:
    #     print('FAILURE HIDDEN')
    # print('=' * 80)
    # continue
    possible_defns = get_possible_defns(clue)
    possible_ans, scores = models.answer_clues(dpr, possible_defns, max_answers, output_strings=True)
    for i in range(len(possible_ans)):
        possible_ans[i] = [ansn.lower().replace(' ', '') for ansn in possible_ans[i] if len(ansn) - ansn.count(' ') == len(ans)]
        possible_ans[i] = [ansn for ansn in possible_ans[i] if ansn in answer_set]
    ok = False # all possible_ans[i] have same words now, any way to use this?
    for i in range(len(possible_ans)):
        if ans in possible_ans[i]:
            ok = True
            print('ANSWER IN POSSIBLE ANSWERS IN LIST', i, possible_defns[i])
    if not ok:
        print('ANSWER NOT IN POSSIBLE ANSWERS, SKIPPING')
        print('=' * 80)
        continue
    already_found = {
        'alternating': [],
        'anagram': [],
        'hidden': [],
        'initials': [],
        'letter_bank': [],
        'reverse': [],
        'spoonerism': [],
        'terminals': [],
    }
    for i in range(len(possible_ans)):
        print('TRYING POSSIBLE ANSWERS IN LIST', i, possible_defns[i])
        print('TOTAL SIZE OF LIST', len(possible_ans[i]))
        try_defn = possible_defns[i]
        try_nondefn = possible_defns[i ^ 1]
        if i % 2 == 0: # double definition is different for efficiency
            possible_ans_intersection = set(possible_ans[i]) & set(possible_ans[i ^ 1])
            if len(possible_ans_intersection) > 0:
                if len(possible_ans_intersection) > 10:
                    possible_ans_intersection = list(possible_ans_intersection)
                    random.shuffle(possible_ans_intersection)
                    print('SUCCESS DOUBLE DEFINITION', possible_ans_intersection[:10])
                else:
                    print('SUCCESS DOUBLE DEFINITION', possible_ans_intersection)
        for try_ans in possible_ans[i]:
            is_correct = try_ans == ans
            found = False
            mute = False
            # alternating
            success, used = try_alternating(try_nondefn, try_ans)
            if success:
                found = True
                if not used in already_found['alternating']:
                    print('SUCCESS ALTERNATING', used, 'ANS?', try_ans)
                    already_found['alternating'].append(used)
                else:
                    mute = True
            # anagram
            success, used = try_anagram(try_nondefn, try_ans)
            if success:
                found = True
                if not used in already_found['anagram']:
                    print('SUCCESS ANAGRAM', used, 'ANS?', try_ans)
                    already_found['anagram'].append(used)
                else:
                    mute = True
            # hidden
            success, used = try_hidden(try_nondefn, try_ans)
            if success:
                found = True
                if not used in already_found['hidden']:
                    print('SUCCESS HIDDEN', used, 'ANS?', try_ans)
                    already_found['hidden'].append(used)
                else:
                    mute = True
            # initials
            success, used = try_initials(try_nondefn, try_ans)
            if success:
                found = True
                if not used in already_found['initials']:
                    print('SUCCESS INITIALS', used, 'ANS?', try_ans)
                    already_found['initials'].append(used)
                else:
                    mute = True
            # letter bank
            success, used = try_letter_bank(try_nondefn, try_ans)
            if success:
                found = True
                if not used in already_found['letter_bank']:
                    print('SUCCESS LETTER BANK', used, 'ANS?', try_ans)
                    already_found['letter_bank'].append(used)
                else:
                    mute = True
            # reverse
            success, used = try_reverse(try_nondefn, try_ans)
            if success:
                found = True
                if not used in already_found['reverse']:
                    print('SUCCESS REVERSE', used, 'ANS?', try_ans)
                    already_found['reverse'].append(used)
                else:
                    mute = True
            # spoonerism
            success, used = try_spoonerism(try_nondefn, try_ans)
            if success:
                found = True
                if not used in already_found['spoonerism']:
                    print('SUCCESS SPOONERISM', used, 'ANS?', try_ans)
                    already_found['spoonerism'].append(used)
                else:
                    mute = True
            # terminals
            success, used = try_terminals(try_nondefn, try_ans)
            if success:
                found = True
                if not used in already_found['terminals']:
                    print('SUCCESS TERMINALS', used, 'ANS?', try_ans)
                    already_found['terminals'].append(used)
                else:
                    mute = True
            if is_correct and not mute:
                if found:
                    print('---> SUCCESS CORRECT ANSWER FOUND <---')
                else:
                    print('---> FAILURE NO METHOD FOUND <---')
    print('=' * 80)