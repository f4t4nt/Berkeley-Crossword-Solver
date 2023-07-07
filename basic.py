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

def solve(crossword):
    solver = BPSolver(crossword, max_candidates=500000)
    solution = solver.solve(num_iters=10, iterative_improvement_steps=5)
    print("*** Solver Output ***")
    print_grid(solution)
    print("*** Gold Solution ***")
    print_grid(crossword.letter_grid)
    solver.evaluate(solution)

puzzle_file = "solver/Mar2321.puz"

with open(puzzle_file, "r") as f:
    print('Running solver on file:', f)
    puzzle = puz_to_json(puzzle_file)
    crossword = Crossword(puzzle)
    solve(crossword)