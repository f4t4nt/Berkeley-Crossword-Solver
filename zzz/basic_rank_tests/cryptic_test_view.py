import pickle
import numpy as np
import matplotlib.pyplot as plt

use_clues = False

if use_clues:
    with open('clues.pkl', 'rb') as f:
        pred_ranks = pickle.load(f)
else:
    with open('defns.pkl', 'rb') as f:
        pred_ranks = pickle.load(f)

threshold = 1000
pred_ranks = [r if r < threshold else threshold + 1 for r in pred_ranks]
pred_map = {}
for r in pred_ranks:
    if r not in pred_map:
        pred_map[r] = 0
    pred_map[r] += 1
pred_map = {k: v for k, v in sorted(pred_map.items(), key=lambda item: item[0])}

plt.bar(pred_map.keys(), pred_map.values(), color='blue', width=1)
plt.plot([np.median(pred_ranks), np.median(pred_ranks)], [0, max(pred_map.values())], color='red', linewidth=3)
plt.plot([np.mean(pred_ranks), np.mean(pred_ranks)], [0, max(pred_map.values())], color='red', linestyle='dashed', linewidth=3)
plt.yscale('log')
plt.xlabel('Rank of correct answer')
plt.ylabel('Number of predictions')
plt.title('Rank of correct answer given {0}'.format('clue' if use_clues else 'definition'))
plt.savefig('{0}_rank.png'.format('clues' if use_clues else 'defns'))

print('Average rank:', np.mean(pred_ranks))
print('Median rank:', np.median(pred_ranks))
print('Percent of answers in top 1:', np.mean([r <= 1 for r in pred_ranks]))
print('Percent of answers in top 10:', np.mean([r <= 10 for r in pred_ranks]))
print('Percent of answers in top 100:', np.mean([r <= 100 for r in pred_ranks]))
print('Percent of answers not found (or found at rank >{0}):'.format(threshold), np.mean([r == threshold + 1 for r in pred_ranks]))