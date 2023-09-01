from english_words import get_english_words_set

words = get_english_words_set(['gcide', 'web2'], lower=True)

with open('words.txt', 'w') as f:
    for word in words:
        f.write(word + '\n')