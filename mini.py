# read in all words from input() splitting by ' '
words = input().split(' ')

# convert each word to set and then back to string and print it

for word in words:
    print(''.join(set(word)), end=' ')