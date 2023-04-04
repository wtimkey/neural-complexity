'''
Use this script to preprocess sentences from CCGBank
This script produces a list of sentences from each partition of the treebank.
Each element of the list is a list of tuples, where the first element is the sentence token, and the second is the word's supertag
This list of sentences is picked and output to a .pt file for each partition.
I use the human readable html parses simply because they proved easier to extract
'''


import nltk
from bs4 import BeautifulSoup, Tag
import glob
import pickle
from tqdm import tqdm
import re

train_file_list = glob.glob('/Users/wpt2011/Desktop/Research/neural-complexity/ccg_tag_preprocessing/ccgbank/data/HTML/train/*/wsj_*.html')
valid_file_list = glob.glob('/Users/wpt2011/Desktop/Research/neural-complexity/ccg_tag_preprocessing/ccgbank/data/HTML/valid/*/wsj_*.html')
test_file_list = glob.glob('/Users/wpt2011/Desktop/Research/neural-complexity/ccg_tag_preprocessing/ccgbank/data/HTML/test/*/wsj_*.html')

def gen_vocabulary(word_counts, tag_counts):
    word_vocab = set()
    tag_vocab = set()
    for word, count in word_counts.items():
        if count > 3:
            word_vocab.add(word)
    for tag, count in tag_counts.items():
        if count > 25:
            tag_vocab.add(tag)
    return word_vocab, tag_vocab

def add_unks(tagged_sents, word_vocab, tag_vocab):
    unked_tagged_sents = []
    for i in range(len(tagged_sents)):
        curr_sent_tagged = []
        for j in range(len(tagged_sents[i])):
            word = tagged_sents[i][j][0]
            tag = tagged_sents[i][j][1]
            if word not in word_vocab:
                word = '<unk>'
            if tag not in tag_vocab:
                tag = '<unk>'
            curr_sent_tagged.append((word, tag))
        unked_tagged_sents.append(curr_sent_tagged)
    return unked_tagged_sents

def strip_tag_features(tag):
    return re.sub(r'\[[^)]*\]', '', tag)

def replace_nums(word_type):
    word_type_m = word_type.replace('.', '')
    word_type_m = word_type.replace(',', '')
    word_type_m = word_type.replace('-', '')
    if word_type_m.isnumeric():
        return '<num>'
    else:
        return word_type

train_word_counts = dict()
train_tag_counts = dict()
for i, file_list in enumerate([train_file_list, valid_file_list, test_file_list]):
    supertagged_sents = []
    for fname in tqdm(file_list):
        with open(fname, 'r') as f:
            html_data = f.read()
        soup = BeautifulSoup(html_data)
        trees_text = [x.text for x in soup.find_all('pre')]
        for tree_i in trees_text:
            tree = nltk.Tree.fromstring(tree_i, brackets='{}')
            pos_tags = tree.pos()
            pos_tags = [(replace_nums(word[0]), strip_tag_features(word[1])) for word in pos_tags]
            supertagged_sents.append(pos_tags)
            if i == 0:
                for j in range(len(pos_tags)):
                    word = pos_tags[j][0]
                    tag = pos_tags[j][1]
                    if word in train_word_counts.keys():
                        train_word_counts[word] += 1
                    else:
                        train_word_counts[word] = 1
                    if tag in train_tag_counts.keys():
                        train_tag_counts[tag] += 1
                    else:
                        train_tag_counts[tag] = 1

    if i == 0:
        word_vocab, tag_vocab = gen_vocabulary(train_word_counts, train_tag_counts)
        out_name = 'ccgbank_supertagged_train.pt'
    elif i == 1:
        out_name = 'ccgbank_supertagged_valid.pt'
    elif i == 2:
        out_name = 'ccgbank_supertagged_test.pt'
    supertagged_sents = add_unks(supertagged_sents, word_vocab, tag_vocab)
    pickle.dump(supertagged_sents, open(out_name, 'wb'))

