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

train_file_list = glob.glob('/Users/wpt2011/Desktop/Research/neural-complexity/ccg_tag_preprocessing/ccgbank/data/HTML/train/*/wsj_*.html')
valid_file_list = glob.glob('/Users/wpt2011/Desktop/Research/neural-complexity/ccg_tag_preprocessing/ccgbank/data/HTML/valid/*/wsj_*.html')
test_file_list = glob.glob('/Users/wpt2011/Desktop/Research/neural-complexity/ccg_tag_preprocessing/ccgbank/data/HTML/test/*/wsj_*.html')

for i, file_list in enumerate([train_file_list, valid_file_list, test_file_list]):
    supertagged_sents = []
    for fname in tqdm(file_list):
        with open(fname, 'r') as f:
            html_data = f.read()
        soup = BeautifulSoup(html_data)
        trees_text = [x.text for x in soup.find_all('pre')]
        for tree_i in trees_text:
            tree = nltk.Tree.fromstring(trees_text[0], brackets='{}')
            pos_tags = tree.pos()
            supertagged_sents.append(pos_tags)
    if i == 0:
        out_name = 'ccgbank_supertagged_train.pt'
    elif i == 1:
        out_name = 'ccgbank_supertagged_valid.pt'
    elif i == 2:
        out_name = 'ccgbank_supertagged_test.pt'
    pickle.dump(supertagged_sents, open(out_name, 'wb'))

