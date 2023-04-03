""" Module for handling data files """

import gzip
import os
import torch
import re
import pickle

re_sentend = re.compile(r'(?<!\b[A-Z]\.)(?<!\b[Mm]rs\.)(?<!\b[MmDdSsJj]r\.)(?<=[\.\?\!])[ \n\t](?!["\'])|(?<!\b[A-Z]\.)(?<!\b[Mm]rs\.)(?<!\b[MmDdSsJj]r\.)(?<=[\.\?\!] ["\'])[ \n\t]+')

def sent_tokenize(instr):
    return(re.split(re_sentend,instr))

def isfloat(instr):
    """ Reports whether a string is floatable """
    try:
        _ = float(instr)
        return(True)
    except:
        return(False)

class Dictionary(object):
    """ Maps between observations and indices """
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        """ Adds a new obs to the dictionary if needed """
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

class SentenceCorpus(object):
    """ Loads train/dev/test corpora and dictionary """
    def __init__(self, path, vocab_file, test_flag=False, interact_flag=False,
                 checkpoint_flag=False, predefined_vocab_flag=False, lower_flag=False,
                 collapse_nums_flag=False,multisentence_test_flag=False,generate_flag=False,
                 trainfname='train.txt',
                 validfname='valid.txt',
                 testfname='test.txt', aux_objective_flag=False, aux_vocab_file='aux_labels.txt'):

        self.lower = lower_flag
        self.collapse_nums = collapse_nums_flag
        self.aux_objective = aux_objective_flag
        if not (test_flag or interact_flag or checkpoint_flag or predefined_vocab_flag or generate_flag):
            # training mode
            self.dictionary = Dictionary()
            if(self.aux_objective):
                self.aux_dictionary = Dictionary()
                self.train, self.train_aux_labels = self.tokenize_auxillary_task_file(os.path.join(path, trainfname))
                self.valid, self.valid_aux_labels = self.tokenize_aux_task_file_with_unks(os.path.join(path, validfname))
            else:
                self.train = self.tokenize(os.path.join(path, trainfname))
                self.valid = self.tokenize_with_unks(os.path.join(path, validfname))
            try:
                # don't require a test set at train time,
                # but if there is one, get a sense of whether unks will be required
                self.test = self.tokenize_with_unks(os.path.join(path, testfname))
            except:
                pass
            self.save_dict(vocab_file, self.dictionary)
            if(self.aux_objective):
                self.save_dict(aux_vocab_file, self.aux_dictionary)
        else:
            # load pretrained model
            self.dictionary = self.load_dict(vocab_file)
            if(self.aux_objective):
                self.aux_dictionary = self.load_dict(aux_vocab_file)

            if test_flag:
                # test mode
                if multisentence_test_flag:
                    self.test = self.tokenize_with_unks(os.path.join(path, testfname))
                else:
                    self.test = self.sent_tokenize_with_unks(os.path.join(path, testfname))
            elif checkpoint_flag or predefined_vocab_flag:
                # load from a checkpoint
                if(self.aux_objective):
                    self.train, self.train_aux_labels = self.tokenize_auxillary_task_file(os.path.join(path, trainfname))
                    self.valid, self.valid_aux_labels = self.tokenize_aux_task_file_with_unks(os.path.join(path, validfname))
                else:
                    self.train = self.tokenize_with_unks(os.path.join(path, trainfname))
                    self.valid = self.tokenize_with_unks(os.path.join(path, validfname))


    def __len__(self):
        return len(self.dictionary)

    def save_dict(self, path, curr_dictionary):
        """ Saves dictionary to disk """
        if path[-3:] == 'bin':
            # This check actually seems to be faster than passing in a binary flag
            # Assume dict is binarized
            import dill
            with open(path, 'wb') as file_handle:
                torch.save(curr_dictionary, file_handle, pickle_module=dill)
        else:
            # Assume dict is plaintext
            with open(path, 'w') as file_handle:
                for word in curr_dictionary.idx2word:
                    file_handle.write(word+'\n')

    def load_dict(self, path):
        """ Loads dictionary from disk """
        assert os.path.exists(path), "Bad path: %s" % path
        if path[-3:] == 'bin':
            # This check actually seems to be faster than passing in a binary flag
            # Assume dict is binarized
            import dill
            with open(path, 'rb') as file_handle:
                fdata = torch.load(file_handle, pickle_module=dill)
                if isinstance(fdata, tuple):
                    # Compatibility with old pytorch LM saving
                    dict_to_return = fdata[3]
                dict_to_return = fdata
        else:
            # Assume dict is plaintext
            dict_to_return = Dictionary()
            with open(path, 'r') as file_handle:
                for line in file_handle:
                    dict_to_return.add_word(line.strip())
        return dict_to_return

    def tokenize(self, path):
        """ Tokenizes a text file. """
        assert os.path.exists(path), "Bad path: %s" % path
        # Add words to the dictionary
        if path[-2:] == 'gz':
            with gzip.open(path, 'rb') as file_handle:
                tokens = 0
                first_flag = True
                for fchunk in file_handle.readlines():
                    for line in sent_tokenize(fchunk.decode("utf-8")):
                        if line.strip() == '':
                            # Ignore blank lines
                            continue
                        if first_flag:
                            words = ['<eos>'] + line.split() + ['<eos>']
                            first_flag = False
                        else:
                            words = line.split() + ['<eos>']
                        tokens += len(words)
                        if self.lower:
                            for word in words:
                                if isfloat(word) and self.collapse_nums:
                                    self.dictionary.add_word('<num>')
                                else:
                                    self.dictionary.add_word(word.lower())
                        else:
                            for word in words:
                                if isfloat(word) and self.collapse_nums:
                                    self.dictionary.add_word('<num>')
                                else:
                                    self.dictionary.add_word(word)

            # Tokenize file content
            with gzip.open(path, 'rb') as file_handle:
                ids = torch.IntTensor(tokens)
                token = 0
                first_flag = True
                for fchunk in file_handle.readlines():
                    for line in sent_tokenize(fchunk.decode("utf-8")):
                        if line.strip() == '':
                            # Ignore blank lines
                            continue
                        if first_flag:
                            words = ['<eos>'] + line.split() + ['<eos>']
                            first_flag = False
                        else:
                            words = line.split() + ['<eos>']
                        if self.lower:
                            for word in words:
                                if isfloat(word) and '<num>' in self.dictionary.word2idx:
                                    ids[token] = self.dictionary.add_word("<num>")
                                else:
                                    ids[token] = self.dictionary.add_word(word.lower())
                                token += 1
                        else:
                            for word in words:
                                if isfloat(word) and '<num>' in self.dictionary.word2idx:
                                    ids[token] = self.dictionary.add_word("<num>")
                                else:
                                    ids[token] = self.dictionary.add_word(word)
                                token += 1
        else:
            with open(path, 'r') as file_handle:
                tokens = 0
                first_flag = True
                for fchunk in file_handle:
                    for line in sent_tokenize(fchunk):
                        if line.strip() == '':
                            # Ignore blank lines
                            continue
                        if first_flag:
                            words = ['<eos>'] + line.split() + ['<eos>']
                            first_flag = False
                        else:
                            words = line.split() + ['<eos>']
                        tokens += len(words)
                        if self.lower:
                            for word in words:
                                if isfloat(word) and self.collapse_nums:
                                    self.dictionary.add_word('<num>')
                                else:
                                    self.dictionary.add_word(word.lower())
                        else:
                            for word in words:
                                if isfloat(word) and self.collapse_nums:
                                    self.dictionary.add_word('<num>')
                                else:
                                    self.dictionary.add_word(word)

            # Tokenize file content
            with open(path, 'r') as file_handle:
                ids = torch.IntTensor(tokens)
                token = 0
                first_flag = True
                for fchunk in file_handle:
                    for line in sent_tokenize(fchunk):
                        if line.strip() == '':
                            # Ignore blank lines
                            continue
                        if first_flag:
                            words = ['<eos>'] + line.split() + ['<eos>']
                            first_flag = False
                        else:
                            words = line.split() + ['<eos>']
                        if self.lower:
                            for word in words:
                                if isfloat(word) and '<num>' in self.dictionary.word2idx:
                                    ids[token] = self.dictionary.add_word("<num>")
                                else:
                                    ids[token] = self.dictionary.add_word(word.lower())
                                token += 1
                        else:
                            for word in words:
                                if isfloat(word) and '<num>' in self.dictionary.word2idx:
                                    ids[token] = self.dictionary.add_word("<num>")
                                else:
                                    ids[token] = self.dictionary.add_word(word)
                                token += 1
        return ids
    
    def tokenize_auxillary_task_file(self, path):
        """Tokenizes a file containing lists of sentences with tokens and their auxillary labels
            An example of such a file is the output of the preprocess_ccg.py script, which pairs
            tokens with their ccg supertags from CCGBank
        """
        assert os.path.exists(path), "Bad path: %s" % path
        with open(path, 'rb') as f:
            data_file = pickle.load(f)
        # Add words to the dictionary
        tokens = 0
        first_flag = True
        for sent in data_file:
            if first_flag:
                words = ['<eos>'] + [word[0] for word in sent] + ['<eos>']
                labels = ['<eos>'] + [word[1] for word in sent] + ['<eos>']
                first_flag = False
            else:
                words = [word[0] for word in sent] + ['<eos>']
                labels = [word[1] for word in sent] + ['<eos>']
            tokens += len(words)
            if self.lower:
                for word in words:
                    if isfloat(word) and self.collapse_nums:
                        self.dictionary.add_word('<num>')
                    else:
                        self.dictionary.add_word(word.lower())
            else:
                for word in words:
                    if isfloat(word) and self.collapse_nums:
                        self.dictionary.add_word('<num>')
                    else:
                        self.dictionary.add_word(word)
            for label in labels:
                self.aux_dictionary.add_word(label)
        
        #tokenize sents and labels
        ids = torch.IntTensor(tokens)
        aux_ids = torch.IntTensor(tokens)
        token = 0
        first_flag = True

        for sent in data_file:
            if first_flag:
                words = ['<eos>'] + [word[0] for word in sent] + ['<eos>']
                labels = ['<eos>'] + [word[1] for word in sent] + ['<eos>']
                first_flag = False
            else:
                words = [word[0] for word in sent] + ['<eos>']
                labels = [word[1] for word in sent] + ['<eos>']
            if self.lower:
                for i in range(len(words)):
                    word = words[i]
                    label = labels[i]
                    if isfloat(word) and '<num>' in self.dictionary.word2idx:
                        ids[token] = self.dictionary.add_word("<num>")
                    else:
                        ids[token] = self.dictionary.add_word(word.lower())
                    aux_ids[token] = self.aux_dictionary.add_word(label)
                    token += 1
            else:
                for i in range(len(words)):
                    word = words[i]
                    label = labels[i]
                    if isfloat(word) and '<num>' in self.dictionary.word2idx:
                        ids[token] = self.dictionary.add_word("<num>")
                    else:
                        ids[token] = self.dictionary.add_word(word)
                    aux_ids[token] = self.aux_dictionary.add_word(label)
                    token += 1

        return ids, aux_ids
    
    def tokenize_aux_task_file_with_unks(self, path):
        """Tokenizes a file containing lists of sentences with tokens and their auxillary labels
            An example of such a file is the output of the preprocess_ccg.py script, which pairs
            tokens with their ccg supertags from CCGBank
        """
        #determine length of corpus
        #todo: this can be optimized if it proves to be a bottleneck
        assert os.path.exists(path), "Bad path: %s" % path
        with open(path, 'rb') as f:
            data_file = pickle.load(f)
        tokens = 0
        first_flag = True
        for sent in data_file:
            if first_flag:
                words = ['<eos>'] + [word[0] for word in sent] + ['<eos>']
                first_flag = False
            else:
                words = [word[0] for word in sent] + ['<eos>']
            tokens += len(words)


        ids = torch.IntTensor(tokens)
        aux_ids = torch.IntTensor(tokens)
        token = 0
        first_flag = True

        for sent in data_file:
            if first_flag:
                words = ['<eos>'] + [word[0] for word in sent] + ['<eos>']
                labels = ['<eos>'] + [word[1] for word in sent] + ['<eos>']
                first_flag = False
            else:
                words = [word[0] for word in sent] + ['<eos>']
                labels = [word[1] for word in sent] + ['<eos>']
            if self.lower:
                for i in range(len(words)):
                    word = words[i]
                    label = labels[i]
                    # Convert OOV to <unk>
                    if word.lower() not in self.dictionary.word2idx:
                        ids[token] = self.dictionary.add_word("<unk>")
                    elif isfloat(word) and '<num>' in self.dictionary.word2idx:
                        ids[token] = self.dictionary.add_word("<num>")
                    else:
                        ids[token] = self.dictionary.word2idx[word.lower()]
                    if label not in self.aux_dictionary.word2idx:
                        aux_ids[token] = self.aux_dictionary.add_word("<unk>")
                    else:
                        aux_ids[token] = self.aux_dictionary.add_word(label)
                    token += 1
            else:
                for i in range(len(words)):
                    word = words[i]
                    label = labels[i]
                    # Convert OOV to <unk>
                    if word not in self.dictionary.word2idx:
                        ids[token] = self.dictionary.add_word("<unk>")
                    elif isfloat(word) and '<num>' in self.dictionary.word2idx:
                        ids[token] = self.dictionary.add_word("<num>")
                    else:
                        ids[token] = self.dictionary.word2idx[word]

                    if label not in self.aux_dictionary.word2idx:
                        aux_ids[token] = self.aux_dictionary.add_word("<unk>")
                    else:
                        aux_ids[token] = self.aux_dictionary.add_word(label)
                    token += 1

        return ids, aux_ids

    def tokenize_with_unks(self, path):
        """ Tokenizes a text file, adding unks if needed. """
        assert os.path.exists(path), "Bad path: %s" % path
        if path[-2:] == 'gz':
            # Determine the length of the corpus
            with gzip.open(path, 'rb') as file_handle:
                tokens = 0
                first_flag = True
                for fchunk in file_handle.readlines():
                    for line in sent_tokenize(fchunk.decode("utf-8")):
                        if line.strip() == '':
                            # Ignore blank lines
                            continue
                        if first_flag:
                            words = ['<eos>'] + line.split() + ['<eos>']
                            first_flag = False
                        else:
                            words = line.split() + ['<eos>']
                        tokens += len(words)

            # Tokenize file content
            with gzip.open(path, 'rb') as file_handle:
                ids = torch.IntTensor(tokens)
                token = 0
                first_flag = True
                for fchunk in file_handle.readlines():
                    for line in sent_tokenize(fchunk.decode("utf-8")):
                        if line.strip() == '':
                            # Ignore blank lines
                            continue
                        if first_flag:
                            words = ['<eos>'] + line.split() + ['<eos>']
                            first_flag = False
                        else:
                            words = line.split() + ['<eos>']
                        if self.lower:
                            for word in words:
                                # Convert OOV to <unk>
                                if word.lower() not in self.dictionary.word2idx:
                                    ids[token] = self.dictionary.add_word("<unk>")
                                elif isfloat(word) and '<num>' in self.dictionary.word2idx:
                                    ids[token] = self.dictionary.add_word("<num>")
                                else:
                                    ids[token] = self.dictionary.word2idx[word.lower()]
                                token += 1
                        else:
                            for word in words:
                                # Convert OOV to <unk>
                                if word not in self.dictionary.word2idx:
                                    ids[token] = self.dictionary.add_word("<unk>")
                                elif isfloat(word) and '<num>' in self.dictionary.word2idx:
                                    ids[token] = self.dictionary.add_word("<num>")
                                else:
                                    ids[token] = self.dictionary.word2idx[word]
                                token += 1
        else:
            # Determine the length of the corpus
            with open(path, 'r') as file_handle:
                tokens = 0
                first_flag = True
                for fchunk in file_handle:
                    for line in sent_tokenize(fchunk):
                        if line.strip() == '':
                            # Ignore blank lines
                            continue
                        if first_flag:
                            words = ['<eos>'] + line.split() + ['<eos>']
                            first_flag = False
                        else:
                            words = line.split() + ['<eos>']
                        tokens += len(words)

            # Tokenize file content
            with open(path, 'r') as file_handle:
                ids = torch.IntTensor(tokens)
                token = 0
                first_flag = True
                for fchunk in file_handle:
                    for line in sent_tokenize(fchunk):
                        if line.strip() == '':
                            # Ignore blank lines
                            continue
                        if first_flag:
                            words = ['<eos>'] + line.split() + ['<eos>']
                            first_flag = False
                        else:
                            words = line.split() + ['<eos>']
                        if self.lower:
                            for word in words:
                                # Convert OOV to <unk>
                                if word.lower() not in self.dictionary.word2idx:
                                    ids[token] = self.dictionary.add_word("<unk>")
                                elif isfloat(word) and '<num>' in self.dictionary.word2idx:
                                    ids[token] = self.dictionary.add_word("<num>")
                                else:
                                    ids[token] = self.dictionary.word2idx[word.lower()]
                                token += 1
                        else:
                            for word in words:
                                # Convert OOV to <unk>
                                if word not in self.dictionary.word2idx:
                                    ids[token] = self.dictionary.add_word("<unk>")
                                elif isfloat(word) and '<num>' in self.dictionary.word2idx:
                                    ids[token] = self.dictionary.add_word("<num>")
                                else:
                                    ids[token] = self.dictionary.word2idx[word]
                                token += 1
        return ids

    def sent_tokenize_with_unks(self, path):
        """ Tokenizes a text file into sentences, adding unks if needed. """
        assert os.path.exists(path), "Bad path: %s" % path
        all_ids = []
        sents = []
        if path[-2:] == 'gz':
            with gzip.open(path, 'rb') as file_handle:
                for fchunk in file_handle.readlines():
                    for line in sent_tokenize(fchunk.decode("utf-8")):
                        if line.strip() == '':
                            # Ignore blank lines
                            continue
                        sents.append(line.strip())
                        words = ['<eos>'] + line.split() + ['<eos>']
                        ids = self.convert_to_ids(words)
                        all_ids.append(ids)
        else:
            with open(path, 'r') as file_handle:
                for fchunk in file_handle:
                    for line in sent_tokenize(fchunk):
                        if line.strip() == '':
                            # Ignore blank lines
                            continue
                        sents.append(line.strip())
                        words = ['<eos>'] + line.split() + ['<eos>']
                        ids = self.convert_to_ids(words)
                        all_ids.append(ids)
        return (sents, all_ids)

    def online_tokenize_with_unks(self, line):
        """ Tokenizes an input sentence, adding unks if needed. """
        all_ids = []
        sents = [line.strip()]

        words = ['<eos>'] + line.strip().split() + ['<eos>']

        ids = self.convert_to_ids(words)
        all_ids.append(ids)
        return (sents, all_ids)


    def convert_to_ids(self, words, tokens=None):
        if tokens is None:
            tokens = len(words)

        # Tokenize file content
        ids = torch.IntTensor(tokens)
        token = 0
        if self.lower:
            for word in words:
                # Convert OOV to <unk>
                if word.lower() not in self.dictionary.word2idx:
                    ids[token] = self.dictionary.add_word("<unk>")
                elif isfloat(word) and '<num>' in self.dictionary.word2idx:
                    ids[token] = self.dictionary.add_word("<num>")
                else:
                    ids[token] = self.dictionary.word2idx[word.lower()]
                token += 1
        else:
            for word in words:
                # Convert OOV to <unk>
                if word not in self.dictionary.word2idx:
                    ids[token] = self.dictionary.add_word("<unk>")
                elif isfloat(word) and '<num>' in self.dictionary.word2idx:
                    ids[token] = self.dictionary.add_word("<num>")
                else:
                    ids[token] = self.dictionary.word2idx[word]
                token += 1
        return(ids)
