'''
Code for training and evaluating a neural language model.
LM can output incremental complexity measures and be made adaptive.
'''

from __future__ import print_function
import argparse
import time
import math
import numpy as np
import sys
import warnings
import torch
import torch.nn as nn
import data
import model

try:
    from progress.bar import Bar
    PROGRESS = True
except ModuleNotFoundError:
    PROGRESS = False

# suppress SourceChangeWarnings
warnings.filterwarnings("ignore")

sys.stderr.write('Libraries loaded\n')

# Parallelization notes:
#   Does not currently operate across multiple nodes
#   Single GPU is better for default: tied,emsize:200,nhid:200,nlayers:2,dropout:0.2
#
#   Multiple GPUs are better for tied,emsize:1500,nhid:1500,nlayers:2,dropout:0.65
#      4 GPUs train on wikitext-2 in 1/2 - 2/3 the time of 1 GPU

parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM Language Model')

# Model parameters
parser.add_argument('--model', type=str, default='CUEBASEDRNN',
                    choices=['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU','CUEBASEDRNN'],
                    help='type of recurrent net')
parser.add_argument('--emsize', type=int, default=128,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=128,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=.1,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--grad_accumulation_steps', type=int, default=1,
                    help='accumulates gradients over N sub-batches to avoid out of memory errors')
parser.add_argument('--bptt', type=int, default=45,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--init', type=float, default=None,
                    help='-1 to randomly Initialize. Otherwise, all parameter weights set to value')
parser.add_argument('--aux_objective', action='store_true',
                    help='use flag to train on an auxillary objective like CCG supertagging')

# Data parameters
parser.add_argument('--model_file', type=str, default='model.pt',
                    help='path to save the final model')
parser.add_argument('--adapted_model', type=str, default='adaptedmodel.pt',
                    help='new path to save the final adapted model')
parser.add_argument('--data_dir', type=str, default='./data/wikitext-2',
                    help='location of the corpus data')
parser.add_argument('--vocab_file', type=str, default='vocab.txt',
                    help='path to save the vocab file')
parser.add_argument('--embedding_file', type=str, default=None,
                    help='path to pre-trained embeddings')
parser.add_argument('--trainfname', type=str, default='train.txt',
                    help='name of the training file')
parser.add_argument('--validfname', type=str, default='valid.txt',
                    help='name of the validation file')
parser.add_argument('--testfname', type=str, default='test.txt',
                    help='name of the test file')
parser.add_argument('--collapse_nums_flag', action='store_true',
                    help='collapse number tokens into a unified <num> token')

# Runtime parameters
parser.add_argument('--test', action='store_true',
                    help='test a trained LM')
parser.add_argument('--load_checkpoint', action='store_true',
                    help='continue training a pre-trained LM')
parser.add_argument('--freeze_embedding', action='store_true',
                    help='do not train embedding weights')
parser.add_argument('--single', action='store_true',
                    help='use only a single GPU (even if more are available)')
parser.add_argument('--multisentence_test', action='store_true',
                    help='treat multiple sentences as a single stream at test time')

parser.add_argument('--adapt', action='store_true',
                    help='adapt model weights during evaluation')
parser.add_argument('--interact', action='store_true',
                    help='run a trained network interactively')

# For getting embeddings
parser.add_argument('--view_emb', action='store_true',
                    help='output the word embedding rather than the cell state')

parser.add_argument('--view_layer', type=int, default=-1,
                    help='which layer should output cell states')
parser.add_argument('--view_hidden', action='store_true',
                    help='output the hidden state rather than the cell state')
parser.add_argument('--verbose_view_layer', action='store_true',
                    help='output the input observation followed by the vector activations')

parser.add_argument('--words', action='store_true',
                    help='evaluate word-level complexities (instead of sentence-level loss)')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='report interval')

parser.add_argument('--lowercase', action='store_true',
                    help='force all input to be lowercase')
parser.add_argument('--nopp', action='store_true',
                    help='suppress evaluation perplexity output')
parser.add_argument('--nocheader', action='store_true',
                    help='suppress complexity header')
parser.add_argument('--csep', type=str, default=' ',
                    help='change the separator in the complexity output')

parser.add_argument('--guess', action='store_true',
                    help='display best guesses at each time step')
parser.add_argument('--guessn', type=int, default=1,
                    help='output top n guesses')
parser.add_argument('--guesssurps', action='store_true',
                    help='display guess surps along with guesses')
parser.add_argument('--guessprobs', action='store_true',
                    help='display guess probs along with guesses')
parser.add_argument('--complexn', type=int, default=0,
                    help='compute complexity only over top n guesses (0 = all guesses)')

# Misc parameters not in README
parser.add_argument('--softcliptopk', action="store_true",
                    help='soften non top-k options instead of removing them')

args = parser.parse_args()

if args.interact:
    # If in interactive mode, force complexity output
    args.words = True
    args.test = True
    # Don't try to process multiple sentences in parallel interactively
    args.single = True
    
if args.adapt:
    # If adapting, we must be in test mode
    args.test = True

if args.view_layer != -1:
    # There shouldn't be a cheader if we're looking at model internals
    args.nocheader = True
    
# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    print("CUDA device availible")
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        print("using CUDA")
        torch.cuda.manual_seed(args.seed)
        if torch.cuda.device_count() == 1:
            args.single = True
np.random.seed(args.seed)
device = torch.device("cuda" if args.cuda else "cpu")
torch.autograd.set_detect_anomaly(True)

###############################################################################
# Load data
###############################################################################

def batchify(data, bsz):
    ''' Starting from sequential data, batchify arranges the dataset into columns.
    For instance, with the alphabet as the sequence and batch size 4, we'd get
    a g m s
    b h n t
    c i o u
    d j p v
    e k q w
    f l r x
    These columns are treated as independent by the model, which means that the
    dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
    batch processing.
    '''
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    # Turning the data over to CUDA at this point may lead to more OOM errors
    return data.to(device)

def trunc_pad_and_mask(sequence, bptt, eos_token, compute_masks):
    #pad sequence
    if(len(sequence) < bptt):
        sequence = np.pad(sequence, (0, bptt-len(sequence)), 'constant', constant_values=(0, 0))
    #truncate sequence
    elif(len(sequence)> bptt):
        sequence = np.array(sequence[:bptt])
    #compute sentence-level attention masks (token zero and eos attend only to position -1, no other token attends to -1)
    #
    if(compute_masks):
        mask = np.full((len(sequence), len(sequence) + 1),-np.inf)
        mask[0,0] = 0 #first token can look at position -1
        for i in range(len(sequence)):
            for j in reversed(range(1, i+1)):
                if(sequence[j] != eos_token and sequence[j-1] != eos_token): #allow attention if i is not eos, and prev is not eos
                        mask[i,j] = 0
                else: #attend to dummy position -1
                    if(i==j):
                        mask[i,0] = 0
                    break
    else:
        mask = None
    return sequence, mask

def get_sent_list_lens_from_data_tensor(data, min_sent_len):
    data_np = data.numpy()
    idx = np.where(data_np!=0)[0]
    idx2sent = np.split(data_np[idx],np.where(np.diff(idx)!=1)[0]+1)
    idx2sent[0] = np.append(0, idx2sent[0])
    for i in range(len(idx2sent)):
        idx2sent[i] = np.append(idx2sent[i], 0)
    sent_lens = dict()
    for i in range(len(idx2sent)):
        sen_len = len(idx2sent[i])
        if sen_len not in sent_lens.keys():
            sent_lens[sen_len] = {i}
        else:
            sent_lens[sen_len].add(i)
    #remove sentences that are too short
    for i in range(min_sent_len):
        if i in sent_lens.keys():
            sent_lens[i] = []
    return idx2sent, sent_lens

def pack_sentences(idx2sent, sent_lens, bptt, bsz, min_sent_len, compute_masks):
    '''Returns a list of sequences corresponding to sent ids (idx2sent) that should be concatinated'''
    sequences = []
    masks = []
    #heuristic 1: match pairs of sentences whose lengths add up to n, +/- some epsilon
    for epsilon in [0,1,-1,2,-2,3,-3,4,-4,5,-5]:
        for i in range(bptt + epsilon):
            len_a = i
            len_b = (bptt + epsilon) - i             
            if len_a in sent_lens.keys() and len_b in sent_lens.keys():
                while(len(sent_lens[len_a]) > 0 and len(sent_lens[len_b]) > 0):
                    if(not(sent_lens[len_a] == sent_lens[len_b] and len(sent_lens[len_a]) <= 1)):
                        sequence = [sent_lens[len_a].pop()] + [sent_lens[len_b].pop()]
                        sequences.append(sequence)
                    else:
                        break
    #next add sequences that are too long and truncate them
    for length in sent_lens.keys():
        if(length > bptt - 5):
            while(len(sent_lens[length]) > 0):
                sequence = [sent_lens[length].pop()]
                sequences.append(sequence)

    #finally, heuristic 2: from largest to smallest, add sequences until can't anymore (allowing +5 overflow)
    for i in reversed(range((bptt - 5) + 1)): #40, 39, 38...
        if(i in sent_lens.keys()):
            while(len(sent_lens[i]) > 0): 
                curr_sequence = [sent_lens[i].pop()]
                for j in reversed(range(1,i+1)):
                    if(j in sent_lens.keys()):
                        while((len(curr_sequence) + j) < (bptt + 5) and len(sent_lens[j]) > 0):
                            curr_sequence += [sent_lens[j].pop()]
                sequences.append(curr_sequence)
    return sequences


def sent_level_batchify(data, bptt, bsz, min_sent_len=5, compute_masks=True, aux_data=None):
    #using (sloppy) heuristics such that every sequence is the start of a sentence, with minimal padding or cropping
    #first split data back into sentences
    masks = None
    aux_labels = None
    aux_objective = (aux_data is not None)
    idx2sent, sent_lens = get_sent_list_lens_from_data_tensor(data, min_sent_len)
    if(aux_objective):
        idx2_sent_aux, _ = get_sent_list_lens_from_data_tensor(aux_data, min_sent_len)
    sequence_idxs = pack_sentences(idx2sent, sent_lens, bptt, bsz, min_sent_len, compute_masks)
    sequences = []
    masks = []
    aux_sequences = []
    for seq_idxs_i in sequence_idxs:
        seq_id_to_tok = []
        for seq_idx in seq_idxs_i:
            seq_id_to_tok.extend(idx2sent[seq_idx])
        seq_id_to_tok = np.array(seq_id_to_tok) 
        sequence, mask = trunc_pad_and_mask(seq_id_to_tok, bptt, 0, compute_masks)
        sequences.append(sequence)
        masks.append(mask)

        if(aux_objective):
            seq_id_to_tok_aux = []
            for seq_idx in seq_idxs_i:
                seq_id_to_tok_aux.extend(idx2_sent_aux[seq_idx])
            seq_id_to_tok_aux = np.array(seq_id_to_tok_aux)
            aux_sequence, _ = trunc_pad_and_mask(seq_id_to_tok_aux, bptt, 0, False)
            aux_sequences.append(aux_sequence)


    nbatch = len(sequences) // bsz 
    even_seq = bsz * nbatch
    remainder = len(sequences) - even_seq
    for i in range(bsz-remainder):
        sequences.append(sequences[even_seq+(i%remainder)])
        if(compute_masks):
            masks.append(masks[even_seq+(i%remainder)])
        if(aux_objective):
            aux_sequences.append(aux_sequences[even_seq+(i%remainder)])        

    sents = np.array(sequences)
    if(aux_objective):
        aux_labels = np.array(aux_sequences)
    if(compute_masks):
        masks = np.array(masks, dtype=np.float32)

    rand_inds = np.arange(sents.shape[0])
    np.random.shuffle(rand_inds)
    sents = torch.from_numpy(sents[rand_inds])
    sents = sents.view(-1,bsz,bptt)
    sents = sents.swapaxes(1,2)
    sents.to(device)
    if compute_masks:
        masks = torch.from_numpy(masks[rand_inds])
        masks = masks.view(-1,bsz,bptt,bptt+1)
        masks.to(device)
    if aux_objective:
        aux_labels = torch.from_numpy(aux_labels[rand_inds])
        aux_labels = aux_labels.view(-1,bsz,bptt)
        aux_labels = aux_labels.swapaxes(1,2)
        aux_labels.to(device)

    return sents, masks, aux_labels



try:
    with open(args.vocab_file, 'r') as f:
        # We're using a pre-existing vocab file, so we shouldn't overwrite it
        args.predefined_vocab_flag = True
except FileNotFoundError:
    # We should create a new vocab file
    args.predefined_vocab_flag = False

corpus = data.SentenceCorpus(args.data_dir, args.vocab_file, args.test, args.interact,
                             checkpoint_flag=args.load_checkpoint,
                             predefined_vocab_flag=args.predefined_vocab_flag,
                             collapse_nums_flag=args.collapse_nums_flag,
                             multisentence_test_flag=args.multisentence_test,
                             lower_flag=args.lowercase,
                             trainfname=args.trainfname,
                             validfname=args.validfname,
                             testfname=args.testfname,
                             aux_objective_flag=args.aux_objective)

if not args.interact:
    if args.test:
        print('testing')
        if args.multisentence_test:
            test_data = [corpus.test]
        else:
            test_sents, test_data = corpus.test
    else:
        compute_masks = (args.model == 'CUEBASEDRNN')
        if(args.aux_objective):
            train_aux_data = corpus.train_aux_labels
            val_aux_data = corpus.valid_aux_labels
        else:
            train_aux_data = None
            val_aux_data = None
        if(compute_masks):
            print("Packing data into batches + computing sentence-level attention masks")
        else:
            print("Packing data into batches")
        train_data, train_masks, train_aux_labels = sent_level_batchify(corpus.train, args.bptt, args.batch_size, compute_masks=compute_masks, aux_data=corpus.train_aux_labels)
        #train_data = batchify(corpus.train, args.batch_size)
        val_data, val_masks, val_aux_labels = sent_level_batchify(corpus.valid, args.bptt, args.batch_size, compute_masks=compute_masks, aux_data=corpus.valid_aux_labels)
print('Data processing complete; Building model')
###############################################################################
# Build/load the model
###############################################################################

if not args.test and not args.interact:
    if args.load_checkpoint:
        # Load the best saved model.
        print('  Continuing training from previous checkpoint')
        with open(args.model_file, 'rb') as f:
            if args.cuda:
                model = torch.load(f).to(device)
            else:
                model = torch.load(f, map_location='cpu')
    else:
        ntokens = len(corpus.dictionary)
        if(args.aux_objective):
            nauxclasses = len(corpus.aux_dictionary)
        else:
            nauxclasses = 0
        if(args.model != 'CUEBASEDRNN'):
            model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid,
                               args.nlayers, embedding_file=args.embedding_file,
                               dropout=args.dropout, tie_weights=args.tied,
                               freeze_embedding=args.freeze_embedding, aux_objective=args.aux_objective).to(device)
        else:
            model = model.CueBasedRNNModel(ntokens, args.emsize, args.nhid,
                               args.nlayers, embedding_file=args.embedding_file,
                               dropout=args.dropout, tie_weights=args.tied,
                               freeze_embedding=args.freeze_embedding, 
                               aux_objective=args.aux_objective,
                               nauxclasses=nauxclasses).to(device)

    if args.cuda and (not args.single) and (torch.cuda.device_count() > 1):
        # If applicable, use multi-gpu for training
        # Scatters minibatches (in dim=1) across available GPUs
        model = nn.DataParallel(model, dim=1)
    if isinstance(model, torch.nn.DataParallel):
        # if multi-gpu, access real model for training
        model = model.module
    # after load the rnn params are not a continuous chunk of memory
    # this makes them a continuous chunk, and will speed up forward pass
    if(args.model != 'CUEBASEDRNN'):
        model.rnn.flatten_parameters()
    # setup model with optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',patience=0,factor=0.1)

criterion = nn.CrossEntropyLoss(reduction='none')

###############################################################################
# Complexity measures
###############################################################################

def get_entropy(state):
    ''' Computes entropy of input vector '''
    # state should be a vector scoring possible classes
    # returns a scalar entropy over state
    if args.complexn == 0:
        beam = state
    else:
        # duplicate state but with all losing guesses set to 0
        beamk, beamix = torch.topk(state, args.complexn, 0)
        if args.softcliptopk:
            beam = torch.FloatTensor(state.size()).to(device).fill_(0).scatter(0, beamix, beamk)
        else:
            beam = torch.FloatTensor(state.size()).to(device).fill_(float("-inf")).scatter(0, beamix, beamk)

    probs = nn.functional.softmax(beam, dim=0)
    # log_softmax is numerically more stable than two separate operations
    logprobs = nn.functional.log_softmax(beam, dim=0)
    prod = probs.data * logprobs.data
    # sum but ignore nans
    return torch.Tensor([-1 * torch.sum(prod[prod == prod])]).to(device)

def get_surps(state):
    ''' Computes surprisal for each element in given vector '''
    # state should be a vector scoring possible classes
    # returns a vector containing the surprisal of each class in state
    if args.complexn == 0:
        beam = state
    else:
        # duplicate state but with all losing guesses set to 0
        beamk, beamix = torch.topk(state, args.complexn, 0)
        if args.softcliptopk:
            beam = torch.FloatTensor(state.size()).to(device).fill_(0).scatter(0, beamix, beamk)
        else:
            beam = torch.FloatTensor(state.size()).to(device).fill_(float("-inf")).scatter(0, beamix, beamk)

    logprobs = nn.functional.log_softmax(beam, dim=0)
    return -1 * logprobs

def get_complexity(state, obs, sentid):
    ''' Generates complexity output for given state, observation, and sentid '''
    Hs = torch.log2(torch.exp(torch.squeeze(apply(get_entropy, state))))
    surps = torch.log2(torch.exp(apply(get_surps, state)))

    for corpuspos, targ in enumerate(obs):
        word = corpus.dictionary.idx2word[int(targ)]
        if word == '<eos>':
            # don't output the complexity of EOS
            continue
        surp = surps[corpuspos][int(targ)]
        if args.guess:
            outputguesses = []
            guessscores, guesses = torch.topk(surps[corpuspos], args.guessn, dim= -1, largest=False)
            for guess_ix in range(args.guessn):
                outputguesses.append(corpus.dictionary.idx2word[int(guesses[corpuspos][guess_ix])])
                if args.guesssurps:
                    # output guess surps
                    outputguesses.append("{:.3f}".format(float(guessscores[guess_ix])))
                elif args.guessprobs:
                    # output probabilities
                    outputguesses.append("{:.3f}".format(2**(float(-1*guessscores[guess_ix]))))
            outputguesses = args.csep.join(outputguesses)
            print(args.csep.join([str(word), str(sentid), str(corpuspos), str(len(word)),
                                  str(float(surp)), str(float(Hs[corpuspos])),
                                  str(max(0, float(Hs[max(corpuspos-1, 0)])-float(Hs[corpuspos]))),
                                  str(outputguesses)]))
        else:
            print(args.csep.join([str(word), str(sentid), str(corpuspos), str(len(word)),
                                  str(float(surp)), str(float(Hs[corpuspos])),
                                  str(max(0, float(Hs[max(corpuspos-1, 0)])-float(Hs[corpuspos])))]))

def apply(func, apply_dimension):
    ''' Applies a function along a given dimension '''
    output_list = [func(m) for m in torch.unbind(apply_dimension, dim=0)]
    return torch.stack(output_list, dim=0)

###############################################################################
# Training code
###############################################################################

def repackage_hidden(in_state):
    """ Wraps hidden states in new Tensors, to detach them from their history. """
    if isinstance(in_state, torch.Tensor):
        return in_state.detach()
    else:
        return tuple(repackage_hidden(value) for value in in_state)

def get_batch(source, i, prebatched=False, masks_source=None, aux_labels_source=None):
    """ get_batch subdivides the source data into chunks of length args.bptt.
    If source is equal to the example output of the batchify function, with
    a bptt-limit of 2, we'd get the following two Variables for i = 0:
    a g m s      b h n t
    b h n t      c i o u
    Note that despite the name of the function, the subdivison of data is not
    done along the batch dimension (i.e. dimension 1), since that was handled
    by the batchify function. The chunks are along dimension 0, corresponding
    to the seq_len dimension in the LSTM. """
    if(not prebatched):
        seq_len = min(args.bptt, len(source) - 1 - i)
        data = source[i:i+seq_len]
        target = source[i+1:i+1+seq_len]
        return data, target.long()
    else:
        masks = None
        aux_labels = None
        data = source[i,:-1,:]
        target = source[i,1:,:]
        if(masks_source is not None):
            masks = masks_source[i,:,:-1,:-1]
        if(aux_labels_source is not None):
            aux_labels = aux_labels_source[i,:-1,:]
        return data, target.long(), masks, aux_labels

def test_get_batch(source):
    """ Creates an input/target pair for evaluation """
    seq_len = len(source) - 1
    data = source[:seq_len]
    target = source[1:1+seq_len].view(-1)
    return data, target.long()

def test_evaluate(test_sentences, data_source):
    """ Evaluate at test time (with adaptation, complexity output) """
    # Turn on evaluation mode which disables dropout.
    if args.adapt:
        # Must disable cuDNN in order to backprop during eval
        torch.backends.cudnn.enabled = False
    model.eval()
    total_loss = 0.
    ntokens = len(corpus.dictionary)
    nwords = 0
    if args.complexn > ntokens or args.complexn <= 0:
        args.complexn = ntokens
        if args.guessn > ntokens:
            args.guessn = ntokens
        sys.stderr.write('Using beamsize: '+str(ntokens)+'\n')
    else:
        sys.stderr.write('Using beamsize: '+str(args.complexn)+'\n')

    if args.words:
        if not args.nocheader:
            if args.complexn == ntokens:
                print('word{0}sentid{0}sentpos{0}wlen{0}surp{0}entropy{0}entred'.format(args.csep), end='')
            else:
                print('word{0}sentid{0}sentpos{0}wlen{0}surp{1}{0}entropy{1}{0}entred{1}'.format(args.csep, args.complexn), end='')
            if args.guess:
                for i in range(args.guessn):
                    print('{0}guess'.format(args.csep)+str(i), end='')
                    if args.guesssurps:
                        print('{0}gsurp'.format(args.csep)+str(i), end='')
                    elif args.guessprobs:
                        print('{0}gprob'.format(args.csep)+str(i), end='')
            sys.stdout.write('\n')
    if PROGRESS:
        bar = Bar('Processing', max=len(data_source))
    for i in range(len(data_source)):
        sent_ids = data_source[i].to(device)
        # We predict all words but the first, so determine loss for those
        if test_sentences:
            sent = test_sentences[i]
        if(args.model != 'CUEBASEDRNN'):
            hidden = model.init_hidden(1) # number of parallel sentences being processed
        data, targets = test_get_batch(sent_ids)
        nwords += targets.flatten().size(0)
        if args.view_layer >= 0:
            for word_index in range(data.size(0)):
                # Starting each batch, detach the hidden state
                
                #hidden = repackage_hidden(hidden)
                model.zero_grad()

                word_input = data[word_index].unsqueeze(0).unsqueeze(1)
                target = targets[word_index].unsqueeze(0)
                if(args.model != 'CUEBASEDRNN'):
                    output, hidden = model(word_input, hidden)
                else:
                    output, hidden = model(word_input)
                output_flat = output.view(-1, ntokens)
                loss = criterion(output_flat, target)
                total_loss += loss.sum().item()
                input_word = corpus.dictionary.idx2word[int(word_input.data)]
                targ_word = corpus.dictionary.idx2word[int(target.data)]
                if input_word != '<eos>': # not in (input_word,targ_word):
                    if args.verbose_view_layer:
                        print(input_word,end=" ")
                    # don't output <eos> markers to align with input
                    # output raw activations
                    if args.view_hidden:
                        # output hidden state
                        print(*list(hidden[0][args.view_layer].view(1, -1).data.cpu().numpy().flatten()), sep=' ')

                    elif args.view_emb:
                        #Get embedding for input word
                        emb = model.encoder(word_input)
                        # output embedding
                        print(*list(emb[0].view(1,-1).data.cpu().numpy().flatten()), sep=' ')

                    else:
                        # output cell state
                        print(*list(hidden[1][args.view_layer].view(1, -1).data.cpu().numpy().flatten()), sep=' ')
                hidden = repackage_hidden(hidden)
        else:
            data = data.unsqueeze(1) # only needed when a single sentence is being processed
            if(args.model != 'CUEBASEDRNN'):
                output, hidden = model(data, hidden)
            else:
                output, hidden = model(data)
            try:
                output_flat = output.view(-1, ntokens)
            except RuntimeError:
                print("Vocabulary Error! Most likely there weren't unks in training and unks are now needed for testing")
                raise
            loss = criterion(output_flat, targets)
            total_loss += loss.sum().item()
            if args.words:
                # output word-level complexity metrics
                get_complexity(output_flat, targets, i)
            else:
                # output sentence-level loss
                if test_sentences:
                    print(str(sent)+":"+str(loss))
                else:
                    print(str(loss))

            if args.adapt:
                loss.mean().backward()

                # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
                torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
                optimizer.step()
                hidden = repackage_hidden(hidden)
                model.zero_grad()

        if PROGRESS:
            bar.next()
    if PROGRESS:
        bar.finish()
    return total_loss / nwords

def evaluate(data_source, aux_source=None):
    """ Evaluate for validation (no adaptation, no complexity output) """
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_lm_loss = 0.
    total_aux_loss = 0.
    ntokens = len(corpus.dictionary)
    if(args.aux_objective):
        nauxclasses = len(corpus.aux_dictionary)
    with torch.no_grad():
        # Construct hidden layers for each sub-batch
        hidden_batch = []
        if(args.model != 'CUEBASEDRNN'):
            for i in range(args.grad_accumulation_steps):
                hidden_batch.append(model.init_hidden(args.batch_size))

        for i in range(data_source.size(0)):
            if(args.model == 'CUEBASEDRNN'):
                batch_data, batch_targets, batch_masks, batch_targets_aux = get_batch(data_source, i, prebatched=True, masks_source=val_masks, aux_labels_source=aux_source)
                output, hidden_subbatch, aux_preds = model(batch_data, batch_masks)
                hidden_batch.append(hidden_subbatch[-1])  
            else:
                batch_data, batch_targets = get_batch(data_source, i, prebatched=True)
                output, hidden_batch = model(batch_data, hidden_batch)
         
            output_flat = output.view(-1, ntokens)
            lm_loss = criterion(output_flat, batch_targets.flatten()).sum().item()
            total_lm_loss += lm_loss
            if(args.aux_objective):
                aux_preds_flat = aux_preds.view(-1, nauxclasses)
                aux_loss = criterion(aux_preds_flat, batch_targets_aux.flatten()).sum().item()
                total_aux_loss += aux_loss

    return (total_lm_loss / (data_source.flatten().size(0))), (total_aux_loss / (data_source.flatten().size(0)))

def train():
    """ Train language model """
    # Turn on training mode which enables dropout.
    model.train()
    total_lm_loss = 0.
    total_aux_loss = 0.
    total_data = 0.
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    nauxclasses = len(corpus.aux_dictionary)
    actual_batch_size = int(args.batch_size / args.grad_accumulation_steps)
    hidden_batch = []
    #if model is RNN, then pre-initialize hidden states for the batch, CUE-based model does this in the forward pass (which is bad?)
    if(args.model != 'CUEBASEDRNN'):
        for i in range(args.grad_accumulation_steps):
            hidden_batch.append(model.init_hidden(actual_batch_size))
        
    for batch in range(0, train_data.size(0)):
        if(args.model == 'CUEBASEDRNN'):
            batch_data, batch_targets, batch_masks, batch_targets_aux = get_batch(train_data, batch, prebatched=True, masks_source=train_masks, aux_labels_source=train_aux_labels)
            output, hidden_batch_all, aux_preds = model(batch_data, masks=batch_masks)
            hidden_batch = hidden_batch_all[-1]
        else:
            batch_data, batch_targets = get_batch(train_data, batch)
            output, hidden_batch = model(batch_data, hidden_batch)
        
        output_flat = output.view(-1, ntokens)
        lm_loss = criterion(output_flat, batch_targets.flatten())
        if(args.aux_objective):
            aux_preds_flat = aux_preds.view(-1, nauxclasses)
            aux_loss = criterion(aux_preds_flat, batch_targets_aux.flatten())
            loss = lm_loss + aux_loss
        else:
            loss = lm_loss
        total_lm_loss += lm_loss.sum().item()
        total_aux_loss += aux_loss.sum().item()
        loss.mean().backward()
        total_data += batch_data.flatten().size(0)
        
        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm(model.parameters(), args.clip)
        optimizer.step()

        # Detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden_batch = repackage_hidden(hidden_batch)
        model.zero_grad()

        if batch % args.log_interval == 0 and batch > 0:
            curr_lm_loss = total_lm_loss / total_data
            curr_aux_loss = total_aux_loss / total_data
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                  'lm loss {:5.2f} | lm ppl {:8.2f} | aux loss {:5.2f} | aux ppl {:8.2f}'.format(
                      epoch, batch, train_data.size(0), float(optimizer.param_groups[0]['lr']),
                      elapsed * 1000 / args.log_interval, curr_lm_loss, math.exp(curr_lm_loss), curr_aux_loss, math.exp(curr_aux_loss)))
            total_lm_loss = 0.
            total_aux_loss = 0.
            total_data = 0.
            start_time = time.time()

print('Beginning evaluation/training')
# Loop over epochs.
best_val_loss = None
no_improvement = 0

# At any point you can hit Ctrl + C to break out of training early.
if not args.test and not args.interact:
    try:
        for epoch in range(1, args.epochs+1):
            epoch_start_time = time.time()
            train()
            val_lm_loss, val_aux_loss = evaluate(val_data, val_aux_labels)
            combined_loss = (val_lm_loss + val_aux_loss)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | lr: {:4.8f} | '
                  'valid lm ppl {:8.2f} | valid aux ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                             float(optimizer.param_groups[0]['lr']), math.exp(val_lm_loss), math.exp(val_aux_loss)))
            print('-' * 89)
            
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or combined_loss< best_val_loss:
                no_improvement = 0
                with open(args.model_file, 'wb') as f:
                    torch.save(model, f)
                    best_val_loss = combined_loss
            else:
                # Anneal the learning rate if no more improvement in the validation dataset.
                no_improvement += 1
                if no_improvement >= 3:
                    print('Covergence achieved! Ending training early')
                    break
            scheduler.step(combined_loss)
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')
else:
    # Load the best saved model.
    with open(args.model_file, 'rb') as f:
        if args.cuda:
            model = torch.load(f).to(device)
        else:
            model = torch.load(f, map_location='cpu')

        if args.init is not None:
            if args.init != -1:
                model.set_parameters(args.init)
            else:
                model.randomize_parameters()

        # after load the rnn params are not a continuous chunk of memory
        # this makes them a continuous chunk, and will speed up forward pass
        if isinstance(model, torch.nn.DataParallel):
            # if multi-gpu, access real model for testing
            model = model.module
        if(args.model != 'CUEBASEDRNN'):
            model.rnn.flatten_parameters()
        
        # setup model with optimizer and scheduler
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',patience=0,factor=0.1)

    # Run on test data.
    if args.interact:
        # First fix Python 2.x input command
        try:
            input = raw_input
        except NameError:
            pass

        n_rnn_param = sum([p.nelement() for p in model.rnn.parameters()])
        n_enc_param = sum([p.nelement() for p in model.encoder.parameters()])
        n_dec_param = sum([p.nelement() for p in model.decoder.parameters()])

        print('#rnn params = {}'.format(n_rnn_param))
        print('#enc params = {}'.format(n_enc_param))
        print('#dec params = {}'.format(n_dec_param))


        # Then run interactively
        print('Running in interactive mode. Ctrl+c to exit')
        if '<unk>' not in corpus.dictionary.word2idx:
            print('WARNING: Model does not have unk probabilities.')
        try:
            while True:
                instr = input('Input a sentence: ')
                test_sents, test_data = corpus.online_tokenize_with_unks(instr)
                try:
                    test_evaluate(test_sents, test_data)
                except:
                    print("RuntimeError: Most likely one of the input words was out-of-vocabulary.")
                    print("    Retrain the model with\
                            A) explicit '<unk>'s in the training set\n    \
                            or B) words in validation that aren't present in training.")
                if args.adapt:
                    with open(args.adapted_model, 'wb') as f:
                        torch.save(model, f)
        except KeyboardInterrupt:
            print(' ')
    else:
        if not args.adapt:
            torch.set_grad_enabled(False)
        if args.multisentence_test:
            test_loss = test_evaluate(None, test_data)
        else:
            test_loss = test_evaluate(test_sents, test_data)
        if args.adapt:
            with open(args.adapted_model, 'wb') as f:
                torch.save(model, f)
    if not args.interact and not args.nopp:
        print('=' * 89)
        print('| End of testing | test loss {:5.2f} | test ppl {:8.2f}'.format(
            test_loss, math.exp(test_loss)))
        print('=' * 89)
