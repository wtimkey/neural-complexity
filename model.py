""" Model data structures """

import numpy as np
import torch.nn as nn
import torch

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers,
                 embedding_file=None, dropout=0.5, tie_weights=False, freeze_embedding=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        if embedding_file:
            # Use pre-trained embeddings
            embed_weights = self.load_embeddings(embedding_file, ntoken, ninp)
            self.encoder = nn.Embedding.from_pretrained(embed_weights)
        else:
            self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        self.init_weights(freeze_embedding)
        if freeze_embedding:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2017)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers:
        # A Loss Framework for Language Modeling" (Inan et al. 2017)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self, freeze_embedding):
        """ Initialize encoder and decoder weights """
        initrange = 0.1
        if not freeze_embedding:
            self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def zero_parameters(self):
        """ Set all parameters to zero (likely as a baseline) """
        self.encoder.weight.data.fill_(0)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.fill_(0)
        for weight in self.rnn.parameters():
            weight.data.fill_(0)

    def random_parameters(self):
        """ Randomly initialize all RNN parameters but not the encoder or decoder """
        initrange = 0.1
        for weight in self.rnn.parameters():
            weight.data.uniform_(-initrange, initrange)

    def load_embeddings(self, embedding_file, ntoken, ninp):
        """ Load pre-trained embedding weights """
        weights = np.empty((ntoken, ninp))
        with open(embedding_file, 'r') as in_file:
            ctr = 0
            for line in in_file:
                weights[ctr, :] = np.array([float(w) for w in line.strip().split()[1:]])
                ctr += 1
        return(torch.tensor(weights).float())

    def forward(self, observation, hidden):
        emb = self.drop(self.encoder(observation))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        """ Initialize a fresh hidden state """
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return (torch.tensor(weight.new(self.nlayers, bsz, self.nhid).zero_()),
                    torch.tensor(weight.new(self.nlayers, bsz, self.nhid).zero_()))
        else:
            return torch.tensor(weight.new(self.nlayers, bsz, self.nhid).zero_())

    def set_parameters(self,init_val):
        for weight in self.rnn.parameters():
            weight.data.fill_(init_val)
        self.encoder.weight.data.fill_(init_val)
        self.decoder.weight.data.fill_(init_val)

    def randomize_parameters(self):
        initrange = 0.1
        for weight in self.rnn.parameters():
            weight.data.uniform_(-initrange, initrange)

class CueBasedRNNModel(nn.Module):
    def __init__(self, ntoken, ninp, nhid, nlayers,
                 embedding_file=None, dropout=0.5, tie_weights=False, freeze_embedding=False,
                 aux_objective=False, nauxclasses=0, attn_score_function='softmax'):
        super().__init__()
        self.tanh = nn.Tanh()
        self.drop = nn.Dropout(dropout)
        if(attn_score_function=='softmax'):
            self.score_attn = nn.Softmax(dim=-1)

        if embedding_file:
            # Use pre-trained embeddingss
            embed_weights = self.load_embeddings(embedding_file, ntoken, ninp)
            self.encoder = nn.Embedding.from_pretrained(embed_weights)
        else:
            self.encoder = nn.Embedding(ntoken, ninp)
        
        #generate query from hidden state and embedding
        self.q = nn.Linear(ninp+nhid,nhid)
        #project from prev hidden state, embedding, query, attn to large intermediate layer
        self.intermediate_h = nn.Linear(nhid*4,nhid*4)
        #from large intermediate layer to current word key, value, and next-word prediction
        self.final_h = nn.Linear(nhid*4,nhid*3)

        self.decoder = nn.Linear(nhid, ntoken)
        self.aux_objective = aux_objective
        if(aux_objective):
            self.aux_decoder = nn.Linear(nhid, nauxclasses)

        self.init_weights(freeze_embedding, aux_objective=False)
        if freeze_embedding:
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.q_norm = torch.nn.LayerNorm(nhid)
        self.int_norm = torch.nn.LayerNorm(nhid * 4)
        self.f_norm = torch.nn.LayerNorm(nhid * 3)


        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2017)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers:
        # A Loss Framework for Language Modeling" (Inan et al. 2017)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.nhid = nhid
        self.attn_div_factor = np.sqrt(nhid)

    def init_weights(self, freeze_embedding, aux_objective):
        """ Initialize encoder and decoder weights """
        initrange = 0.1
        if not freeze_embedding:
            self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)
        if(aux_objective):
            self.aux_decoder.bias.data.fill_(0)
            self.aux_decoder.weight.data.uniform_(-initrange, initrange)


    def zero_parameters(self):
        """ Set all parameters to zero (likely as a baseline) """
        self.encoder.weight.data.fill_(0)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.fill_(0)
        for weight in self.rnn.parameters():
            weight.data.fill_(0)

    def random_parameters(self):
        """ Randomly initialize all RNN parameters but not the encoder or decoder """
        initrange = 0.1
        for module in [self.q, self.intermediate_h, self.final_h]:
            for weight in module.parameters():
                weight.data.uniform_(-initrange, initrange)

    def load_embeddings(self, embedding_file, ntoken, ninp):
        """ Load pre-trained embedding weights """
        weights = np.empty((ntoken, ninp))
        with open(embedding_file, 'r') as in_file:
            ctr = 0
            for line in in_file:
                weights[ctr, :] = np.array([float(w) for w in line.strip().split()[1:]])
                ctr += 1
        return(torch.tensor(weights).float())
    
    #b = batch size, n = sequence length, d = dimensionality 
    #masks are of size b * n * n+1 - dim 3 is token a attending token b in dim 4
    def forward(self, observation, masks):
        #todo - initialize outside of forward pass
        hidden, key_cache, value_cache = self.init_cache(observation)
        seq_len = observation.size(dim=0)
        emb = self.drop(self.encoder(observation))

        for i in range(seq_len):
            #self-attention
            #generate query from prev. hidden state and curr. word embedding
            query = self.drop(self.tanh(self.q_norm(self.q(torch.cat((emb[i],hidden[i]), -1))))) #b * d
            query_n = query.unsqueeze(-1) #b * n * 1

            #compute attention scores against all keys w_-1 to w_i-1
            #todo: make w_-1 inaccessible for attention after first pass

            #[b * i * d] * [b * d * 1] = b * i * 1 squeeze() -> [b * i]
            attn_scores = torch.bmm(key_cache.swapaxes(0,1), query_n).squeeze(dim=-1)
            masked_scores = attn_scores + masks[:,i,:i+1]
            #divide scores by sqrt(nhid) for more stable gradients, then compute score using specified function (default: softmax)
            attn_weights = self.score_attn(masked_scores / self.attn_div_factor)  #b * i or b when i=0 - is this a problem?
            #weighted sum over attention scores to get final attention vector                    
            #[(i * b) * 1] * [i * b * d] = i * b * d.sum(axis = 0) = b * d
            attn = (attn_weights.T.unsqueeze(-1) * value_cache).sum(axis=0) #b * d  - attn_weights b * i * 1

            #feed-forward component
            #project to large intermediate layer
            intermediate = self.drop(self.tanh(self.int_norm(self.intermediate_h(torch.cat((emb[i],query,attn,hidden[i]),-1))))) #[b * 4d] -> [b * 4d]
            
            #project to final layer to generate current word key, final hidden state used for prediction
            key_cache_i, value_cache_i, hidden_i = self.drop(self.tanh(self.f_norm(self.final_h(intermediate)))).split(self.nhid, dim=-1) #[b * 4d] -> [b * 3d]
            #update cache
            #todo: are there possible speedups where there's no need to concatinate dimensions like this?
            #todo: (cont) maybe one option is torch.zeros() but only do not product on first i tokens, to avoid issue with overwriting
            #each is [i*b*d]
            hidden = torch.cat((hidden, hidden_i.unsqueeze(0)), dim=0)
            key_cache = torch.cat((key_cache, key_cache_i.unsqueeze(0)), dim=0)
            value_cache = torch.cat((value_cache, value_cache_i.unsqueeze(0)), dim=0)


        output = hidden[1:]
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2))) #[35 * 20 * 128] -> reshape [700 * 128]
        decoded = decoded.view(output.size(0), output.size(1), decoded.size(1)) #
        if(self.aux_objective):
            decoded_aux = self.aux_decoder((output.view(output.size(0)*output.size(1), output.size(2))))
            decoded_aux = decoded_aux.view(output.size(0), output.size(1), decoded.size(1))
        else:
            decoded_aux = None
        return decoded, hidden

    def init_cache(self, observation):
        if len(observation.size())>1:
            bsz = observation.size(dim=-1)
        else:
            bsz = 1
        seq_len = observation.size(dim=0)

        return torch.zeros(1, bsz, self.nhid), torch.zeros(1, bsz, self.nhid), torch.zeros(1, bsz, self.nhid)

    def set_parameters(self,init_val):
        for module in [self.q, self.intermediate_h, self.final_h]:
            for weight in module.parameters():
                weight.data.fill_(init_val)
            self.encoder.weight.data.fill_(init_val)
            self.decoder.weight.data.fill_(init_val)

    def randomize_parameters(self):
        initrange = 0.1
        for module in [self.q, self.intermediate_h, self.final_h]:
            for weight in module.parameters():
                weight.data.uniform_(-initrange, initrange)

