from data import START_TOKEN
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers, bias=True):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                           num_layers=num_layers, bias=bias, batch_first=True,
                           bidirectional=True)
        self.reduce_h = nn.Linear(2*hidden_size, hidden_size)
        self.reduce_c = nn.Linear(2*hidden_size, hidden_size)

    def forward(self, embeded_ids, sentence_len):
        '''
        embeded_ids: FloatTensor(batch_size * l * input_size), 
                     l = max(sentence_len).
        sentence_len: Tensor(batch_size)
                      sentence_len is in decreasing order which reqiures
                      ids is in decreasing order in the first(batch) dimension.

        Return:
            outputs: FloatTensor(batch_size, l, hidden_size)
            h_n: FloatTensor(num_layers, batch, hidden_size)
            c_n: FloatTensor(num_layers, batch, hidden_size)

        '''
        # total_length is used for handling the problem with data_parallel
        # More detials at:
        #   https://pytorch.org/docs/stable/notes/faq.html#pack-rnn-unpack-with-data-parallelism
        _, total_length, _ = embeded_ids.size()
        packed_inputs = nn.utils.rnn.pack_padded_sequence(
                embeded_ids, sentence_len, batch_first=True
            )
        #self.rnn.flatten_parameters()
        packed_outputs, (h_n, c_n) = self.rnn(packed_inputs)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
                packed_outputs, batch_first=True, total_length=total_length
            )
        h_n = torch.stack([
                    torch.cat(
                        (h_n[2*i], h_n[2*i+1]), dim=1
                    ) for i in range(self.num_layers)
            ])
        c_n = torch.stack([
                    torch.cat(
                        (c_n[2*i], c_n[2*i+1]), dim=1
                    ) for i in range(self.num_layers)
            ])

        h_n = F.relu(self.reduce_h(h_n.permute(1, 0, 2)))
        c_n = F.relu(self.reduce_c(c_n.permute(1, 0, 2)))
        outputs = F.relu(self.reduce_h(outputs))

        return outputs, h_n.permute(1, 0, 2), c_n.permute(1, 0, 2)


class Decoder(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers, bias=True):
        super(Decoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.LSTMCell(input_size, hidden_size))
            input_size = hidden_size

    def forward(self, input_vec, h_t, c_t):
        '''
        Input_vec: FloatTensor (batch_size, input_size)
        h_t: FloatTensor (num_layers, batch_size, hidden_size)
        c_t: FloatTensor (num_layers, batch_size, hidden_size)

        Return:
            output: FloatTensor (batch_size, hidden_size)
            h_tp1: FloatTensor (num_layers, batch_size, hidden_size)
            c_tp1: FloatTensor (num_layers, batch_size, hidden_size)
        '''
        h_tp1 = []
        c_tp1 = []
        for i, layer in enumerate(self.layers):
            h_i_tp1, c_i_tp1 = layer(input_vec, (h_t[i], c_t[i]))
            h_tp1 = h_tp1 + [h_i_tp1]
            c_tp1 = c_tp1 + [c_i_tp1]
            input_vec = h_i_tp1

        h_tp1 = torch.stack(h_tp1, dim=0)
        c_tp1 = torch.stack(c_tp1, dim=0)
        output = input_vec

        return output, h_tp1, c_tp1



class Attention(nn.Module):
    
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size

        self.attn_matrix = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, dec_vec, enc_states, mask):
        '''
        dec_vec: FloatTensor(batch_size, hidden_size)
        enc_states: FloatTensor(batch_size, l, hidden_size)
        mask: FloatTensor(batch_size, l)

        Here we perform attention in Luong 2015
        score(h_dec, h_enc) = h_dec^T W h_enc

        Return:
            context_vec: FloatTensor(batch_size, hidden_size)
        '''
        # score (batch_size, l)
        score = torch.squeeze(
                torch.bmm(
                    enc_states, torch.unsqueeze(self.attn_matrix(dec_vec), -1)
                    )
                )
        # attn_weight (batch_size, l)
        attn_weight = F.softmax(score, dim=1)
        attn_weight = attn_weight * mask
        div = torch.unsqueeze(torch.sum(attn_weight, dim=1), -1)
        div = div.expand_as(attn_weight)
        attn_weight = attn_weight / div

        # weighted sum of enc_states
        attn_weight = torch.unsqueeze(attn_weight, dim=1) #(batch_size, 1, l)
        # context_vec (batch_size, hidden_size)
        context_vec = torch.squeeze(torch.bmm(attn_weight, enc_states))

        return context_vec



class Discriminator(nn.Module):
    
    def __init__(self, input_size, hidden_size, num_layers):
        super(Discriminator, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(nn.Linear(input_size, hidden_size))
            elif i + 1 == num_layers:
                self.layers.append(nn.Linear(hidden_size, 1))
            else:
                self.layers.append(nn.Linear(hidden_size, hidden_size))

    def forward(self, embed_vec):
        '''
        embed_vec: FloatTensor (batch_size, ..., input_size)

        Return:
            probs: FloatTensor (batch_size, ...)
                  The probability of embed_vec coming from source language
        '''
        o = embed_vec
        for i, layer in enumerate(self.layers):
            o = layer(o)
            if i < self.num_layers - 1:
                o = F.leaky_relu(o)
        probs = F.sigmoid(o)
        return probs.squeeze()


class Embedding(nn.Module):
    
    def __init__(self, embed_dim, vocab_size, num_special_tokens,
                 embedding_matrix=None):
        '''
        If embedding_matrix is not None. It must be a numpy array.
        The size of it is (vocab_size + num_special_tokens, embed_dim)
        '''
        super(Embedding, self).__init__()
        self.embed_layer = nn.Embedding(
                vocab_size+num_special_tokens, embed_dim
                )
        if embedding_matrix is not None:
            self._init_embedding_matrix(embedding_matrix)

    def _init_embedding_matrix(self, embedding_matrix):
        self.embed_layer.weight.data = embedding_matrix

    def forward(self, ids):
        return self.embed_layer(ids)


class Seq2seq_Model(nn.Module):

    def __init__(self, src_embedding, trg_embedding, encoder, decoder,
                 attention, input_size, hidden_size, vocab_size,
                 num_special_tokens, device):
        super(Seq2seq_Model, self).__init__()
        self.src_embedding = src_embedding
        self.trg_embedding = trg_embedding
        self.encoder = encoder
        self.decoder = decoder
        self.attention = attention

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_special_tokens = num_special_tokens
        self.device = device

        self.fc = nn.Linear(2*hidden_size, hidden_size)
        self.output_layer = nn.Linear(
                hidden_size, vocab_size + num_special_tokens
            )

    
    def forward(self, in_ids, ref_ids, in_mask, in_lengths, direction):
        """
        in_ids: LongTensor(batch_size * l1)
        ref_ids: LongTensor(batch_size * l2)
                 ref_ids have STOP_TOKEN
        in_mask: FloatTensor(batch_size * l1)
        in_lengths: LongTensor(batch_size)
        deriection: A String ('src2trg' or 'trg2src')

        Return:
            log_probs: A FloatTensor of size
                       (batch_size * l2 * (vocab_size+num_special_tokens)).
            enc_outputs: FloatTensor(batch_size * l1 * hidden_size)
        """
        if direction == 'src2trg':
            in_embed_layer = self.src_embedding
            ref_embed_layer = self.trg_embedding
        elif direction == 'trg2src':
            in_embed_layer = self.trg_embedding
            ref_embed_layer = self.src_embedding
        elif direction == 'src2src':
            in_embed_layer = self.src_embedding
            ref_embed_layer = self.src_embedding
        elif direction == 'trg2trg':
            in_embed_layer = self.trg_embedding
            ref_embed_layer = self.trg_embedding
        else:
            raise("Invalid direction! Can not use proper embedding!")

        # Embed batch and noise_batch
        batch_size = in_lengths.size(0)
        start_ids = torch.from_numpy(
                np.array([START_TOKEN for i in range(batch_size)])
                ).long()
        start_ids = start_ids.to(self.device)
        ref_ids = torch.cat([start_ids.unsqueeze(-1), ref_ids], dim=1)

        #print('Tensor device:{}'.format(in_ids.type()))
        #print("model device:{}".format(self.src_embedding.embed_layer.weight.type()))

        # batch_size * l1 * embed_size(input_size)
        embeded_in_ids = in_embed_layer(in_ids)
        # batch_size * l2s * embed_size(input_size) 
        embeded_ref_ids = ref_embed_layer(ref_ids) 

        # Encode
        min_len = torch.min(in_lengths).item()
        if min_len <= 0:
            print("Warning minimum length <= 0!")

        enc_outputs, h_n, c_n = self.encoder(embeded_in_ids, in_lengths)

        # Decode and calculate output probability.
        h_t = h_n
        c_t = c_n
        log_probs = []
        for dec_input_vec in torch.split(embeded_ref_ids, 1, 1):
            dec_input_vec = torch.squeeze(dec_input_vec)
            dec_output, h_t, c_t = self.decoder(dec_input_vec, h_t, c_t)
            context_vec = self.attention(dec_output, enc_outputs, in_mask)
            attention_vec = F.tanh(self.fc(
                    torch.cat([context_vec, dec_output], dim=1)
                ))
            log_prob = F.log_softmax(self.output_layer(attention_vec), dim=1)
            log_probs.append(log_prob)

        log_probs = torch.stack(log_probs[:-1], dim=1)

        return log_probs, enc_outputs
