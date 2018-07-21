from data import START_TOKEN, STOP_TOKEN, NUM_OF_SPECIAL_TOKENS
from utils import pad_ids_batch
import copy
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


class Hypothesis(object):

    def __init__(self, beam_size, batch_size, start_words, start_scores, 
                 init_h_t, init_c_t):
        '''
        beam_size: Int
        batch_size: Int
        start_words: [['w11', 'w12', ...],
                      ['w21', 'w22', ...],
                      ......
                      ['wk1', 'wk2', ...]]
                    (batch_size * beam_size)
                    Each element being a string.
        start_scores: [[s11, s12, ...],
                       [s21, s22, ...],
                       ......
                       [sk1, sk2, ...]]
                    (batch_size * beam_size)
                    Each element being a float.
        init_h_t: FloatTensor(num_layers, batch_size, hidden_dim)
        init_c_t: FloatTensor(num_layers, batch_size, hidden_dim)
        '''
        self.beam_size = beam_size
        self.batch_size = batch_size
        self.states = []

        h_t = init_h_t.permute(1, 0, 2)
        c_t = init_c_t.permute(1, 0, 2)
        for j in range(beam_size):
            for i in range(batch_size):
                self.states.append([[start_scores[i][j]], [start_words[i][j]],
                                    h_t[i], c_t[i], False])


    def collect(self, batch_idx):
        ret = []
        for i in range(self.beam_size):
            s = self.states[i * self.batch_size + batch_idx]
            ret.append(
                [s[0], s[1], None, None, s[4]]
                )

        return ret

    def sort_and_choose(self, candidates):
        assert len(candidates) == self.beam_size * self.beam_size
        sorted_candidates = sorted(
            candidates, key=lambda item: -sum(item[0]) / len(item[0])
            )

        return sorted_candidates[:self.beam_size]

    def get_hidden(self):
        h_t = []
        c_t = []
        for item in self.states:
            h_t.append(item[2])
            c_t.append(item[3])
        h_t = torch.stack(h_t, dim=0)
        c_t = torch.stack(c_t, dim=0)

        return h_t, c_t

    def get_prev_words(self, vocab):
        prev_words = []
        for item in self.states:
            prev_words.append(item[1][-1])

        prev_words = [vocab.word2id[w] for w in prev_words]
        prev_words = torch.from_numpy(np.array(prev_words)).long()

        return prev_words

    def get_candidates(self, i_batch, scores, indices, h_t, c_t, vocab,
                       max_len):
        '''
        scores: FloatTensor(batch_size*beam_size, beam_size)
        indices: LongTensor(batch_size*beam_size, beam_size)
        h_t: FloatTensor(batch_size*beam_size, num_layers, hidden_size)
        c_t: FloatTensor(batch_size*beam_size, num_layers, hidden_size)
        vocab: A data.Vocab instance.
        max_len: A integer.
        '''
        prev = self.collect(i_batch)
        candidates = [[copy.deepcopy(prev[l]) for m in range(self.beam_size)] \
                       for l in range(self.beam_size)]
        idx = [i_batch + n * self.batch_size for n in range(self.beam_size)]
        idx = torch.from_numpy(np.array(idx)).long()
        scores_i_batch = scores[idx].cpu().numpy().tolist()
        indices_i_batch = indices[idx].cpu().numpy().tolist()

        h_t_i_batch = h_t[idx]
        c_t_i_batch = c_t[idx]

        for j in range(self.beam_size):
            for k in range(self.beam_size):
                w = vocab.id2word[indices_i_batch[j][k]]
                if (len(candidates[j][k][1]) == max_len) or (w == '</s>'):
                    candidates[j][k][-1] = True

                if not candidates[j][k][-1]:
                    candidates[j][k][0].append(scores_i_batch[j][k])
                    candidates[j][k][1].append(w)
                    

                candidates[j][k][2] = h_t_i_batch[j]
                candidates[j][k][3] = c_t_i_batch[j]

        candidates_ = []
        for j in range(self.beam_size):
            for k in range(self.beam_size):
                candidates_.append(candidates[j][k])

        return candidates_

    def extend(self, new_entries):
        '''
        new_entries: A List with each element also being a List.
                     [[prob1, prob2, ...], ['w1', 'w2', ...], h_t, c_t]
                     len(new_entries) = beam_size * batch_size
        '''
        for i in range(self.batch_size):
            for j in range(self.beam_size):
                self.states[j*self.batch_size+i] = \
                new_entries[i*self.beam_size+j]


class WBW_Translator(object):

    def __init__(self, src2trg_path, trg2src_path):
        self.src2trg = {}
        self.trg2src = {}
        self._build_dict(src2trg_path, 'src2trg')
        self._build_dict(trg2src_path, 'trg2src')

    def _build_dict(self, dict_path, direction):
        f = open(dict_path, 'r')
        lines = f.readlines()
        for l in lines:
            w1, w2 = l.strip().split()
            if direction == 'src2trg':
                self.src2trg[w1] = []
            elif direction == 'trg2src':
                self.trg2src[w1] = []
            else:
                raise("Invalid direction!")

        for l in lines:
            w1, w2 = l.strip().split()
            if direction == 'src2trg':
                self.src2trg[w1].append(w2)
            elif direction == 'trg2src':
                self.trg2src[w1].append(w2)

    def __call__(self, sentence_batch, direction):
        return self.translate(sentence_batch, direction)

    def translate(self, sentence_batch, direction):
        '''
        sentence_batch: A list of length batch_size
                        Each item in the list is also a list.
                        ['w1', 'w2', ...]
        direction: 'src2trg' or trg2src

        Return:
            translated_batch: A list of length batch_size
                              Each item in it is ['w1', 'w2', ...]
        '''
        if direction == 'src2trg':
            ref = self.src2trg
        elif direction == 'trg2src':
            ref = self.trg2src
        else:
            raise ("Invalid direction")

        translated_batch = []
        for sent in sentence_batch:
            translated_sent = []
            for w in sent:
                try:
                    choices = ref[w]
                    w_ = random.choice(choices)
                except KeyError:
                    w_ = '<unk>'
                translated_sent.append(w_)
            translated_batch.append(translated_sent)

        # Get mask and lengths for translated_batch
        translated_lengths = [len(s) for s in translated_batch]
        max_l = max(translated_lengths)
        translated_mask = [torch.cat(
                [torch.ones(l), torch.zeros(max_l - l)], dim=0
            ) for l in translated_lengths]
        translated_mask = torch.stack(translated_mask, dim=0)
        translated_lengths = torch.from_numpy(
                np.array(translated_lengths)
            ).long()

        return translated_batch, translated_mask, translated_lengths


class NMT_Translator(nn.Module):

    def __init__(self, src_embedding, trg_embedding, encoder, decoder,
                  attention, fc, output_layer, max_len, max_ratio, src_vocab,
                  trg_vocab, device):
        '''
        Here src_embedding, trg_embedding, encoder, decoder, attention,
        fc, output_layer are all the same as in model.Seq2seq_Model except
        their parameters are fixed.

        max_len: The maximum translation length
        max_ratio: The maximum ratio of translation length / original length.
        vocab: A data.Vocab instance
        device: Perform calculation on which device.
        '''
        super(NMT_Translator, self).__init__()
        self.src_embedding = src_embedding
        self.trg_embedding = trg_embedding
        self.encoder = encoder
        self.decoder = decoder
        self.attention = attention
        self.fc = fc
        self.output_layer = output_layer
        self.max_len = max_len
        self.max_ratio = max_ratio
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.device = device


    def forward(self, src_batch, mask, lengths, direction, beam_size=None):
        """
        src_batch: LongTensor(batch_size * l)
        mask: FloatTensor(batch_size * l)
        lengths: LongTensor(batch_size)
        direction: A String ('src2trg' or 'trg2src')
        beam_size: If not None, an integer

        Return:
            translated_batch: A List of length batch_size
                              Each element in the list is a list
                              ['w1', 'w2', ...]
            translated_mask: FloatTensor on cpu( batch_size, l) 
                             l=max(translated_lengths)
            translated_lengths: LongTensor on cpu(batch_size)
        """
        if direction == 'src2trg':
            in_embed_layer = self.src_embedding
            ref_embed_layer = self.trg_embedding
            ref_vocab = self.trg_vocab
        elif direction == 'trg2src':
            in_embed_layer = self.trg_embedding
            ref_embed_layer = self.src_embedding
            ref_vocab = self.src_vocab
        else:
            raise("Invalid direction! Can not use proper embedding!")

        # Embed batch and noise_batch
        batch_size = lengths.size(0)
        
        start_ids = np.array([START_TOKEN for i in range(batch_size)])
        start_ids = torch.from_numpy(start_ids).long().to(self.device)
        embeded_in_ids = in_embed_layer(src_batch) #batch_size * l * embed_size

        #If beam_size == None, (batch_size ,embed_size)
        #If beam_size != None, (batch_size*beam_size, embed_size)
        dec_input_vec = ref_embed_layer(start_ids) 

        # Encode
        enc_outputs, h_n, c_n = self.encoder(embeded_in_ids, lengths)

        # Decode and calculate output probability.
        h_t = h_n
        c_t = c_n
        translations = []

        def decode_one_step(dec_input_vec, h_t, c_t, enc_outputs, in_mask):
            dec_output, h_t, c_t = self.decoder(dec_input_vec, h_t, c_t)
            context_vec = self.attention(dec_output, enc_outputs, in_mask)
            attention_vec = F.tanh(self.fc(
                    torch.cat([context_vec, dec_output], dim=1)
                ))
            probs = F.softmax(self.output_layer(attention_vec), dim=1)
            return probs, h_t, c_t

        #Above this line is shared by greedy mode and beam_search mode.
        ##############################################################

        if beam_size == None:
            for step in range(self.max_len):
                probs, h_t, c_t = decode_one_step(
                        dec_input_vec, h_t, c_t, enc_outputs, mask
                    )
                _, indices = torch.max(probs, dim=1)

                dec_input_vec = ref_embed_layer(indices)
                translations.append(
                        ref_vocab.ids2sentence(indices.cpu().numpy().tolist())
                        )

            # Post process translated batch
            translated_lengths = lengths.cpu().numpy().tolist()
            translated_lengths = [
                min(
                    int(l*self.max_ratio), self.max_len
                ) for l in translated_lengths
            ]
            not_finished = set(range(batch_size))
            translated_batch = [[] for i in range(batch_size)]

            for i in range(self.max_len):
                for idx, w in enumerate(translations[i]):
                    if idx in not_finished:
                        if w == '</s>':
                            not_finished.remove(idx)
                        elif w != '<s>' and w != '<pad>':
                            translated_batch[idx].append(w)

                        if len(translated_batch[idx]) == translated_lengths[idx]:
                            not_finished.remove(idx)
        else:
            # Initialize beam search
            # 1.Initialize hypothesis
            probs, h_t, c_t = decode_one_step(
                        dec_input_vec, h_t, c_t, enc_outputs, mask
                    )
            scores, indices = torch.topk(probs, beam_size, dim=1)
            start_scores = scores.cpu().detach().numpy().tolist()
            start_words = ref_vocab.ids_batch2sentence(
                    indices.cpu().numpy().tolist()
                )
            hyps = Hypothesis(beam_size, batch_size, start_words, start_scores,
                              h_t, c_t)

            # 2. Initialize decoder input.
            dec_input_vec = ref_embed_layer(
                        hyps.get_prev_words(ref_vocab).to(self.device)
                        )

            # 3.Initialize maxum translation length.
            translation_max_len = lengths.cpu().numpy().tolist()
            translation_max_len = [min(
                        int(self.max_ratio*translation_max_len[i]),
                        self.max_len
                    ) for i in range(batch_size)]
            for step in range(self.max_len):
                h_t, c_t = hyps.get_hidden()

                # split dec_input_vec
                h_t_list = torch.split(h_t, batch_size, 0)
                c_t_list = torch.split(c_t, batch_size, 0)
                dec_input_list = torch.split(dec_input_vec, batch_size, 0)

                h_tp1_list = []
                c_tp1_list = []
                scores_list = []
                indices_list = []
                for beam_idx in range(beam_size):
                    # get probs for each beam
                    probs, h_tp1, c_tp1 = decode_one_step(
                            dec_input_list[beam_idx],
                            h_t_list[beam_idx].permute(1, 0, 2),
                            c_t_list[beam_idx].permute(1, 0, 2),
                            enc_outputs, mask
                        )
                    scores, indices = torch.topk(probs, beam_size, dim=1)
                    scores_list.append(scores.detach())
                    indices_list.append(indices)
                    h_tp1_list.append(h_tp1.permute(1, 0, 2))
                    c_tp1_list.append(c_tp1.permute(1, 0, 2))

                # concat scores, indices and states
                scores = torch.cat(scores_list, dim=0)
                indices = torch.cat(indices_list, dim=0)
                h_tp1 = torch.cat(h_tp1_list, dim=0)
                c_tp1 = torch.cat(c_tp1_list, dim=0)
                new_entries = []

                for i_batch in range(batch_size):
                    new_entries += hyps.sort_and_choose(
                        hyps.get_candidates(
                            i_batch, scores, indices, h_tp1, c_tp1, ref_vocab,
                            translation_max_len[i_batch]
                            )
                        )
                hyps.extend(new_entries)

                dec_input_vec = ref_embed_layer(
                        hyps.get_prev_words(ref_vocab).to(self.device)
                        )

            #Post process
            translated_batch = []
            for i_batch in range(batch_size):
                l = hyps.collect(i_batch)
                l = sorted(l, key=lambda item:-sum(item[0]) / len(item[0]))
                translated_batch.append(
                    [w for w in l[0][1] if w not in ['<pad>', '<s>', '</s>']]
                )

        #Below this line is shared between greedy and beam search
        #########################################################

        #Get lengths and mask corresponding to translated_batch
        translated_lengths = [len(s) for s in translated_batch]
        translated_mask = [torch.cat(
                [torch.ones(l), torch.zeros(self.max_len - l)], dim=0
            ) for l in translated_lengths]
        translated_mask = torch.stack(translated_mask, dim=0)
        translated_lengths = torch.from_numpy(
                np.array(translated_lengths)
            ).long()

        #Transfer all return variables to current network's device
        translated_batch = ref_vocab.sentence_batch2ids(translated_batch)
        translated_batch = pad_ids_batch(translated_batch, self.max_len)
        translated_batch = np.asarray(translated_batch)
        translated_batch = torch.from_numpy(
                    translated_batch
                ).long()
        translated_mask = translated_mask
        translated_lengths = translated_lengths

        return translated_batch, translated_mask, translated_lengths
