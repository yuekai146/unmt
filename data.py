from torch.utils.data import Dataset, DataLoader
from utils import add_noise
import numpy as np
import torch


PAD = 0
UNK = 1
START_TOKEN = 2
STOP_TOKEN = 3
NUM_OF_SPECIAL_TOKENS = 4


class Vocab(object):

    def __init__(self, words, language, vocab_size=50000):
        '''
        words: A list containing all words.
        language: A string. 'en', 'de' or 'fr'
        vocab_size: The size of vocab used in translation model.
        '''

        # Initialize attributes.
        self.language = language
        self.vocab_size = vocab_size
        self.id2word = ['<pad>', '<unk>', '<s>', '</s>']
        self.word2id = {'<pad>':0, '<unk>':1, '<s>':2, '</s>':3}
        num = NUM_OF_SPECIAL_TOKENS

        # Build vocab.
        assert len(words) >= vocab_size
        for i in range(vocab_size):
            self.id2word.append(words[i])
            self.word2id[words[i]] = num
            num += 1

    def sentence2ids(self, sentence, tokenized=False, sos=False, eos=False):
        # sentence: A string. Each word is separated by ' '.
        # return ids: A list of integers. Each int is the corresponding word id.
        if not tokenized:
            l = sentence.strip().split()
        else:
            l = sentence
        ids = []

        if sos:
            ids = [START_TOKEN]

        for w in l:
            try:
                w_id = self.word2id[w]
            except KeyError:
                w_id = UNK
            ids.append(w_id)

        if eos:
            ids.append(STOP_TOKEN)

        return ids

    def ids2sentence(self, ids):
        # ids: A list of integers.
        # return sentence: A list of words.
        sentence = []
        for w_id in ids:
            assert isinstance(w_id, int)
            assert w_id < self.vocab_size + NUM_OF_SPECIAL_TOKENS
            sentence.append(self.id2word[w_id])
        return sentence

    def sentence_batch2ids(self, sentence_batch, sos=False, eos=False):
        '''
        sentence_batch: A list of length batch_size
                        Each item in the list is also a list.
                        ['w1', 'w2', ...]
        Return:
            ids_batch: A list of length batch_size
                       Each item in the list is [id1, id2, ...]
        '''
        ids_batch = []

        for sent in sentence_batch:
            ids_batch.append(self.sentence2ids(sent, True, sos, eos))

        return ids_batch

    def ids_batch2sentence(self, ids_batch):
        '''
        ids_batch: A list of length batch_size
                   Each item in the list is [id1, id2, ...]
        Return:
            sentence_batch: A list of length batch_size
                            Each item in the list is also a list.
                            ['w1', 'w2', ...]
        '''
        sentence_batch = []
        for ids in ids_batch:
            sentence_batch.append(self.ids2sentence(ids))

        return sentence_batch


class WMT_Dataset(Dataset):

    def __init__(self, corpus_path, language, vocab, sos=False, eos=False,
                 max_len=50, drop_prob=0.1, k=3):
        '''
        corpus_path: A string. 
        language: A string. 'en', 'de' or 'fr'.
        vocab: A Vocab instance for converting sentence to ids.
        max_len: An integer, maximium length of a sentence, default 50.
        drop_prob: A float. The probability of dropping a word in a sentence.
        k: The maximium permutation length.
        '''
        f = open(corpus_path, 'r')
        self.corpus = f.readlines()
        f.close()

        self.language = language
        self.vocab = vocab
        self.sos= sos
        self.eos = eos
        self.max_len = max_len

        self.drop_prob = drop_prob
        self.k = k

    def _pad_ids(self, ids, max_len):
        padded_ids = [0 for x in range(max_len)]
        for i, idx in enumerate(ids):
            padded_ids[i] = idx

            if i == max_len - 1:
                break
                
        return padded_ids

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, idx):
        '''
        Return:
            ids: LongTensor(batch_size * max_len)
            sentence_len: Tensor (batch_size)
            mask: FloatTensor(batch_size * max_len)

            noise_ids: LongTensor(batch_size * max_len)
            noise_sentence_len: Tensor (batch_size)
            noise_mask: FloatTensor(batch_size * max_len)

        '''
        sentence = self.corpus[idx]
        ids = self.vocab.sentence2ids(sentence, sos=self.sos, eos=self.eos)
        sentence_len = len(ids)

        max_len = self.max_len

        if self.sos:
            max_len += 1
        if self.eos:
            max_len += 1

        ids = self._pad_ids(ids, max_len)
        mask = [1 if x < sentence_len else 0 for x in range(max_len)]

        noise_sentence = add_noise(sentence, self.drop_prob, self.k)
        noise_ids = self.vocab.sentence2ids(noise_sentence)
        noise_sentence_len = len(noise_ids)
        noise_ids = self._pad_ids(noise_ids, self.max_len)
        noise_mask = [
            1 if x < noise_sentence_len else 0 for x in range(self.max_len)
            ]

        ids = torch.from_numpy(np.array(ids)).long()
        mask = torch.from_numpy(np.array(mask)).float()

        noise_ids = torch.from_numpy(np.array(noise_ids)).long()
        noise_mask = torch.from_numpy(np.array(noise_mask)).float()

        ret = {}
        ret["ids"] = ids
        ret["sentence_len"] = sentence_len
        ret["mask"] = mask
        ret["noise_ids"] = noise_ids
        ret["noise_sentence_len"] = noise_sentence_len
        ret["noise_mask"] = noise_mask

        return ret
    

def get_embedding(embedding_path, embed_size=0, vocab_size=0):
    '''
    Return:
        words: A list containing all the words in the vocabulary.
        embedding_matrix: A numpy array of size (vocab_size * embed_dim)
    '''
    f = open(embedding_path, 'r')
    l = f.readline()
    num_words, dim = l.strip().split()
    num_words, dim = int(num_words), int(dim)

    if vocab_size > 0:
        num_words = vocab_size

    words = []
    embeds = []

    for i in range(num_words):
        l = f.readline()
        l_ = l.strip().split()
        w = l_[0]
        if embed_size > 0:
            assert isinstance(embed_size, int)
            embed = l_[1:(embed_size+1)]
        else:
            embed = l_[1:]
        embed = [float(x) for x in embed]

        words.append(w)
        embeds.append(embed)
    f.close()


    embed_dim = len(embeds[0])
    special_token_embeddings = torch.zeros(NUM_OF_SPECIAL_TOKENS, embed_dim)
    embedding_matrix = torch.from_numpy(np.array(embeds)).float()
    embedding_matrix = torch.cat(
            [special_token_embeddings, embedding_matrix], dim=0
        )

    return words, embedding_matrix
