import numpy as np
import torch


START_TOKEN = 2
STOP_TOKEN = 3


def add_noise(sentence, drop_prob, k):
    """
    sentence: A string. Each word is separated by ' '.
    p: The probability fo dropping each word.
    k: Max permutation length.

    Return:
        A string. Permuted sentence, each word is separated 
        by ' ' with '\n' in the last
    """
    # Permute sentence
    sentence = sentence.strip().split()
    q = []
    for i in range(len(sentence)):
        q.append(i + np.random.uniform(0, k + 1))
    sorted_q = sorted(q)
    index = [q.index(x) for x in sorted_q]
    sent = [sentence[idx] for idx in index]

    # Drop words with probability p.
    index = []
    for i, w in enumerate(sent):
        prob = np.random.uniform()
        if prob >= drop_prob:
            index.append(i)
    sent = [sent[idx] + ' ' for idx in index]
    sent_ = ''
    for w in sent:
        sent_ += w

    sent_ = sent_[:-1] + '\n'

    return sent_


def batch_add_noise(batch, drop_prob, k):
    '''
    Inputs:
        batch: A list of length batch_size. Each element in it is a list.
               [id1, id2, ...]
        drop_prob: A float.
        k: An integer.
    Return:
        noise_batch: torch.LongTensor(batch_size, max_l)
                     max_l = max(noise_lengths)
        noise_mask: torch.FloatTensor(batch_size, max_l)
        noise_lengths: torch.LongTensor(batch_size)
    '''
    noise_batch = []
    for s in batch:
        q = []
        for i in range(len(s)):
            q.append(i + np.random.uniform(0, k + 1))
        sorted_q = sorted(q)
        index = [q.index(x) for x in sorted_q]
        s_noise = []
        for idx in index:
            prob = np.random.uniform()
            if prob >= drop_prob:
                s_noise.append(s[idx])
        noise_batch.append(s_noise)

    noise_lengths = [len(s) for s in noise_batch]
    max_l = max(noise_lengths)
    noise_batch = [torch.cat(
            [torch.from_numpy(np.array(s)).long(),
             torch.zeros(max_l-len(s)).long()]
        ) for s in noise_batch]
    noise_batch = torch.stack(noise_batch, dim=0).long()
    noise_mask = [torch.cat(
            [torch.ones(l), torch.zeros(max_l-l)]
        ) for l in noise_lengths]
    noise_mask = torch.stack(noise_mask, dim=0).float()
    noise_lengths = torch.from_numpy(np.array(noise_lengths)).long()

    return noise_batch, noise_mask, noise_lengths


def pad_ids(ids, max_len, sos=False, eos=False):
    if sos:
        max_len += 1

    if eos:
        max_len += 1

    padded_ids = [0 for x in range(max_len)]

    for i, idx in enumerate(ids):
        padded_ids[i] = idx

    return padded_ids

def pad_ids_batch(ids_batch, max_len, is_tensor=False, sos=False, eos=False):
    '''
    ids_batch: A list of length batch_size
               Each item in the list is [id1, id2, ...]
    max_len: An integer. The maximum sentence length.
    is_tensor: Bool value. If True, ids_batch is a LongTensor.

    Return:
        padded_ids_batch: A list of length batch_size
                          Each item in it is [id1, id2, ..., 0, 0]
                          Each item is of length max_len.
    '''
    if is_tensor:
        ids_batch = ids_batch.cpu().numpy()
        ids_batch = ids_batch.tolist()
    
    padded_ids_batch = []
    for ids in ids_batch:
        padded_ids_batch.append(pad_ids(ids, max_len, sos, eos))

    return padded_ids_batch


def sort_inputs(ids, mask, sentence_len):
    '''
    ids: LongTensor(batch_size * max_len)
    sentence_len: Tensor (batch_size)
    mask: FloatTensor(batch_size * max_len)

    Return:
        sorted_ids: LongTensor(batch_size * l) l=max(sentence_len)
        sorted_mask: mask corresponding to sorted_ids
        sorted_sentence_len: 
            A Tensor contains sentence length in decreasing order.
        indices: The indice in which sentence_len can be decreasing.
    '''
    l = torch.max(sentence_len)
    l = l.item()
    ids = ids[:, :l]
    mask = mask[:, :l]

    sorted_sentence_len, indices = torch.sort(sentence_len, descending=True)
    sorted_ids = ids[indices]
    sorted_mask = mask[indices]

    return sorted_ids, sorted_mask, sorted_sentence_len, indices


def sort_align_batch(in_ids, in_mask, in_sentence_len,
                     ref_ids, ref_mask, ref_sentence_len):
    '''
    in_ids, in_mask, in_sentence_len are the same input type and size as in
    function sort_inputs.

    ref_ids, ref_mask, ref_sentence_len: 
        Reference that needed to be aligned with inputs.

    Return:
        sorted_in_ids, sorted_in_mask, sorted_in_sentence_len, 
        aligned_ref_ids, aligned_mask, aligned_ref_sentence_len
    '''
    sorted_in_ids, sorted_in_mask, sorted_in_sentence_len, indices = \
            sort_inputs(
                in_ids, in_mask, in_sentence_len
            )
    aligned_ref_ids = ref_ids[indices]
    aligned_ref_mask = ref_mask[indices]
    aligned_ref_sentence_len = ref_sentence_len[indices]

    return sorted_in_ids, sorted_in_mask, sorted_in_sentence_len,\
           aligned_ref_ids, aligned_ref_mask, aligned_ref_sentence_len


def add_sos(ids, mask, lengths):
    '''
    ids: LongTensor(batch_size, l)
    mask: FloatTensor (batch_size, l)
    lengths: LongTensor (batch_size)
    '''
    batch_size = lengths.size(0)
    sos_col = [START_TOKEN for i in range(batch_size)]
    sos_col = torch.from_numpy(np.array(sos_col)).long()
    ids_ = torch.cat([sos_col.unsqueeze(-1), ids], dim=1)
    mask_ = torch.cat([torch.ones(batch_size).unsqueeze(-1), mask], dim=1)
    lengths_ = lengths + 1

    return ids_, mask_, lengths_.long()

def add_eos(ids, mask, lengths):
    '''
    ids: LongTensor(batch_size, l)
    mask: FloatTensor (batch_size, l)
    lengths: LongTensor (batch_size)
    '''
    batch_size = lengths.size(0)
    ids_ = torch.cat(
            [ids, torch.zeros(batch_size).unsqueeze(-1).long()], dim=1
            )
    mask_ = torch.cat([mask, torch.zeros(batch_size).unsqueeze(-1)], dim=1)
    r = torch.arange(batch_size).unsqueeze(-1).long()
    ids_[r, lengths.unsqueeze(-1)] = STOP_TOKEN
    mask_[r, lengths.unsqueeze(-1)] = 1.0
    lengths_ = lengths + 1

    return ids_.long(), mask_, lengths_.long()
