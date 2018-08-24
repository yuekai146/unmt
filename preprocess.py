import functools

def get_vocab(corpus_path, embedding_path, vocab_size=50000):
    # Build a full vocabulary containing all words in corpus
    f = open(corpus_path, 'r')
    lines = f.readlines()
    full_vocab = {}
    for l in lines:
        for w in l.strip().split():
            full_vocab[w.lower()] = 0

    for l in lines:
        for w in l.strip().split():
            full_vocab[w.lower()] += 1

    f.close()
    del lines

    # Build a vocabulary containing all words in embedding file
    embed_vocab = []
    f = open(embedding_path, 'r')
    lines = f.readlines()
    for l in lines:
        w = l.strip().split()[0]
        embed_vocab.append(w.lower())

    f.close()
    del lines

    # Find all overlapping words
    full_s = set(full_vocab.keys())
    embed_s = set(embed_vocab)
    overlap = full_s & embed_s

    print("Number of overlap words are {}".format(len(overlap)))

    #Find the most frequent overlapping words.
    if vocab_size > len(overlap):
        vocab_size = len(overlap)

    overlap_freq = {}
    for w in overlap:
        overlap_freq[w] = full_vocab[w]
    final_vocab = sorted(overlap_freq.items(), key=lambda item:-item[1])
    final_vocab = final_vocab[:vocab_size]
    
    return final_vocab

def store_vocab(vocab, path, embedding_size=300):
    f = open(path, 'a')
    f.writelines(str(len(vocab)) + ' ' + str(embedding_size) + "\n")

    for item in vocab:
        f.writelines(item[0] + "\n")


def get_embed_vec(vocab_path, embedding_path):
    # Build two lists containing all words and their embeddings
    f = open(embedding_path, 'r')
    l = f.readline()
    num_words, dim = l.strip().split()
    num_words, dim = int(num_words), int(dim)

    embeds = []
    embed_vocab = []

    for i in range(num_words):
        l = f.readline()
        w_with_embed = l.strip().split()
        w = w_with_embed[0]
        embed = w_with_embed[1:]
        embeds.append(embed)
        embed_vocab.append(w)
    f.close()

    lower_case_embed_vocab = [w.lower() for w in embed_vocab]

    # Get vocabulary
    f = open(vocab_path, 'r')
    l = f.readline()
    num_words, dim = l.strip().split()
    num_words, dim = int(num_words), int(dim)

    vocab = []
    for i in range(num_words):
        l = f.readline()
        w = l.strip().split()[0]
        vocab.append(w)
    f.close()

    # Find corresponding embedding vector
    embedding_list = []
    for w in vocab:
        index = lower_case_embed_vocab.index(w)
        embed = embeds[index]
        embedding_list.append(embed)

    print("All embedding vector needed got!")

    return vocab, embedding_list

def store_embedding_vec(vocab, embedding_list, path):
    f = open(path, 'a')
    num_words = len(vocab)
    dim = len(embedding_list[0])
    f.writelines(str(num_words) + ' ' + str(dim) + '\n')

    for w, embed in zip(vocab, embedding_list):
        l = [' ' + num for num in embed]
        l = functools.reduce(lambda a,b: a+b, l)
        l = w + l
        l = l + '\n'
        f.writelines(l)
    print("Store embedding vector complete!")

    f.close()

def throw_long_sents(src_path, trg_path, new_src_path, new_trg_path,
                     max_len=50, threshold=1.5):
    f = open(src_path, 'r')
    src = f.readlines()
    f.close()

    f = open(trg_path)
    trg = f.readlines()
    f.close()

    src_f = open(new_src_path, 'a')
    trg_f = open(new_trg_path, 'a')
    for src_sent, trg_sent in zip(src, trg):
        src_sent_ = src_sent.strip().split()
        trg_sent_ = trg_sent.strip().split()
        src_sent_ = [w.lower() for w in src_sent_]
        trg_sent_ = [w.lower() for w in trg_sent_]
        ratio = max(len(src_sent_) / len(trg_sent_), len(trg_sent_) / len(src_sent_))

        if (len(src_sent_) <= max_len) and (len(trg_sent_) <= max_len) and (ratio <= threshold):
            src_f.writelines(src_sent.lower())
            trg_f.writelines(trg_sent.lower())

    src_f.close()
    trg_f.close()


def main():
    corpus_path = '/data/zhaoyuekai/iwslt14/data/train.de-en.en'
    embedding_path = '/data/zhaoyuekai/unmt/others/crawl-300d-2M.vec'
    vocab_path = '/data/zhaoyuekai/iwslt14/vocab.en'
    store_embedding_path = '/data/zhaoyuekai/iwslt14/embeddings.en'

    src_path = '/data/zhaoyuekai/iwslt14/data/train.de-en.en'
    trg_path = '/data/zhaoyuekai/iwslt14/data/train.de-en.de'
    new_src_path = '/data/zhaoyuekai/iwslt14/final_train.en'
    new_trg_path = '/data/zhaoyuekai/iwslt14/final_train.de'

    #final_vocab = get_vocab(corpus_path, embedding_path, vocab_size=30000)
    #store_vocab(final_vocab, vocab_path, embedding_size=300)
    #final_vocab, embedding_list = get_embed_vec(vocab_path, embedding_path)
    #store_embedding_vec(final_vocab, embedding_list, store_embedding_path)


    corpus_path = '/data/zhaoyuekai/iwslt14/data/train.de-en.de'
    embedding_path = '/data/zhaoyuekai/unmt/others/cc.de.300.vec'
    vocab_path = '/data/zhaoyuekai/iwslt14/vocab.de'
    store_embedding_path = '/data/zhaoyuekai/iwslt14/embeddings.de'

    #final_vocab = get_vocab(corpus_path, embedding_path, vocab_size=30000)
    #store_vocab(final_vocab, vocab_path, embedding_size=300)
    #final_vocab, embedding_list = get_embed_vec(vocab_path, embedding_path)
    #store_embedding_vec(final_vocab, embedding_list, store_embedding_path)

    throw_long_sents(src_path, trg_path, new_src_path, new_trg_path,
                     max_len=50, threshold=1.5)


main()
