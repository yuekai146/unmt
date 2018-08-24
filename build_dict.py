import pickle


def merge_text(src_path, trg_path, merge_path):
    f_src = open(src_path, 'r')
    f_trg = open(trg_path, 'r')
    f_merge = open(merge_path, 'a')

    src = f_src.readlines()
    trg = f_trg.readlines()

    for l_src, l_trg in zip(src, trg):
        l = l_src.strip() + ' ||| ' + l_trg
        f_merge.writelines(l)

    f_src.close()
    f_trg.close()
    f_merge.close()


def build_vocab(corpus_path, vocab_path, size=100000):
    f = open(corpus_path, 'r')
    lines = f.readlines()
    vocab = {}

    for l in lines:
        for w in l.strip().split():
            vocab[w] = 0

    
    for l in lines:
        for w in l.strip().split():
            vocab[w] += 1
    f.close()

    vocab = sorted(vocab.items(), key=lambda item: -item[1])
    size = min(size, len(vocab))
    vocab = vocab[:size]
    f = open(vocab_path, 'a')

    for item in vocab:
        f.writelines(item[0] + '\n')


def init_dict(vocab_path, src_path, trg_path, align_path, output_path,
              reverse=False):
    vocab = {}
    f = open(vocab_path, 'r')
    lines = f.readlines()
    for w in lines:
        w = w.strip().split()[0]
        vocab[w] = {}

    f.close()
    del lines

    f_src = open(src_path, 'r')
    f_trg = open(trg_path, 'r')
    f_align = open(align_path, 'r')
    src = f_src.readlines()
    trg = f_trg.readlines()
    align = f_align.readlines()

    length = len(src)

    for i in range(length):
        l_src = src[i].strip().split()
        l_trg = trg[i].strip().split()
        l_align = align[i].strip().split()

        for m in l_align:
            src_idx, trg_idx = m.split('-')
            src_idx, trg_idx = int(src_idx), int(trg_idx)

            if reverse:
                src_idx, trg_idx = trg_idx, src_idx

            try:
                vocab[l_src[src_idx]][l_trg[trg_idx]] = 0
            except KeyError:
                pass

    f = open(output_path, 'wb')
    pickle.dump(vocab, f)

    f_src.close()
    f_trg.close()
    f_align.close()
    f.close()


def get_dict(src_path, trg_path, align_path, init_path, output_path,
             reverse=False):
    f_src = open(src_path, 'r')
    f_trg = open(trg_path, 'r')
    f_align = open(align_path, 'r')
    f_init = open(init_path, 'rb')

    src = f_src.readlines()
    trg = f_trg.readlines()
    align = f_align.readlines()
    init = pickle.load(f_init)

    length = len(src)

    for i in range(length):
        l_src = src[i].strip().split()
        l_trg = trg[i].strip().split()
        l_align = align[i].strip().split()

        for m in l_align:
            src_idx, trg_idx = m.split('-')
            src_idx, trg_idx = int(src_idx), int(trg_idx)

            if reverse:
                src_idx, trg_idx = trg_idx, src_idx
            
            try:
                init[l_src[src_idx]][l_trg[trg_idx]] += 1
            except KeyError:
                pass

    for key, val in init.items():
        sorted_val = sorted(val.items(), key=lambda item: -item[1])
        init[key] = sorted_val

    f = open(output_path, 'wb')
    pickle.dump(init, f)

    f_src.close()
    f_trg.close()
    f_align.close()
    f_init.close()
    f.close()


def get_overlap(src_dict_path, trg_dict_path, output_path):
    f = open(src_dict_path, 'rb')
    src_dict = pickle.load(f)
    f.close()

    f = open(trg_dict_path, 'rb')
    trg_dict = pickle.load(f)
    f.close()

    src_words = src_dict.keys()
    trg_words = trg_dict.keys()

    overlap_ = []
    for w_src in src_words:
        if w_src in trg_words:
            overlap_.append(w_src)

    overlap = []
    for w in overlap_:
        if not w.isalpha():
            overlap.append(w)

    f = open(output_path, 'wb')
    pickle.dump(overlap, f)
    f.close()


def post_process(dict_path, overlap_path, output_path):
    f = open(dict_path, 'rb')
    final_dict = pickle.load(f)
    f.close()

    f = open(overlap_path, 'rb')
    overlap = pickle.load(f)
    f.close()
    
    output = []
    for w in final_dict.keys():
        if w in overlap:
            output.append(w + ' ' + w)
        else:
            try:
                output.append(w + ' ' + final_dict[w][0][0])
            except IndexError:
                output.append(w + ' ' + w)

    f = open(output_path, 'a')
    for l in output:
        f.writelines(l + '\n')
    f.close()


def main():
    ROOT_PATH='/data/zhaoyuekai/iwslt14/'
    SRC_PATH=ROOT_PATH + 'final_train.en'
    TRG_PATH=ROOT_PATH + 'final_train.de'
    MERGE_PATH=ROOT_PATH + 'merged_text.txt'
    
    #merge_text(SRC_PATH, TRG_PATH, MERGE_PATH)
    CORPUS_PATH=ROOT_PATH + 'final_train.en'
    VOCAB_PATH=ROOT_PATH + 'vocab_large.en'
    ALIGN_PATH=ROOT_PATH + 'forward.align'
    OUTPUT_PATH=ROOT_PATH + 'dict_draft.en'

    #build_vocab(CORPUS_PATH, VOCAB_PATH, size=50000)
    #init_dict(VOCAB_PATH, SRC_PATH, TRG_PATH, ALIGN_PATH, OUTPUT_PATH, reverse=False)

    INIT_PATH=ROOT_PATH + 'dict_draft.en'
    OUTPUT_PATH=ROOT_PATH + 'final_dict.en'
    #get_dict(SRC_PATH, TRG_PATH, ALIGN_PATH, INIT_PATH, OUTPUT_PATH)


    CORPUS_PATH=ROOT_PATH + 'final_train.de'
    VOCAB_PATH=ROOT_PATH + 'vocab_large.de'
    ALIGN_PATH=ROOT_PATH + 'reverse.align'
    OUTPUT_PATH=ROOT_PATH + 'dict_draft.de'

    #build_vocab(CORPUS_PATH, VOCAB_PATH, size=50000)
    #init_dict(VOCAB_PATH, TRG_PATH, SRC_PATH, ALIGN_PATH, OUTPUT_PATH, reverse=True)

    INIT_PATH=ROOT_PATH + 'dict_draft.de'
    OUTPUT_PATH=ROOT_PATH + 'final_dict.de'
    #get_dict(TRG_PATH, SRC_PATH, ALIGN_PATH, INIT_PATH, OUTPUT_PATH, reverse=True)
    
    SRC_DICT_PATH=ROOT_PATH + 'dict_draft.en'
    TRG_DICT_PATH=ROOT_PATH + 'dict_draft.de'
    OUTPUT_PATH=ROOT_PATH + 'same_words'
    get_overlap(SRC_DICT_PATH, TRG_DICT_PATH, OUTPUT_PATH)
    
    DICT_PATH=ROOT_PATH + 'final_dict.en'
    OVERLAP_PATH=ROOT_PATH + 'same_words'
    OUTPUT_PATH=ROOT_PATH + 'en_dict.txt'
    post_process(DICT_PATH, OVERLAP_PATH, OUTPUT_PATH)


    DICT_PATH=ROOT_PATH + 'final_dict.de'
    OVERLAP_PATH=ROOT_PATH + 'same_words'
    OUTPUT_PATH=ROOT_PATH + 'de_dict.txt'
    post_process(DICT_PATH, OVERLAP_PATH, OUTPUT_PATH)



main()
