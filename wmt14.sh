ROOT_PATH=/data/zhaoyuekai/unmt
SRC_EMB=$ROOT_PATH/embeddings.en
TRG_EMB=$ROOT_PATH/embeddings.de
SRC_DICT=$ROOT_PATH/en_dict.txt
TRG_DICT=$ROOT_PATH/de_dict.txt
SRC_CORPUS=$ROOT_PATH/final_train.en
TRG_CORPUS=$ROOT_PATH/final_train.de

#python train.py --src_embeddings $SRC_EMB --trg_embeddings $TRG_EMB --src_dict $SRC_DICT --trg_dict $TRG_DICT --src_corpus $SRC_CORPUS --trg_corpus $TRG_CORPUS

#Following vars are for test purpose
MODEL_PATH=./checkpoint/checkpoint_0.pth

CUDA_VISIBLE_DEVICES=1 python train.py \
		--src_embeddings $SRC_EMB \
		--trg_embeddings $TRG_EMB \
		--src_dict $SRC_DICT \
		--trg_dict $TRG_DICT \
		--src_corpus $SRC_CORPUS \
		--trg_corpus $TRG_CORPUS \
		--batch_size 8\
		--cuda
		--resume_seq2seq $MODEL_PATH \
		--resume_dis $MODEL_PATH
