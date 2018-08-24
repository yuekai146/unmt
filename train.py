from collections import OrderedDict
from data import *
from loss import *
from model import *
from tensorboardX import SummaryWriter
from translator import *
from utils import *
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim

import pdb


def get_optimizer(net, lr, method='adam', betas=None):
    """
    Define an optimizer.
    Input:
        net: A Seq2seq_Model or discriminator instance.
        method: A string. Which optmization algorithm to use.
        lr: learning_rate
        beta_1, beta_2: Adam parameters.
    output:
        optimizer: A torch.optim instance.
    """
    if method == 'adam':
        if betas == None:
            optimizer = optim.Adam(net.parameters(), lr=lr)
        else:
            assert len(betas) == 2
            optimizer = optim.Adam(net.parameters(), lr=lr, betas=betas)

    return optimizer


def store_state(step, seq2seq, seq2seq_optimizer, discriminator,
                dis_optimizer, checkpoint_path):
    """
    Input:
        step: Training has been performed for step batches.
        seq2seq: Sequence to sequence model to be stored.
        seq2seq_optimizer: optimizer along with seq2seq.
        discriminator: Discriminator to be stored.
        dis_optimizer: optimizer along with discriminator.
        checkpoint_path: Where to store optimizer and net.
    """
    state_dict = {
            "step":step,
            "seq2seq_state_dict":seq2seq.state_dict(),
            "seq2seq_optimizer_state_dict":seq2seq_optimizer.state_dict(),
            "discriminator_state_dict":discriminator.state_dict(),
            "dis_optimizer_state_dict":dis_optimizer.state_dict()
            }
    torch.save(state_dict,
               checkpoint_path + 'checkpoint_' + str(step) + '.pth')


def load_state(checkpoint_path, net, optimizer, which='seq2seq'):
    """
    Input:
        checkpoint_path: Where to load state_dict
        net: A Seq2seq_Model or discriminator instance.
        optimizer: Used to optimize net.
        which: A string 'seq2seq' or 'dis'
    """
    state_dict = torch.load(checkpoint_path)
    step = state_dict["step"]
    new_params = OrderedDict()

    if which == 'seq2seq':
        optimizer.load_state_dict(state_dict['seq2seq_optimizer_state_dict'])
        to_load = state_dict['seq2seq_state_dict']
    elif which == 'dis':
        optimizer.load_state_dict(state_dict['dis_optimizer_state_dict'])
        to_load = state_dict['discriminator_state_dict']

    for key, val in to_load.items():
        new_params[key] = val

    net.load_state_dict(new_params)

    return step, net, optimizer


def calc_auto_encode(batch, device, direction, seq2seq_model):
    '''
    Inputs:
        batch: A dict sampled from src_dataloader or trg_dataloader
        device: torch.device("cpu") or torch.device("cuda")
        direction; 'src2src' or 'trg2trg'
        seq2seq_model: A Seq2seq_Model instance
    Return:
        l_auto: Auto encode loss for specified direction.
    '''
    in_ids = batch["noise_ids"]
    ref_ids = batch["ids"]
    in_mask = batch["noise_mask"]
    ref_mask = batch["mask"]
    in_lengths = batch["noise_sentence_len"]
    ref_lengths = batch["sentence_len"]

    in_ids, in_mask, in_lengths, ref_ids, ref_mask, ref_lengths  = \
            sort_align_batch(
                in_ids, in_mask, in_lengths, ref_ids, ref_mask,
                ref_lengths
            )
    ref_ids, ref_mask, ref_lengths = add_eos(
            ref_ids, ref_mask, ref_lengths
            )

    in_ids = in_ids.to(device)
    ref_ids = ref_ids.to(device)
    in_mask = in_mask.to(device)
    ref_mask = ref_mask.to(device)

    log_probs_auto, enc_outputs = seq2seq_model(
            in_ids, ref_ids, in_mask, in_lengths, direction
            )

    loss_auto = l_auto(log_probs_auto, ref_ids, ref_mask)

    return loss_auto, enc_outputs, in_mask


def calc_cross_domain(src_batch, direction, translator, src_vocab, trg_vocab,
                      seq2seq_model, args, tranlator_type='WBW'):
    ids = src_batch['ids']
    lengths = src_batch["sentence_len"]
    mask = src_batch['mask']
    device = seq2seq_model.device

    if tranlator_type == 'WBW':
        # For WBW translator
        ids_ = ids.cpu().numpy().tolist()
        lengths_ = lengths.numpy().tolist()
        batch_size = len(lengths)

        sentence_batch = [
                ids_[i][:lengths_[i]] for i in range(batch_size)
                ]

        sentence_batch = src_vocab.ids_batch2sentence(sentence_batch)

        src_translated_batch, _, _ = translator(sentence_batch, direction)

        in_ids = []
        for s in src_translated_batch:
            s_ids = []
            for w in s:
                try:
                    s_ids.append(trg_vocab.word2id[w])
                except KeyError:
                    s_ids.append(UNK)
            in_ids.append(s_ids)

    elif tranlator_type == 'NMT':
        # For NMT translator
        ids, mask, lengths, _ = sort_inputs(
                ids, mask, lengths
            )
        ids_, mask_, lengths_ = \
                ids.to(device), mask.to(device), lengths

        src_translated_ids, _, src_translated_lengths = translator(
                ids_, mask_, lengths_,
                direction
                )

        in_ids = src_translated_ids.cpu().numpy().tolist()
        src_translated_lengths = src_translated_lengths.numpy().tolist()
        batch_size = len(src_translated_lengths)
        
        in_ids = [
                in_ids[i][:src_translated_lengths[i]] for i in range(batch_size)
                ]
    else:
        raise("Invalid tranlation type! Must be 'WBW' or 'NMT'.")


    in_ids, in_mask, in_lengths = batch_add_noise(
            in_ids, args.drop_prob, args.k
            )
    
    in_ids, in_mask, in_lengths, ref_ids, ref_mask, ref_lengths = \
            sort_align_batch(
                    in_ids, in_mask, in_lengths,
                    ids, mask, lengths
                    )

    ref_ids, ref_mask, ref_lengths = add_eos(ref_ids, ref_mask, ref_lengths)

    if direction == 'src2trg':
        reverse_direction = 'trg2src'
    elif direction == 'trg2src':
        reverse_direction = 'src2trg'

    log_probs, enc_outputs = seq2seq_model(
            in_ids.to(device), ref_ids.to(device), in_mask.to(device),
            in_lengths, reverse_direction
            )
    ref_mask = ref_mask.to(device)
    loss_cd = l_cd(log_probs, ref_ids, ref_mask)

    return loss_cd, enc_outputs, in_mask.to(device)


def build_seq2seq_components(args):
    src_embed_layer = Embedding(
                args.embed_size, args.vocab_size, NUM_OF_SPECIAL_TOKENS,
                args.src_embedding_matrix
                 )

    trg_embed_layer = Embedding(
            args.embed_size, args.vocab_size, NUM_OF_SPECIAL_TOKENS,
            args.trg_embedding_matrix
            )

    encoder = Encoder(
            args.embed_size, args.enc_hidden_size, args.enc_num_layers,
            bias=args.dis_enc_bias
            )

    decoder = Decoder(
            args.enc_hidden_size, args.enc_hidden_size,
            args.dec_num_layers, bias=args.dis_dec_bias
            )

    attention = Attention(args.enc_hidden_size)

    fc = nn.Linear(2*args.enc_hidden_size, args.enc_hidden_size)

    output_layer = nn.Linear(
            args.enc_hidden_size, NUM_OF_SPECIAL_TOKENS+args.vocab_size
            )

    return src_embed_layer, trg_embed_layer, encoder, decoder,\
            attention, fc, output_layer


def update_translator(seq2seq_model, args, src_vocab, trg_vocab,
                      BT_translator=None):
    if BT_translator == None:
        src_embed_layer, trg_embed_layer, encoder, decoder,\
        attention, fc, output_layer = build_seq2seq_components(args)
        
        BT_translator = NMT_Translator(
                    src_embed_layer, trg_embed_layer, encoder, decoder,
                    attention, fc, output_layer, args.max_len,
                    args.max_ratio, src_vocab, trg_vocab,
                    seq2seq_model.device
                )

        if args.cuda:
            BT_translator = BT_translator.cuda()

        if args.local_rank >= 0:
            BT_translator = nn.parallel.distributed.DistributedDataParallel(
                    BT_translator, device_ids=range(args.num_gpus)
                    )

    new_params = seq2seq_model.src_embedding.state_dict()
    BT_translator.src_embedding.load_state_dict(new_params)

    new_params = seq2seq_model.trg_embedding.state_dict()
    BT_translator.trg_embedding.load_state_dict(new_params)

    new_params = seq2seq_model.encoder.state_dict()
    BT_translator.encoder.load_state_dict(new_params)

    new_params = seq2seq_model.decoder.state_dict()
    BT_translator.decoder.load_state_dict(new_params)

    new_params = seq2seq_model.attention.state_dict()
    BT_translator.attention.load_state_dict(new_params)

    new_params = seq2seq_model.fc.state_dict()
    BT_translator.fc.load_state_dict(new_params)

    new_params = seq2seq_model.output_layer.state_dict()
    BT_translator.output_layer.load_state_dict(new_params)

    return BT_translator


def main():
    parser = argparse.ArgumentParser(
            description="Unsupervised NMT training hyper parameters."
            )

    # Arguments related to data.
    parser.add_argument(
            '--src_embeddings', help='path to src language embeddings'
            )
    parser.add_argument(
            '--trg_embeddings', help='path to trg language embeddings'
            )
    parser.add_argument(
            '--src_dict', help='path to src dictionary'
            )
    parser.add_argument(
            '--trg_dict', help='path to trg dictionary'
            )
    parser.add_argument(
            '--src_corpus', help='path to src training corpus'
            )
    parser.add_argument(
            '--trg_corpus', help='path to trg training corpus'
            )
    parser.add_argument('--embed_size', type=int, default=0,
            help='embedding size of word vector'
            )
    parser.add_argument(
            '--vocab_size', type=int, default=0, help='vocabulary size'
            )
    parser.add_argument(
            '--drop_prob', type=float, default=0.1,
            help='The probability of drop a word in input sequence'
            )
    parser.add_argument(
            '-k', type=int, default=3,
            help='Maximum permutation length'
            )
    parser.add_argument(
            '--max_len', type=int, default=50,
            help='Maximum sentence length'
            )

    # Arguments related to network architecture.
    # Encoder
    parser.add_argument(
            '--enc_hidden_size', type=int, default=300,
            help='Encoder hidden size'
            )
    parser.add_argument(
            '--enc_num_layers', type=int, default=3,
            help='Number of Encoder layers'
            )
    parser.add_argument(
            '--dis_enc_bias', action='store_false',
            help='Whether to use bias in Encoder'
            )
    # Decoder
    parser.add_argument(
            '--dec_num_layers', type=int, default=3,
            help='Number of Decoder layers'
            )
    parser.add_argument(
            '--dis_dec_bias', action='store_false',
            help='Whether to use bias in Decoder'
            )
    # Discriminator
    parser.add_argument(
            '--dis_num_layers', type=int, default=3,
            help='Number of discriminator layers'
            )
    parser.add_argument(
            '--dis_hidden_size', type=int, default=1024,
            help='Discriminator hideen size'
            )

    # Arguments related to translator
    parser.add_argument(
            '--max_ratio', type=int, default=1.5,
            help='Maximum ratio of translated length divided by source length'
            )

    # Arguments related to training
    parser.add_argument(
            '--batch_size', type=int, default=32, help='batch size'
            )
    parser.add_argument(
            '--seq2seq_lr', type=float, default=3e-4,
            help='Adam learning rate for seq2seq model'
            )
    parser.add_argument(
            '--seq2seq_betas', default=None,
            help='Adam betas for seq2seq model'
            )
    parser.add_argument(
            '--dis_lr', type=float, default=5e-4,
            help='Adam learning rate for discriminator'
            )
    parser.add_argument(
            '--dis_betas', default=None,
            help='Adam betas for discriminator'
            )
    parser.add_argument(
            '--num_epochs', type=int, default=4,
            help='Number of epochs to train the network'
            )
    parser.add_argument(
            '--num_workers', type=int, default=4,
            help='How many threads used to load data'
            )

    # Arguments related to checkpoint and log
    parser.add_argument(
            '--checkpoint_path', type=str, default='./checkpoint/',
            help='path to store and resume model'
            )
    parser.add_argument(
            '--log_dir', type=str, default='./logs/',
            help='path to store logging information'
            )
    parser.add_argument(
            '--store_interval', type=int, default=1000,
            help='Store the chekpoint after how many batches of training'
            )
    parser.add_argument(
            '--max_store_num', type=int, default=10
            )
    parser.add_argument(
            '--log_interval', type=int, default=100,
            help='Print logging information after how many batches of training'
            )
    parser.add_argument(
            '--resume_seq2seq', default=None,
            help='If not None, where to resume seq2seq model state_dict'
            )
    parser.add_argument(
            '--resume_dis', default=None,
            help='If not None, where to resume discriminator state_dict'
            )

    # Arguments related to using gpus
    parser.add_argument(
            '--cuda', action='store_true',
            help='Whether to use gpu for training'
            )
    parser.add_argument(
            '--local_rank', type=int, default=-1,
            help='Local process rank when using data parallel in training'
            )
    parser.add_argument(
            '--num_gpus', type=int, default=1,
            help='Use how many gpus to train the network.'
            )

    # Arguments related to loss
    parser.add_argument(
            '--lambda_auto', type=float, default=1.0
            )
    parser.add_argument(
            '--lambda_cd', type=float, default=1.0
            )
    parser.add_argument(
            '--lambda_adv', type=float, default=1.0
            )
    parser.add_argument(
            '--smooth', type=float, default=0.8,
            )
    parser.add_argument(
            '--grad_norm', default=None
            )

    args = parser.parse_args()

    # Make checkpoint and log directory
    if os.path.exists(args.checkpoint_path) == False:
        os.mkdir(args.checkpoint_path)

    if os.path.exists(args.log_dir) == False:
        os.mkdir(args.log_dir)

    # Initialize Vocab and dataset instances for src and trg
    src_words, src_embedding_matrix = get_embedding(
            args.src_embeddings, args.embed_size, args.vocab_size
            )
    args.vocab_size, args.embed_size = src_embedding_matrix.size()
    args.vocab_size = args.vocab_size - NUM_OF_SPECIAL_TOKENS
    print("Source embedding matrix is of size {}".format(
            src_embedding_matrix.size()
            )
        )
    args.src_embedding_matrix = src_embedding_matrix

    src_vocab = Vocab(src_words, 'src', args.vocab_size)
    src_dataset = WMT_Dataset(
            args.src_corpus, 'src', src_vocab, max_len=args.max_len,
            drop_prob=args.drop_prob, k=args.k
            )
    src_dataloader = DataLoader(
            src_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers
            )

    trg_words, trg_embedding_matrix = get_embedding(
            args.trg_embeddings, args.embed_size, args.vocab_size
            )
    print("Target embedding matrix is of size {}".format(
            trg_embedding_matrix.size()
            )
        )
    args.trg_embedding_matrix = trg_embedding_matrix
    '''
    args.vocab_size, args.embed_size = trg_embedding_matrix.size()
    args.vocab_size = args.vocab_size - NUM_OF_SPECIAL_TOKENS
    '''

    trg_vocab = Vocab(trg_words, 'trg', args.vocab_size)
    trg_dataset = WMT_Dataset(
            args.trg_corpus, 'trg', trg_vocab, max_len=args.max_len,
            drop_prob=args.drop_prob, k=args.k
            )
    trg_dataloader = DataLoader(
            trg_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers
            )
    #Build model

    src_embed_layer, trg_embed_layer, encoder, decoder,\
    attention, fc, output_layer = build_seq2seq_components(args)
    
    discriminator = Discriminator(
            args.enc_hidden_size, args.dis_hidden_size, args.dis_num_layers
            )
    
    if args.cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    seq2seq_model = Seq2seq_Model(
            src_embed_layer, trg_embed_layer, encoder, decoder,
            attention, args.embed_size, args.enc_hidden_size,
            args.vocab_size, NUM_OF_SPECIAL_TOKENS, device
            )

    # Whether use cuda, use parallel or resume model.
    # Build optimizer
    if args.cuda:
        seq2seq_model = seq2seq_model.cuda()
        discriminator = discriminator.cuda()

    if args.local_rank >= 0:
        torch.distributed.init_process_group(
                backend='nccl', rank=args.local_rank
                )
        seq2seq_model = nn.parallel.distributed.DistributedDataParallel(
                seq2seq_model, device_ids=range(args.num_gpus)
                )
        discriminator = nn.parallel.distributed.DistributedDataParallel(
                discriminator, device_ids=range(args.num_gpus)
                )
    
    print("Build seq2seq model and discriminator finished!")
    print(seq2seq_model)
    print(discriminator)

    seq2seq_optimizer = get_optimizer(
            seq2seq_model, args.seq2seq_lr, method='adam',
            betas=args.seq2seq_betas
            )
    dis_optimizer = get_optimizer(
            discriminator, args.dis_lr, method='adam',
            betas=args.dis_betas
            )

    print("Build optimizers finished!")
    print(seq2seq_optimizer)
    print(dis_optimizer)
    
    pretrained_step = 0
    if args.resume_seq2seq:
        pretrained_step, seq2seq_model, seq2seq_optimizer = \
                load_state(
                    args.resume_seq2seq, seq2seq_model, seq2seq_optimizer
                )
        print("Resume seq2seq model from {}".format(args.resume_seq2seq))

    if args.resume_dis:
        _, discriminator, dis_optimizer = \
                load_state(
                    args.resume_dis, discriminator, dis_optimizer, 'dis'
                )
        print("Resume discriminator from {}".format(args.resume_dis))

    # Initialize log writer
    if args.local_rank >= 0:
        if args.local_rank == 0:
            log_writer = SummaryWriter(args.log_dir)
    else:
        log_writer = SummaryWriter(args.log_dir)

    # Initialize some intermediate variable.
    num_batches_trained = 0
    total_loss = 0
    avg_loss = 0
    num_batches = len(src_dataset) / args.batch_size
    num_epochs_trained = int(pretrained_step / num_batches)

    #This code block is for test purpose
    #num_epochs_trained = 1

    # Initialize or reconstruct translator for back translation.
    if num_epochs_trained > 0:
        BT_translator = update_translator(
                seq2seq_model, args,
                src_vocab, trg_vocab
                )
    else:
        BT_translator = WBW_Translator(args.src_dict, args.trg_dict)

    for epoch in range(num_epochs_trained, args.num_epochs):
        for i_batch, (src_batch, trg_batch) in enumerate(zip(
                src_dataloader, trg_dataloader
                )):
            # Perform source and target language auto encoding
            l_auto_src, src_enc_outputs_auto, src_enc_mask_auto = \
                    calc_auto_encode(
                        src_batch, device, 'src2src', seq2seq_model
                    )
            
            l_auto_trg, trg_enc_outputs_auto, trg_enc_mask_auto = \
                    calc_auto_encode(
                        trg_batch, device, 'trg2trg', seq2seq_model
                    )

            # Perform source and target language back translation
            if epoch == 0:
                l_cd_src, src_enc_outputs_cd, src_enc_mask_cd = \
                        calc_cross_domain(
                            src_batch, 'src2trg', BT_translator, src_vocab, 
                            trg_vocab, seq2seq_model, args, 'WBW'
                        )

                l_cd_trg, trg_enc_outputs_cd, trg_enc_mask_cd = \
                        calc_cross_domain(
                            trg_batch, 'trg2src', BT_translator, trg_vocab,
                            src_vocab, seq2seq_model, args, 'WBW'
                        )
            else:
                l_cd_src, src_enc_outputs_cd, src_enc_mask_cd = \
                        calc_cross_domain(
                            src_batch, 'src2trg', BT_translator, src_vocab,
                            trg_vocab, seq2seq_model, args, 'NMT'
                        )

                l_cd_trg, trg_enc_outputs_cd, trg_enc_mask_cd = \
                        calc_cross_domain(
                            trg_batch, 'trg2src', BT_translator, trg_vocab,
                            src_vocab, seq2seq_model, args, 'NMT'
                        )

            #This code block is for testing
            #l_ = torch.sum(trg_enc_outputs_auto)
            #l_.backward()
            ##############################

            l_adv_total = 0.25 * (
                    l_adv(
                        discriminator(src_enc_outputs_auto),
                        src_enc_mask_auto, 'src'
                        ) +\
                    l_adv(
                        discriminator(trg_enc_outputs_auto),
                        trg_enc_mask_auto, 'trg'
                        ) +\
                    l_adv(
                        discriminator(src_enc_outputs_cd),
                        src_enc_mask_cd, 'src'
                        ) +\
                    l_adv(
                        discriminator(trg_enc_outputs_cd),
                        trg_enc_mask_cd, 'trg'
                        )
                    )

            l_dis_total = 0.25 * (
                    l_dis(
                        discriminator(src_enc_outputs_auto.detach()),
                        src_enc_mask_auto, args.smooth, 'src'
                        ) +\
                    l_dis(
                        discriminator(trg_enc_outputs_auto.detach()),
                        trg_enc_mask_auto, args.smooth, 'trg'
                        ) +\
                    l_dis(
                        discriminator(src_enc_outputs_cd.detach()),
                        src_enc_mask_cd, args.smooth, 'src'
                        ) +\
                    l_dis(
                        discriminator(trg_enc_outputs_cd.detach()),
                        trg_enc_mask_cd, args.smooth, 'trg'
                        )
                    )

            l_seq2seq_model = args.lambda_auto*0.5*(l_auto_src + l_auto_trg) +\
                              args.lambda_cd*0.5*(l_cd_src + l_cd_trg) +\
                              args.lambda_adv*l_adv_total
            
            seq2seq_optimizer.zero_grad()
            if args.grad_norm is not None:
                nn.utils.clip_grad_norm_(
                        seq2seq_model.parameters(), args.grad_norm
                        )
            l_seq2seq_model.backward()
            seq2seq_optimizer.step()


            dis_optimizer.zero_grad()
            if args.grad_norm is not None:
                nn.utils.clip_grad_norm_(
                        discriminator.parameters(), args.grad_norm
                        )
            l_dis_total.backward()
            dis_optimizer.step()


            num_batches_trained = int(epoch * num_batches + i_batch)
            if num_batches_trained % args.log_interval == 0:
                log_writer.add_scalar(
                        'losses/l_auto_src', l_auto_src.item(),
                        num_batches_trained
                        )
                log_writer.add_scalar(
                        'losses/l_auto_trg', l_auto_trg.item(),
                        num_batches_trained
                        )
                log_writer.add_scalar(
                        'losses/l_cd_src', l_cd_src.item(),
                        num_batches_trained
                        )
                log_writer.add_scalar(
                        'losses/l_cd_trg', l_cd_trg.item(),
                        num_batches_trained
                        )
                log_writer.add_scalar(
                        'losses/l_dis_total', l_dis_total.item(),
                        num_batches_trained
                        )
                log_writer.add_scalar(
                        'losses/l_adv_total', l_adv_total.item(),
                        num_batches_trained
                        )
                log_writer.add_scalar(
                        'losses/l_seq2seq_model', l_seq2seq_model.item(),
                        num_batches_trained
                        )


            if num_batches_trained % args.store_interval == 0:
                num_stored = len(os.listdir(args.checkpoint_path))
                if num_stored == args.max_store_num:
                    # Find the checkpoint with least batches trained.
                    temp = []
                    for checkpoint in os.listdir(args.checkpoint_path):
                        temp.append(
                            int(checkpoint.split('.')[0][11:])
                                )
                    minimum = min(temp)
                    os.remove(
                            args.checkpoint_path + 'checkpoint_' + \
                            str(minimum) + '.pth'
                            )

                store_state(
                        pretrained_step+num_batches_trained,
                        seq2seq_model, seq2seq_optimizer,
                        discriminator, dis_optimizer,
                        args.checkpoint_path
                        )

            print("Batch {} finished!".format(num_batches_trained))

        if epoch == 0:
            BT_translator = update_translator(
                    seq2seq_model, args, src_vocab, trg_vocab
                    )
        else:
            BT_translator = update_translator(
                    seq2seq_model, args, src_vocab, trg_vocab,
                    BT_translator
                    )


main()
