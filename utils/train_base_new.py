#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import shutil
import pickle
import os
from models.new_models import Shared_Encoder, Shared_Decoder,Shared_MT

def preprare_model(opt, dataset, vocab_dict, logger):
    np.random.seed(opt.seed)
    if opt.swemb_size:
        assert opt.emb_size >= opt.swemb_size

    if opt.dict_tgtlangs is None:
        opt.dict_tgtlangs = list(range(len(vocab_dict.V_size)))[1:] # lang = 4 --> [1, 2, 3], 2 =[1]

    class_number  = len(set(opt.lang_class))
    is_unsupervised = len(dataset.lines_id_mono[0]) > 0
    if not is_unsupervised:
        encoder = Shared_Encoder(opt.enc_dec_layer[0], opt.emb_size, opt.h_size, opt.dr_rate, vocab_dict)
    else:
        encoder = None
        assert class_number == 1, "For an unsupervised model, set lang_class as 0 for all languages"
    decoder = Shared_Decoder(2*class_number, opt.enc_dec_layer[1], opt.emb_size, opt.h_size, opt.dr_rate, vocab_dict, is_unsupervised)
    model = Shared_MT(encoder, decoder, vocab_dict, opt.emb_size,opt.swemb_size, int(opt.share_vocab), opt.subword, opt.dr_rate, opt.pretrained_emb)
    for name, param in model.named_parameters():
        if opt.pretrained_emb:
            if name in ["embedding_weight.emb_list.1.weight","embedding_weight.a","embedding_weight.b"]:
                continue
        param.data.uniform_(-0.1, 0.1)
    model.Set_lang_class(opt.lang_class)

    return model, dataset, vocab_dict

def generate_bacth_idx(lengths_para, lengths_mono, lengths_multi, batch_size, logger):
    #dataset: sorted by src length
    batch_idx_list = [[] for _ in range(2)]
    if len(lengths_para[0]) or len(lengths_multi[0]):
        if len(lengths_para[0]):
            N_para = len(lengths_para[0])
        elif len(lengths_multi[0]):
            N_para = len(lengths_multi[0][0])
        for i in range(0, N_para, batch_size):
            batch_idx = list(range(i, min(i + batch_size, N_para)))  # batch_size
            batch_idx_list[0].append(batch_idx)
        batch_idx_list[0] = np.array(batch_idx_list[0],dtype=object)
        logger.info("Number of cross-lingual mini-batches " + str(len(batch_idx_list[0])))
    if len(lengths_mono):
        N_mono = len(lengths_mono[0])
        for i in range(0, N_mono, batch_size):
            batch_idx = list(range(i, min(i + batch_size, N_mono)))  # batch_size
            batch_idx_list[1].append(batch_idx)
        batch_idx_list[1] = np.array(batch_idx_list[1],dtype=object)
        logger.info("Number of monolingual mini-batches " + str(len(batch_idx_list[1])))
    return batch_idx_list

def length_normalize(matrix):
    #matrix: V, emb_size
    norms = np.sqrt(np.sum(matrix ** 2, axis=1))
    norms[norms == 0] = 1
    matrix /= norms[:, np.newaxis]
    return matrix

def mean_center(matrix):
    # matrix: V, emb_size
    avg = np.mean(matrix, axis=0)
    matrix -= avg
    return matrix


def normalize(matrix, actions):
    for action in actions:
        print(action)
        if action == 'unit':
            matrix = length_normalize(matrix)
        elif action == 'center':
            matrix = mean_center(matrix)

    return matrix



def Save_mapped_emb(model, filename):
    model.eval()
    with torch.no_grad():
        for lang in range(model.lang_size):
            mapped_vec = model.embedding_weight.Mapped_Whole_Emb(lang)
            first_line = str(len(mapped_vec))+ " "+ str(model.emb_size)
            with open(filename + '.lang' + str(lang), "w") as f_new:
                f_new.write(first_line)
                for j in range(len(mapped_vec)):
                    out = model.embedding_weight.word_list[lang][j] + " " + " ".join(map(str, mapped_vec[j])) + "\n"
                    f_new.write(out)

def out_wordemb(id2vocab, emb_list, vocab = None, grad = False):
    if grad:
        Emb = emb_list
    else:
        Emb = emb_list.data.tolist()

    if vocab is None:
        vocab = id2vocab.keys()

    vocab2emb = {}
    for id in vocab:
        emb = Emb[id]  # demb
        vocab2emb[id2vocab[id]] = emb

    return vocab2emb

def New_Out_Wordemb(id2vocab, emb_list, vocab = None, grad = False):
    if grad:
        Emb = emb_list.weight
    else:
        Emb = emb_list.weight.data.tolist()

    if vocab is None:
        vocab = id2vocab.keys()

    vocab2emb = {}
    for id in vocab:
        emb = Emb[id]  # demb
        vocab2emb[id2vocab[id]] = emb

    return vocab2emb

def Out_Mapped_Wordemb(id2vocab, emb_list, W, vocab = None, grad = False):
    if grad:
        mm = torch.matmul
        Emb = emb_list.weight
        W_map = W.weight
    else:
        mm = np.matmul
        Emb = emb_list.weight.data.tolist()
        W_map = W.weight.data.tolist()


    if vocab is None:
        vocab = id2vocab.keys()

    vocab2emb = {}
    for id in vocab:
        emb = Emb[id] #demb
        vocab2emb[id2vocab[id]] = mm(W_map, emb)

    return vocab2emb

def save_emb(vocab2emb, N_dim, filename):
    vocab = list(vocab2emb.keys())
    first_line = str(len(vocab)) + " " + str(N_dim)
    with open(filename, "w", encoding="utf8") as word_output:
        word_output.write(first_line + "\n")
    for word in vocab:
        out = word  + " " +  " ".join(map(str, vocab2emb[word])) + "\n"
        with open(filename, "a", encoding="utf8") as word_output:
            word_output.write(out)


def PAD_Sentences(lengths, lines_id, batch_idx, PAD_id, BOS_id, EOS_id, ignore_idx):
    decoder_input_len = []
    fwd_input = []
    fwd_output = []
    bkw_input = []
    bkw_output = []

    for lang in range(len(lengths)):
        s_lengths = lengths[lang][batch_idx]
        max_length = max(s_lengths)
        padding_len = max_length - s_lengths

        BOS_lines_id= []
        lines_id_EOS = []
        BOS_lines_id_bkw = []
        lines_id_EOS_bkw = []

        for i, j in enumerate(batch_idx):
            line = lines_id[lang][j]
            line_output = [x-2 for x in line] #omit PAD (0) and MASK(1)
            BOS_lines_id.append([BOS_id]+line + [PAD_id] * padding_len[i])
            lines_id_EOS.append(line_output + [EOS_id-2] + [ignore_idx] *padding_len[i] )
            BOS_lines_id_bkw.append([EOS_id] + line[::-1] + [PAD_id] * padding_len[i])
            lines_id_EOS_bkw.append(line_output[::-1] + [BOS_id-2] + [ignore_idx] * padding_len[i])
        # s_lengths+=1 #EOS or BOS

        decoder_input_len.append(s_lengths+1)
        fwd_input.append(BOS_lines_id)
        bkw_input.append(BOS_lines_id_bkw)
        fwd_output.append(lines_id_EOS)
        bkw_output.append(lines_id_EOS_bkw)

    return decoder_input_len, [fwd_input,bkw_input] ,[fwd_output, bkw_output]

def PAD_Sentences_Source(lengths, lines_id, batch_idx, PAD_id):

    s_lengths = lengths[batch_idx]
    max_length = max(s_lengths)
    padding_len = max_length - s_lengths
    lines_id_list = []
    for i, j in enumerate(batch_idx):
        lines_id_list.append(lines_id[j] + padding_len[i] * [PAD_id])
    return s_lengths, lines_id_list

def PAD_MASK_Sentences_Source(lengths, lines_id, batch_idx, direction, PAD_id, BOS_id, EOS_id, MASK_id, mask_p):

    s_lengths = lengths[batch_idx]
    max_length = max(s_lengths)
    padding_len = max_length - s_lengths
    BOS_lines_id= []
    keep = np.random.rand(len(batch_idx), max_length) >= mask_p
    for i, j in enumerate(batch_idx):
        line = [word if keep[i][k] else MASK_id for k, word in enumerate(lines_id[j])]
        BOS_lines_id.append(line + padding_len[i] * [PAD_id])
    # s_lengths += 2
    return s_lengths, BOS_lines_id

def check_options(opt):
    if opt.eval_dict is not None:
        for i in range(len(opt.eval_dict)):
            assert os.path.exists(opt.eval_dict[i])
    ######write option to a file######
    shutil.copy("data/" + opt.data + "_inputs.txt", opt.save_dir + '/' +opt.data + "_params.txt")
    with open(opt.save_dir + '/' +opt.data + "_params.txt", "a") as f:
        opt_dict = vars(opt)
        for variable in opt_dict:
            f.write(str(variable) + ": " + str(opt_dict[variable]) + "\n")
        f.close()



def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp



def load_data(data, logger):
    file = open(data + ".data", 'rb')
    dataset = pickle.load(file)

    file = open(data + ".vocab_dict", 'rb')
    vocab_dict = pickle.load(file)

    for i in range(dataset.lang_size):
        logger.info("lang: "+ str(i))
        logger.info("V_size: "+ str(vocab_dict.V_size[i]))
        logger.info("train para sents: "+ str(len(dataset.lines_id[i])))
        logger.info("train mono sents: "+ str(len(dataset.lines_id_mono[i])))

    return dataset, vocab_dict




