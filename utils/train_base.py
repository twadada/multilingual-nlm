#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
import shutil
import pickle
from models.model import Shared_Langage_Model

def Out_Wordemb(id2vocab, lm, vocab_size = 0):
    Emb = getattr(lm, "emb")  # lookup table for all languages
    W = Emb.weight.data.tolist() # Vocab_size , emb_size
    vocab2emb_list = []
    if vocab_size == 0:
        vocab_size = len(W)
    for lang in range(len(id2vocab)):
        vocab2emb = {}
        for id in list(id2vocab[lang].keys())[:vocab_size]:
            emb = W[id] #demb
            vocab2emb[id2vocab[lang][id]] = emb
        vocab2emb_list.append(vocab2emb)

    return vocab2emb_list

def Save_Emb(vocab2emb_list, N_dim, filename):
    lang_size = len(vocab2emb_list)
    for lang in range(lang_size):
        vocab2emb = vocab2emb_list[lang]
        N_word = len(vocab2emb)
        first_line = str(N_word) + " " + str(N_dim)
        with open(filename + '.lang' + str(lang) + '.vec', "w") as word_output:
            word_output.write(first_line + "\n")
        vocab = list(vocab2emb.keys())
        vocab.remove("<PAD>")
        for word in vocab:
            out = word  + " " +  " ".join(map(str, vocab2emb[word])) + "\n"
            with open(filename+'.lang'+ str(lang)+'.vec', "a") as word_output:
                word_output.write(out)

def PAD_Sentences(model, lengths, lines_id_input, lines_id_output, index):

    s_lengths = lengths[index]
    max_length = max(s_lengths)
    padding_len = max_length - s_lengths

    BOS_lines_id_input = []
    lines_id_output_EOS = []
    BOS_lines_id_input_bkw = []
    lines_id_output_EOS_bkw = []

    for i, j in enumerate(index):
        input_line = lines_id_input[j]
        output_line = lines_id_output[j]
        BOS_lines_id_input.append([model.BOS_fwd_index] + input_line + padding_len[i] * [model.PAD_index])
        lines_id_output_EOS.append(output_line + [model.EOS_index] + padding_len[i] * [model.ignore_index])
        BOS_lines_id_input_bkw.append([model.BOS_bkw_index] + input_line[::-1] + padding_len[i] * [model.PAD_index])
        lines_id_output_EOS_bkw.append(output_line[::-1] + [model.EOS_index] + padding_len[i] * [model.ignore_index])

    return s_lengths, BOS_lines_id_input, lines_id_output_EOS, BOS_lines_id_input_bkw, lines_id_output_EOS_bkw

def check_options(opt):
    np.random.seed(opt.seed)
    print("emb_size", opt.emb_size)
    print("hidden_size", opt.h_size)
    ######write option to a file######
    shutil.copy("data/" + opt.data + "_inputs.txt", opt.save_dir + '/' +opt.data + "_params.txt")
    with open(opt.save_dir + '/' +opt.data + "_params.txt", "a") as f:
        opt_dict = vars(opt)
        for variable in opt_dict:
            f.write(str(variable) + ": " + str(opt_dict[variable]) + "\n")
        f.close()

def build_model(n_layer, emb_size, h_size, dr_rate, gpuid, Langage_Model_Class, vocab_dict):

    lm = Shared_Langage_Model(n_layer, emb_size, h_size, dr_rate, vocab_dict)
    model = Langage_Model_Class(lm, len(vocab_dict.vocab2id_input), vocab_dict.vocab2id_input[0],
                          vocab_dict.vocab2id_output[0])

    for param in model.parameters():
        param.data.uniform_(-0.1, 0.1)
    ##zero embedding for padding##
    model.lm.emb.weight.data[model.PAD_index] *= 0

    model.Register_vocab(vocab_dict.vocab2id_input, vocab_dict.vocab2id_output, vocab_dict.id2vocab_input,
                         vocab_dict.id2vocab_output)
    model.set_device(gpuid)
    if gpuid >= 0:
        torch.cuda.set_device(gpuid)
        model.to('cuda')

    return model

def load_data(data):
    file = open("data/" + data + ".data", 'rb')
    dataset = pickle.load(file)

    file = open("data/" + data + ".vocab_dict", 'rb')
    vocab_dict = pickle.load(file)

    for i in range(dataset.lang_size):
        print("lang: ", i)
        print("V_size: ", vocab_dict.V_size[i])
        print("train sents: ", len(dataset.lines_id_input[i]))

    return dataset, vocab_dict





