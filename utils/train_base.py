#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
import time
import os

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

        BOS_lines_id_input.append(model.LongTensor(
            [model.BOS_fwd_index] + input_line + padding_len[i] * [model.PAD_index]))
        lines_id_output_EOS.append(model.LongTensor(
            output_line + [model.EOS_index] + padding_len[i] * [model.ignore_index]))
        BOS_lines_id_input_bkw.append(model.LongTensor(
            [model.BOS_bkw_index] + input_line[::-1] + padding_len[i] * [model.PAD_index]))
        lines_id_output_EOS_bkw.append(model.LongTensor(
            output_line[::-1] + [model.EOS_index] + padding_len[i] * [model.ignore_index]))

    return s_lengths, BOS_lines_id_input, lines_id_output_EOS, BOS_lines_id_input_bkw, lines_id_output_EOS_bkw

class Trainer_base():
    def __init__(self, dataset, opt, file_name):
        self.dataset = dataset
        self.file_name = file_name
        self.cumloss_old = np.inf
        self.cumloss_new = np.inf

    def Update_params_base(self, model, optimizer, s_id, s_id_EOS, s_lengths,*args):
        model.zero_grad()
        softmax_score = model(s_id, s_lengths, *args)
        loss = model.Calc_loss(softmax_score, s_id_EOS)
        loss.backward()
        clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        return loss.data.tolist()

    def Update_params(self, model, dataset, optimizer, index, *args):

        loss_all = 0
        for lang in range(model.lang_size):
            model.lm.Switch_Lang(lang)
            s_lengths, BOS_lines_id_input, lines_id_output_EOS, BOS_lines_id_input_bkw, lines_id_output_EOS_bkw = \
                PAD_Sentences(model, dataset.lengths[lang], dataset.lines_id_input[lang],
                                  dataset.lines_id_output[lang], index)

            model.lm.Switch_fwdbkw("fwd")
            loss_all += self.Update_params_base(model, optimizer, BOS_lines_id_input, lines_id_output_EOS, s_lengths)

            model.lm.Switch_fwdbkw("bkw")
            loss_all += self.Update_params_base(model, optimizer, BOS_lines_id_input_bkw, lines_id_output_EOS_bkw,
                                           s_lengths)
        return loss_all

    def set_optimiser(self, model, opt_type, lr_rate):
        if opt_type == "SGD":
            self.optimizer = optim.SGD(model.parameters(), lr=lr_rate)
        elif opt_type == "ASGD":
            self.optimizer = optim.ASGD(model.parameters(), lr=lr_rate)

    def main(self,model,epoch_size, stop_threshold,remove_models =False):

        print ("epoch start")
        old_model_name = None
        for epoch in range(1, epoch_size+1): #for each epoch
            print("epoch: ",epoch)
            self.cumloss_old = self.cumloss_new
            cumloss = 0
            batch_idx_list = np.random.permutation(self.dataset.batch_idx_list) # shuffle batch order
            start = time.time()
            for bt_idx in batch_idx_list:
                loss = self.Update_params(model, self.dataset, self.optimizer, bt_idx)
                cumloss = cumloss + loss
            #end of epoch

            self.cumloss_new = cumloss/len(batch_idx_list)
            elapsed_time = time.time() - start
            print("Train elapsed_time:{0}".format(elapsed_time) + "[sec]")
            print("loss: ",cumloss)
            print(self.file_name +"_epoch" + str(epoch))
            new_model_name = self.file_name + "_epoch" + str(epoch) +'.model'
            torch.save(model.state_dict(), new_model_name)
            if (remove_models and epoch != 1):
                print("remove the previous model")
                os.remove(old_model_name)
            old_model_name = new_model_name
            improvement_rate = self.cumloss_new / self.cumloss_old
            print("loss improvement rate:",improvement_rate)
            if (improvement_rate > stop_threshold):
                break

        return model

