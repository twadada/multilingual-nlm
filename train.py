# # #!/usr/bin/env python
# # # # # # # -*- coding: utf-8 -*-
#
import os
import numpy as np
from utils.train_base import Trainer_base, Out_Wordemb, Save_Emb
from models.model import Shared_Langage_Model
import pickle
import argparse
from train_option import global_train_parser
from utils.minibatch_processing import Sort_data_by_sentlen, Generate_bacth_idx
import torch
import torch.nn as nn
import time
import warnings
import shutil

class Langage_Model_Class(nn.Module):

    def __init__(self, lm, lang_size, vocab2id_input, vocab2id_output):
        super().__init__()
        self.lm = lm
        self.lang_size = lang_size
        self.ignore_index = vocab2id_output["<ignore_idx>"]
        self.PAD_index = vocab2id_input["<PAD>"] #PAD Embedding index
        self.BOS_fwd_index = vocab2id_input["<BOS_fwd>"]
        self.BOS_bkw_index = vocab2id_input["<BOS_bkw>"]
        self.EOS_index = vocab2id_output["<EOS>"]
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=self.ignore_index, reduction='none')
    def set_device(self, gpuid):
        is_cuda = gpuid >= 0
        self.lm.set_device(is_cuda)
        if is_cuda:
            self.LongTensor = torch.cuda.LongTensor
        else:
            self.LongTensor = torch.LongTensor

    def __call__(self, s_id, s_lengths, *args):
        _, _, _, ht, softmax_score = self.lm(s_id, s_lengths, *args)
        return softmax_score

    def Calc_loss(self,softmax_score, t_id_EOS):
        #softmax_score: bs, s_len, tgtV
        #t_id_EOS: bs, s_len
        t_id_EOS = torch.stack(t_id_EOS)
        batch_size, s_len, tgtV = softmax_score.size()
        loss = self.cross_entropy(softmax_score.view(batch_size*s_len, tgtV), t_id_EOS.view(-1))  # (bs * maxlen_t,)
        loss = torch.sum(loss) / batch_size
        return loss

    def Register_vocab(self,vocab2id_input, vocab2id_output,id2vocab_input,id2vocab_output):
        self.vocab2id_input = vocab2id_input
        self.vocab2id_output = vocab2id_output
        self.id2vocab_input = id2vocab_input
        self.id2vocab_output = id2vocab_output


class Trainer():
    def __init__(self, opt):
        self.opt = opt
        self.file_name = opt.save_dir + '/' +opt.data
        self.check_options()

    def check_options(self):
        np.random.seed(self.opt.seed)
        print("emb_size", self.opt.emb_size)
        print("hidden_size", self.opt.h_size)
        ######write option to a file######
        shutil.copy("data/" + self.opt.data + "_inputs.txt", self.file_name + "_params.txt")
        with open(self.file_name + "_params.txt", "a") as f:
            opt_dict = vars(self.opt)
            for variable in opt_dict:
                f.write(str(variable) + ": " + str(opt_dict[variable]) + "\n")
            f.close()

    def build_model(self, n_layer, emb_size, h_size, dr_rate, Langage_Model_Class, vocab_dict):

        lm = Shared_Langage_Model(n_layer, emb_size, h_size, dr_rate, vocab_dict)
        model = Langage_Model_Class(lm, len(vocab_dict.vocab2id_input), vocab_dict.vocab2id_input[0],
                              vocab_dict.vocab2id_output[0])

        for param in model.parameters():
            param.data.uniform_(-0.1, 0.1)
        ##zero embedding for padding##
        model.lm.emb.weight.data[model.PAD_index] *= 0



        model.Register_vocab(vocab_dict.vocab2id_input, vocab_dict.vocab2id_output, vocab_dict.id2vocab_input,
                             vocab_dict.id2vocab_output)

        return model

    def load_data(self, data):
        file = open("data/" + data + ".data", 'rb')
        dataset = pickle.load(file)

        file = open("data/" + data + ".vocab_dict", 'rb')
        vocab_dict = pickle.load(file)

        for i in range(dataset.lang_size):
            print("lang: ", i)
            print("V_size: ", vocab_dict.V_size[i])
            print("train sents: ", len(dataset.lines_id_input[i]))

        return dataset, vocab_dict

    def Generate_MiniBatch(self,dataset,batch_size):
        #### generate mini-bathces #####
        ####Sort by length####
        dataset = Sort_data_by_sentlen(dataset)
        dataset.batch_idx_list = Generate_bacth_idx(dataset, batch_size)
        return dataset

    def main(self, dataset, vocab_dict, Langage_Model_Class, Trainer_base):

        print("Save model as: ", self.file_name)

        model = self.build_model(self.opt.n_layer,
                                 self.opt.emb_size, self.opt.h_size,
                                 self.opt.dr_rate, Langage_Model_Class, vocab_dict)

        print("Number of mini-batches",len(dataset.batch_idx_list ))
        #### end #####
        ####CPU->GPU###
        model.set_device(self.opt.gpuid)
        if self.opt.gpuid >= 0:
            torch.cuda.set_device(self.opt.gpuid)
            model.to('cuda')

        trainer = Trainer_base(dataset, self.opt, self.file_name)
        trainer.set_optimiser(model, self.opt.opt_type, self.opt.learning_rate)
        bestmodel = trainer.main(model, self.opt.epoch_size, self.opt.stop_threshold, self.opt.remove_models)
        vocab2emb_list = Out_Wordemb(vocab_dict.id2vocab_input, bestmodel.lm)
        Save_Emb(vocab2emb_list, self.opt.emb_size, self.file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[global_train_parser])
    options = parser.parse_args()
    if (os.path.isdir(options.save_dir)):
        message = options.save_dir + ' exists already.'
        warnings.warn(message)
    else:
        os.mkdir(options.save_dir)

    trainer = Trainer(options)
    dataset, vocab_dict = trainer.load_data(options.data)
    dataset = trainer.Generate_MiniBatch(dataset,options.batch_size)
    trainer.main(dataset, vocab_dict ,Langage_Model_Class, Trainer_base)

