# # #!/usr/bin/env python
# # # # # # # -*- coding: utf-8 -*-
#
import os
import argparse
from train_option import global_train_parser
import warnings
from utils.train_base import check_options, load_data, Setup_model, Out_Wordemb, Save_Emb
from utils.minibatch_processing import Generate_MiniBatch
from utils.train_class import Langage_Model_Class, Trainer
from models.model import Shared_Langage_Model


if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[global_train_parser])
    opt = parser.parse_args()

    if (os.path.isdir(opt.save_dir)):
        message = 'Directory ' + "'" + opt.save_dir + "'" +' already exists.'
        warnings.warn(message)
    else:
        os.mkdir(opt.save_dir)

    check_options(opt)

    file_name = opt.save_dir + '/' + opt.data
    print("Save model as: ", file_name)

    dataset, vocab_dict = load_data(opt.data)
    dataset = Generate_MiniBatch(dataset, opt.batch_size)
    print("Number of mini-batches", len(dataset.batch_idx_list))

    lm = Shared_Langage_Model(opt.n_layer, opt.emb_size,  opt.h_size, opt.dr_rate, vocab_dict)
    model = Langage_Model_Class(lm, len(vocab_dict.vocab2id_input), vocab_dict.vocab2id_input[0],
                          vocab_dict.vocab2id_output[0])

    model = Setup_model(model, opt.gpuid, vocab_dict)

    trainer = Trainer(dataset, file_name)
    trainer.set_optimiser(model, opt.opt_type, opt.learning_rate)
    bestmodel = trainer.main(model, opt.epoch_size, opt.stop_threshold, opt.remove_models)

    print("save embeddings")
    vocab2emb_list = Out_Wordemb(vocab_dict.id2vocab_input, bestmodel.lm)
    Save_Emb(vocab2emb_list, opt.emb_size, file_name)
