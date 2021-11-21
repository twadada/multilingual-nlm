# # #!/usr/bin/env python
# # # # # # # -*- coding: utf-8 -*-
import os
import numpy as np
import argparse
import pickle
from logging import getLogger, StreamHandler, FileHandler, basicConfig, INFO, WARNING
basicConfig(level=WARNING) #Use INFO to print more information
import sys,os
sys.path.append(os.getcwd())
from train_option import global_train_parser
import warnings
from utils.train_base_new import check_options, load_data, save_emb, out_wordemb, generate_bacth_idx, preprare_model
from utils.train_class import Trainer_MT
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(parents=[global_train_parser])
    opt = parser.parse_args()
    if opt.subword:
        assert opt.share_vocab != 0, "Subwords are not used"
    opt.save_dir = opt.save_dir +'-'.join([str(x) for x in opt.lang_class])
    logger = getLogger('Log')
    handler = StreamHandler()
    logger.addHandler(handler)
    if (os.path.isdir(opt.save_dir)):
        message = 'Directory ' + "'" + opt.save_dir + "'" + ' already exists.'
        warnings.warn(message)
    else:
        os.mkdir(opt.save_dir)
    fh = FileHandler(opt.save_dir + '/test.log')
    logger.addHandler(fh)
    check_options(opt)
    with open(opt.save_dir+"/options.pkl", 'wb') as f:
        pickle.dump(opt, f)  # N_idiom, N_sample, s_len, dim
    file_name = opt.save_dir + '/' + opt.data
    logger.info("Save model as: "+ file_name)
    dataset, vocab_dict = load_data("data/"+opt.data, logger)
    dataset.batch_idx_list = generate_bacth_idx(dataset.lengths, dataset.lengths_mono, dataset.lengths_multi, opt.batch_size, logger)
    model, dataset, vocab_dict = preprare_model(opt, dataset, vocab_dict,logger)
    logger.info(model)
    model.set_device(opt.gpu)
    if opt.gpu:
        model.to('cuda')
    if torch.cuda.device_count() > 1:
        from models.new_models import MyDataParallel
        logger.critical("Use multiple GPUs")
        model = MyDataParallel(model)

    trainer = Trainer_MT(model, dataset, file_name, vocab_dict, opt)
    trainer.Set_Optimiser(model, opt.opt_type, opt.learning_rate_SGD)
    bestmodel = trainer.main(model, opt.epoch_size, opt.stop_threshold, opt.remove_models, opt.early_stop)
    logger.info("save embeddings")
    bestmodel.eval()
    if opt.pretrained_emb:
        param_a = bestmodel.embedding_weight.a.data.cpu().numpy()
        np.save(file_name + ".param_a", param_a)
        param_b = bestmodel.embedding_weight.b.data.cpu().numpy()
        np.save(file_name + ".param_b", param_b)
    with torch.no_grad(): #save word embeddings
        for lang in range(bestmodel.lang_size):
            emb_weight = bestmodel.embedding_weight(lang)
            vocab2emb = out_wordemb(vocab_dict.id2vocab[lang], emb_weight)
            save_name = file_name + '.lang' + str(lang) + '.vec'
            save_emb(vocab2emb, opt.emb_size, save_name)
