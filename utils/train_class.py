import torch
import torch.nn as nn
import numpy as np
from logging import getLogger, basicConfig, INFO
# basicConfig(level = INFO)
import time
import os
from utils.train_base_new import PAD_Sentences, PAD_Sentences_Source
from torch.nn.utils import clip_grad_norm_
from torch import optim
from utils.matching_func import MUSE_get_word_translation_accuracy, load_dictionary
logger = getLogger("Log").getChild("sub")
class Trainer_MT():
    def __init__(self, model, dataset, file_name, vocab_dict, opt):
        self.model = model
        self.save_point = opt.save_point
        self.dataset = dataset
        self.special_token_V = 5
        self.lang_size = dataset.lang_size
        self.file_name = file_name
        self.early_stop_count = 0
        self.cumloss = 0
        self.cumloss_old = np.inf
        self.cumloss_new = np.inf
        self.valid_loss = -1 * np.inf
        self.enable_decay = False
        self.subword = opt.subword != None
        self.bestmodel = None
        self.old_model_name = None
        self.Mask_id = vocab_dict.vocab2id[0]["<MASK>"]
        self.PAD_id = vocab_dict.vocab2id[0]["<PAD>"]
        self.BOS_id = vocab_dict.vocab2id[0]["<BOS>"]
        self.EOS_id = vocab_dict.vocab2id[0]["<EOS>"]
        self.ignore_idx = -1
        self.best_acc_epoch = 0
        self.best_acc_ave = 0
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=self.ignore_idx, reduction='none')
        self.lstm_direction = [0, 1] # self.lstm_direction = [0] for l2r only
        if opt.eval_dict is not None:
            logger.info('load a dictionary for evaluation')
            self.dict_srclang = opt.dict_srclang
            self.dict_tgtlangs = opt.dict_tgtlangs
            self.eval_dict = self.load_dict(opt.eval_dict)
        else:
            self.eval_dict = None
        if len(dataset.batch_idx_list[1]) > 0:
            assert len(dataset.batch_idx_list[0]) ==0 ##assert no monolingual corpus
        self.para_Nbatch = len(dataset.batch_idx_list[0])
        self.mono_Nbatch = len(dataset.batch_idx_list[1])
        if len(dataset.lines_id_multi[0])> 0 : #multi
            logger.info("multi-supervised training")
            self.N_batch = self.para_Nbatch
        elif len(dataset.batch_idx_list[0]) != 0:
            logger.info("Supervised training")
            self.N_batch = self.para_Nbatch
        elif len(dataset.batch_idx_list[1]) != 0:
            logger.info("Unsupervised training")
            self.N_batch = self.mono_Nbatch

    def main(self, model, epoch_size, stop_threshold, remove_models, early_stop):
        logger.info("epoch start")
        for i in range(1, epoch_size+1): #for each epoch
            self.epoch = i
            self.main_epoch()
            improvement_rate = self.cumloss_new/self.cumloss_old
            logger.critical("epoch: " + str(i))
            logger.critical("current_loss: " + str(self.cumloss_new))
            logger.critical("current_loss/previous_loss: " + str(improvement_rate))
            if i == 1 or i % self.save_point == 0:
                self.Eval_model()
            new_model_name = self.file_name + "_epoch" + str(self.epoch) +'.model'
            self.save_model(new_model_name, remove_models)
            if (improvement_rate > stop_threshold or (early_stop and self.early_stop_count == 2)):
                break
        logger.info("finish training")
        self.write_scores()
        if self.bestmodel is not None:
            logger.info("load best model")
            if torch.cuda.device_count() > 1:
                model = model.module
            model.load_state_dict(torch.load(self.bestmodel))

        return model

    def main_epoch(self):
        self.cumloss = 0
        self.cumloss_old = self.cumloss_new
        start = time.time()
        if len(self.dataset.lines_id_multi[0])>0: #multilingual training
            self.dataset.batch_idx_list[0] = np.random.permutation(
                self.dataset.batch_idx_list[0])  # shuffle batch order
            for bt_idx in self.dataset.batch_idx_list[0]:
                self.Update_params_supervised_multi(self.dataset, bt_idx)

        elif len(self.dataset.batch_idx_list[0]) > 0 : #bilingual
            self.dataset.batch_idx_list[0] = np.random.permutation(self.dataset.batch_idx_list[0])  # shuffle batch order
            for bt_idx in self.dataset.batch_idx_list[0]:
                self.Update_params_supervised(self.dataset, bt_idx)

        elif len(self.dataset.batch_idx_list[1]) > 0: # unsupervised
            self.dataset.batch_idx_list[1] = np.random.permutation(self.dataset.batch_idx_list[1])  # shuffle batch order
            for bt_idx in self.dataset.batch_idx_list[1]:
                self.Update_params_unsupervised(self.dataset, bt_idx)

        # end of epoch
        self.cumloss_new = self.cumloss / self.N_batch
        elapsed_time = time.time() - start
        logger.info("Train elapsed_time:{0}".format(elapsed_time) + "[sec]")
        logger.info("loss: " + str(self.cumloss))
        logger.info(self.file_name + "_epoch" + str(self.epoch))

    def write_scores(self):
        with open(self.file_name + "_scores", "w") as f:
            f.write("best acc: " + str(self.best_acc_ave) + "\n")
            f.write("best acc epoch: " + str(self.best_acc_epoch) + "\n")
            f.write("best valid_loss: " + str(self.valid_loss) + "\n")

    def Eval_model(self):
        self.model.eval()
        self.model.zero_grad()
        with torch.no_grad():
            if self.eval_dict is not None:
                acc_ave = self.Eval_BLI(self.eval_dict)
                acc_ave = round(acc_ave, 3)
                if self.best_acc_ave <= acc_ave:
                    self.best_acc_ave = acc_ave
                    self.best_acc_epoch = self.epoch
                    logger.critical("best acc on Eval Dict: " + str(self.best_acc_ave))
                    if self.bestmodel is not None:
                        os.remove(self.bestmodel)
                    if torch.cuda.device_count() > 1:
                        torch.save(self.model.module.state_dict(), self.file_name + "_epoch" + str(self.epoch) + '.bestmodel')
                    else:
                        torch.save(self.model.state_dict(), self.file_name + "_epoch" + str(self.epoch) +'.bestmodel')
                    self.bestmodel = self.file_name + "_epoch" + str(self.epoch) +'.bestmodel'
                else:
                    if self.enable_decay:
                        logger.info("decay learning rate")
                        self.lr_rate = self.lr_rate * 0.7
                        for param_groups in self.optimizer.param_groups:
                            param_groups['lr'] = self.lr_rate
        self.model.train()

    def Update_params_supervised(self, dataset, index, *args):
        loss_all = 0
        for src in range(dataset.lang_size):
            self.model.Set_SrcLang(src)
            src_len = dataset.lengths[src][index]
            index = [x for _, x in sorted(zip(src_len, index), reverse=True)]
            t_lengths, t_id, t_id_EOS = \
                PAD_Sentences(dataset.lengths, dataset.lines_id, index, self.PAD_id, self.BOS_id, self.EOS_id,
                              self.ignore_idx)
            s_lengths, s_id = \
                PAD_Sentences_Source(dataset.lengths[src], dataset.lines_id[src], index, self.PAD_id)
            for direction in self.lstm_direction:
                self.model.zero_grad()
                tgt_list = list(range(dataset.lang_size))
                if torch.cuda.device_count() <= 1:  # If single GPU, encode s_id once for computation efficiency
                    s_id_emb, hs, hs_list = self.model.encode(s_id, s_lengths)
                for tgt in tgt_list:  # translation and reconstuction
                    self.model.Set_TgtLang(tgt, direction)
                    if torch.cuda.device_count() <= 1:
                        score, t_id_emb, _ = self.model.decode(t_id[direction][tgt], hs, s_lengths, s_id_emb)
                        loss = self.model.cross_entropy(score, self.model.torch.LongTensor(t_id_EOS[direction][tgt]).view(-1))  # (bs * maxlen_t,)
                    else:
                        # For multiple GPUs, loss should be calculated inside the model for faster computation
                        # Also, all inputs need to be in GPU
                        loss = self.model(self.model.torch.LongTensor(s_id),
                                          self.model.torch.LongTensor(s_lengths),
                                          self.model.torch.LongTensor(t_id[direction][tgt]),
                                          self.model.torch.LongTensor(t_lengths[tgt]),
                                          self.model.torch.LongTensor(t_id_EOS[direction][tgt]))
                    batch_size = len(s_id)
                    loss = torch.sum(loss)/batch_size
                    loss_all += loss
                loss_all.backward()
                clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()
                self.cumloss += loss_all.data.tolist()
                loss_all = 0

    def Update_params_supervised_multi(self, dataset, index, *args):
        loss_all = 0
        for pair_idx in range(dataset.lang_size - 1):
            for i, src_lang in enumerate([0, pair_idx+1]): #for each lang pair
                self.model.Set_SrcLang(src_lang)
                src_len = dataset.lengths_multi[pair_idx][i][index]
                index = [x for _, x in sorted(zip(src_len, index), reverse=True)]
                t_lengths, t_id, t_id_EOS = \
                    PAD_Sentences(dataset.lengths_multi[pair_idx], dataset.lines_id_multi[pair_idx], index, self.PAD_id,
                                  self.BOS_id, self.EOS_id,
                                  self.ignore_idx)
                s_lengths, s_id = \
                    PAD_Sentences_Source(dataset.lengths_multi[pair_idx][i], dataset.lines_id_multi[pair_idx][i], index,
                                         self.PAD_id)
                for direction in self.lstm_direction:
                    self.model.zero_grad()
                    if torch.cuda.device_count() <= 1:  # Single GPU: encode s_id only once for computation efficiency
                        s_id_emb, hs, hs_list = self.model.encode(s_id, s_lengths)
                    for j, tgt_lang in enumerate([0, pair_idx+1]): # translation and reconstuction
                        self.model.Set_TgtLang(tgt_lang, direction)
                        if torch.cuda.device_count() <= 1:  # Single GPU: encode only once for computation efficiency
                            score, t_id_emb, attn_matix = self.model.decode(t_id[direction][j], hs, s_lengths,s_id_emb)
                            loss = self.model.cross_entropy(score, self.model.torch.LongTensor(t_id_EOS[direction][j]).view(-1))  # (bs * maxlen_t,)
                        else:
                            # For multiple GPUs, loss should be calculated inside the model for faster computation
                            # Also, all inputs need to be in GPU
                            loss = self.model(self.model.torch.LongTensor(s_id),
                                              self.model.torch.LongTensor(s_lengths),
                                              self.model.torch.LongTensor(t_id[direction][j]),
                                              self.model.torch.LongTensor(t_lengths[j]),
                                              self.model.torch.LongTensor(t_id_EOS[direction][j]))
                        batch_size = len(s_id)
                        loss = torch.sum(loss)/batch_size
                        loss_all += loss
                    loss_all.backward()
                    clip_grad_norm_(self.model.parameters(), 5.0)
                    self.optimizer.step()
                    self.cumloss += loss_all.data.tolist()
                    loss_all = 0

    def Update_params_unsupervised(self, dataset, index, *args):
        t_lengths, t_id, t_id_EOS = \
            PAD_Sentences(dataset.lengths_mono, dataset.lines_id_mono, index,
                          self.PAD_id, self.BOS_id, self.EOS_id,self.ignore_idx)
        for src in range(dataset.lang_size): #es, fr en,
            for direction in self.lstm_direction:
                self.model.zero_grad()
                self.model.Set_TgtLang(src, direction)
                loss = self.model(None, None,
                                  self.model.torch.LongTensor(t_id[direction][src]),
                                  self.model.torch.LongTensor(t_lengths[src]),
                                  self.model.torch.LongTensor(t_id_EOS[direction][src]))
                batch_size = len(t_id[direction][src])
                loss = torch.sum(loss)/batch_size
                loss.backward()
                clip_grad_norm_(self.model.parameters(), 5.0)
                self.optimizer.step()
                self.cumloss += loss.data.tolist()

    def save_model(self, new_model_name, remove_models):
        if torch.cuda.device_count() > 1:
            torch.save(self.model.module.state_dict(), new_model_name)
        else:
            torch.save(self.model.state_dict(), new_model_name)
        if (remove_models and self.epoch != 1):
            logger.info("remove the previous model")
            os.remove(self.old_model_name)
        self.old_model_name = new_model_name

    def load_dict(self, eval_dict):
        eval_dict_loaded = []
        for pair, tgt in enumerate(self.dict_tgtlangs):
            dico = load_dictionary(eval_dict[pair], self.model.vocab2id[self.dict_srclang], self.model.vocab2id[tgt], init_idx=self.special_token_V)
            eval_dict_loaded.append(dico)
        return eval_dict_loaded

    def set_device(self, is_cuda):
        self.model.set_device(is_cuda)
        if is_cuda:
            self.torch = torch.cuda
        else:
            self.torch = torch

    def Set_Optimiser(self, model, opt_type, lr_rate):
        if opt_type == "SGD":
            self.lr_rate = lr_rate
            self.optimizer = optim.SGD(model.parameters(), lr=self.lr_rate)
            self.enable_decay = True
        elif opt_type == "Adam":
            self.optimizer = optim.Adam(model.parameters())

    def Eval_BLI(self, dict):
        self.model.eval()
        with torch.no_grad():
            acc_ave = 0
            for pair, tgt in enumerate(self.dict_tgtlangs):
                SRCEmb_W = self.model.embedding_weight(self.dict_srclang)[self.special_token_V:] #omit special tokens ("<PAD>","<MASK>","<BOS>","<EOS>","UNK")
                TGTEmb_W = self.model.embedding_weight(tgt)[self.special_token_V:] #omit special tokens ("<PAD>","<MASK>","<BOS>","<EOS>","UNK")
                results = MUSE_get_word_translation_accuracy(dict[pair], SRCEmb_W.detach(), TGTEmb_W.detach())
                acc_ave += np.log(results[0]+0.0000001)
            acc_ave = np.exp(acc_ave/len(self.dict_tgtlangs))
            logger.info(acc_ave)
        self.model.train()
        return acc_ave