# -*- coding: utf-8 -*-
#!/usr/bin/env python

import torch
import torch.nn as nn

class Shared_Langage_Model(nn.Module):

    def __init__(self, n_layer, emb_size, h_size, dr_rate, vocab_dict,*args):
        super().__init__()

        ####______shared params_______####
        self.dr_rate = dr_rate
        self.Ws_share = nn.Linear(h_size, 1, bias=False) #W for EOS
        self.lstm_fwd = nn.LSTM(
            input_size=emb_size,
            hidden_size=h_size,
            num_layers=n_layer,
            batch_first=True,
            dropout=dr_rate
)
        self.lstm_bkw = nn.LSTM(
            input_size=emb_size,
            hidden_size=h_size,
            num_layers=n_layer,
            batch_first=True,
            dropout=dr_rate
        )
        self.dropout = nn.Dropout(p=dr_rate)
        ####______shared params_______####

        ####______specific params_______####
        Max_Word_idx = max(vocab_dict.id2vocab_input[-1].keys())+1 #[0,1,2,3], max = 3, idx_len = 3+1
        self.emb = nn.Embedding(Max_Word_idx, emb_size, padding_idx= vocab_dict.vocab2id_input[0]["<PAD>"]) #lookup table for all languages
        layer = []
        for lang in range(len(vocab_dict.id2vocab_output)):
            layer.append(nn.Linear(h_size, vocab_dict.V_size[lang]-1, bias=False))

        self.Ws_i = nn.ModuleList(layer)
        ####______specific params_______####

    def __call__(self, BOS_t_id, t_lengths, *args):
        return self.forward(BOS_t_id, t_lengths, *args)

    def Switch_Lang(self, lang):

        self.Ws = self.Ws_i[lang] #switch output layer

    def Switch_fwdbkw(self,type):
        if (type == "fwd"):
            self.lstm = self.lstm_fwd

        elif (type == "bkw"):
            self.lstm = self.lstm_bkw

        else:
            raise Exception("Invalid type")

    def forward(self,BOS_t_id, t_lengths, *args):

        t_id_emb, h_last, c_last, ht = self.decode(BOS_t_id, t_lengths, *args)
        score_V = self.Ws(self.dropout(ht))
        score_eos = self.Ws_share(self.dropout(ht))
        score = torch.cat([score_eos, score_V], dim=2)  # (bs, maxlen_t, tgtV)

        return t_id_emb, h_last, c_last, ht, score

    def decode(self, t_id, t_lengths, *args):

        t_id_emb = self.emb(torch.stack(t_id))  #bs, max_s_len, * emb_size

        ht, (h_last, c_last) = self.lstm(t_id_emb)  # ht: bt * len_t * demb(最上位層の出力 from each hidden state)

        return t_id_emb, h_last, c_last, ht

    def set_device(self,is_cuda):
        if is_cuda:
            self.torch = torch.cuda
        else:
            self.torch = torch
