# -*- coding: utf-8 -*-
#!/usr/bin/env python
import torch
import torch.nn as nn

from logging import getLogger, StreamHandler, FileHandler, basicConfig, INFO
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import torch.nn.functional  as F
import numpy as np

def cossim(src,tgt):
    src = src / src.norm(2, 1, keepdim=True).expand_as(src)  # s_len, emb
    tgt = tgt / tgt.norm(2, 1, keepdim=True).expand_as(tgt)  # t_len, emb
    cos_similarity = src.mm(tgt.transpose(0, 1)).data.cpu().numpy()  # s_len, t_len
    return cos_similarity

logger = getLogger("Log").getChild("sub")

class MyDataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

def load_w2v(file, max_count = np.inf):
    word_list = []
    vec_list = []
    count = 0
    with open(file, 'r', errors='ignore') as f:
        first_line = f.readline()
        dim = int(first_line.split(' ')[1])
        for line in f:
            line = line.rstrip('\n')
            line = line.rstrip(' ')
            w = line.split(' ')[0]
            vec = line.split(' ')[1:]
            count+=1
            if len(vec) == dim:# and w not in word_list:
                word_list.append(w)
                vec_list.append(vec)
            else:
                print(len(vec))
                print(vec)
            if count>=max_count:
                return first_line, dim, word_list, np.array(vec_list).astype(np.float32)

    return first_line, dim, word_list, np.array(vec_list).astype(np.float32)

def Load_pretrain_emb(file, vocab_dict, emb_size, tgt_lang, emb_shape):
    assert len(file) == 1
    assert tgt_lang == 1
    words_found = 0
    first_line, dim, word_list, vec_list = load_w2v(file[0])
    assert emb_size == dim
    vocab2emb = dict(zip(word_list, vec_list))
    new_weights_matrix = np.zeros((emb_shape[0],emb_shape[1]))
    mask_vector = np.ones((emb_shape[0], 1))  # V_size, emb_size
    for word_id in vocab_dict.id2vocab[tgt_lang].keys():
        word = vocab_dict.id2vocab[tgt_lang][word_id]
        if word in word_list:
            new_weights_matrix[word_id] = vocab2emb[word]
            mask_vector[word_id] = 0
            words_found += 1
        else:
            new_weights_matrix[word_id] = np.zeros(dim)
    logger.info("#hits"+str(words_found)+ "#words " + str(len(vocab_dict.id2vocab[tgt_lang].keys())))
    logger.critical(str(words_found) + " words in the vocabulary were found in " + file[0])
    if words_found<1000:
        logger.warning("LESS THAN 1,000 WORDS WERE FOUND in "+ file[0])
    # if words_found/len(vocab_dict.id2vocab[tgt_lang].keys()) <0.5:
    #     raise Exception("Pretrained Embeddings are too low")
    emb_new = nn.Embedding.from_pretrained(torch.FloatTensor(new_weights_matrix))
    emb_new.weight.requires_grad = False #freeze embeddings
    return emb_new, torch.FloatTensor(mask_vector)

class Word_Embedding(nn.Module):

    def __init__(self, lang_size, vocab_dict, emb_size, dr_rate, pretrained_emb, *args):
        super().__init__()
        layer = []
        self.dr_rate = dr_rate
        self.emb_dropout = nn.Dropout(p=dr_rate)
        self.emb_size = emb_size
        for lang in range(lang_size):
            layer.append(nn.Embedding(vocab_dict.V_size[lang] + len(vocab_dict.shared_V), emb_size,
                                      padding_idx=vocab_dict.vocab2id[0]["<PAD>"]))
        self.emb_list = nn.ModuleList(layer)
        self.use_pretrained_emb = False
        if pretrained_emb:
            assert lang_size == 2
            self.use_pretrained_emb = True
            tgt_lang = 1
            self.preemb_dim = emb_size
            emb, mask_vector_list = Load_pretrain_emb(pretrained_emb, vocab_dict, emb_size, tgt_lang, self.emb_list[tgt_lang].weight.shape)
            self.emb_list[tgt_lang] = emb
            self.mask_vec_list = mask_vector_list
            self.OOV_emb = nn.Embedding(vocab_dict.V_size[tgt_lang] + len(vocab_dict.shared_V),
                                        emb_size, padding_idx=vocab_dict.vocab2id[0]["<PAD>"])
            self.a = nn.Parameter(torch.ones(self.preemb_dim))
            self.b = nn.Parameter(torch.zeros(self.preemb_dim))

    def forward(self,lang):
        W_emb = self.emb_list[lang].weight  # V * emb_size
        if self.use_pretrained_emb and lang == 1: #tgt lang
            self.mask_vec_list = self.mask_vec_list.to(W_emb.device)
            OOV_wordemb = self.OOV_emb.weight * self.mask_vec_list
            W_emb = (self.a * W_emb + self.b) * (1 - self.mask_vec_list)
            return self.emb_dropout(W_emb + OOV_wordemb)
        else:
            return self.emb_dropout(W_emb)

class SubWord_Embedding(nn.Module):
    def __init__(self, lang_size, vocab_dict, emb_size, share_vocab, subword_file, dr_rate, pretrained_emb):
        super().__init__()
        self.emb_dropout = nn.Dropout(p=dr_rate)
        self.emb_size = emb_size
        self.swemb_size = emb_size
        self.Register_subword_word2sub(subword_file, lang_size, vocab_dict)
        self.sw_emb = nn.Embedding(len(self.subword2id), self.swemb_size)
        self.share_vocab = share_vocab
        layer = []
        for lang in range(lang_size):
            layer.append(nn.Embedding(vocab_dict.V_size[lang] + len(vocab_dict.shared_V) , self.emb_size,
                                      padding_idx=vocab_dict.vocab2id[0]["<PAD>"]))
        self.emb_list = nn.ModuleList(layer)

        if self.share_vocab == 4: #CNN
            self.CNN = nn.Conv1d(self.swemb_size, self.emb_size, kernel_size=3, stride=1, padding=1)
            self.CNNmask = []
            for lang in range(len(self.word2subword_len_list)):
                V, max_n_sw = self.word2subword_list[lang].size()
                mask = np.ones((V, max_n_sw, self.swemb_size))  # V, max_n_sw, swemb_size
                for i, l in enumerate(self.word2subword_len_list[lang]):  # for each batch
                    mask[i, l:] = 0
                if torch.cuda.is_available():
                    self.CNNmask.append(torch.cuda.FloatTensor(mask))
                else:
                    self.CNNmask.append(torch.FloatTensor(mask))
        self.use_pretrained_emb = False
        if pretrained_emb:
            assert lang_size == 2
            self.use_pretrained_emb = True
            tgt_lang = 1
            self.preemb_dim = emb_size
            emb, mask_vector_list= Load_pretrain_emb(pretrained_emb, vocab_dict, emb_size, tgt_lang, self.emb_list[tgt_lang].weight.shape)
            self.emb_list[tgt_lang] = emb
            self.mask_vec_list = mask_vector_list
            self.OOV_emb = nn.Embedding(vocab_dict.V_size[tgt_lang] + len(vocab_dict.shared_V),
                                        emb_size, padding_idx=vocab_dict.vocab2id[0]["<PAD>"])
            self.a = nn.Parameter(torch.ones(self.preemb_dim))
            self.b = nn.Parameter(torch.zeros(self.preemb_dim))

    def word_embedding(self, lang):
        W_emb = self.emb_list[lang].weight  # bs, max_s_len, * emb_size
        return W_emb

    def Register_subword_word2sub(self, subword_file, lang_size, vocab_dict):
        #subword_file: e.g. "understandable under stand able"
        subwords = [[] for _ in range(len(subword_file))]
        subword_max_len = [0 for _ in range(len(subword_file))]
        for lang in range(len(subword_file)):
            for line in open(subword_file[lang], encoding="utf8"):
                word = line.rstrip('\n').split()[0]
                assert word in vocab_dict.vocab2id[lang].keys(), word + " not in the vocabulary"
                line = line.rstrip('\n').split()
                subwords[lang].extend(line[1:])
                if subword_max_len[lang] < len(line[1:]):
                    subword_max_len[lang] = len(line[1:])
            logger.info("subword_max_len"+str(lang)+": " + str(subword_max_len[lang]))
            logger.info("subword vocab size" + str(lang) +": " + str(len(set(subwords[lang]))))

        subwords = sum(subwords, []) #flatten
        special_tokens =["<PAD>", "<MASK>", "<BOS>", "<EOS>","UNK"] #list of subwords; MASK not used
        subwords = sorted(list(set(subwords))) + special_tokens #list of subwords
        self.subword2id = dict(zip(subwords, range(len(subwords))))
        logger.critical("Number of shared subwords: " + str(len(self.subword2id)))
        self.word2subword_list = []
        self.word2subword_len_list = []
        for lang in range(lang_size):
            word2subword = [[] for _ in range((vocab_dict.V_size[lang] + len(vocab_dict.shared_V)))]
            word2subword_len = [[None] for _ in range((vocab_dict.V_size[lang] + len(vocab_dict.shared_V)))]
            for line in open(subword_file[lang], encoding="utf8"):
                line = line.rstrip('\n').split()
                word = line[0]
                subwords = line[1:]
                subword_ids = [self.subword2id[sw] for sw in subwords if sw in self.subword2id]
                subword_len = len(subword_ids)
                if subword_len < 1:
                    print (word)
                    print (subwords)
                    raise Exception
                word_id = vocab_dict.vocab2id[lang][word]
                word2subword[word_id] = subword_ids + [0] * (subword_max_len[lang]-subword_len) #padding
                word2subword_len[word_id] = subword_len
            for word in special_tokens:
                word_id = vocab_dict.vocab2id[lang][word]
                word2subword[word_id] = [self.subword2id[word]]+ [0] * (subword_max_len[lang]-1)
                word2subword_len[word_id] = 1

            if torch.cuda.is_available():
                ###TODO: multi-GPU
                self.word2subword_list.append(torch.cuda.LongTensor(word2subword))
                self.word2subword_len_list.append(torch.cuda.LongTensor(word2subword_len))
            else:
                self.word2subword_list.append(torch.LongTensor(word2subword))
                self.word2subword_len_list.append(torch.LongTensor(word2subword_len))

    def forward(self, lang):
        Word_emb = self.word_embedding(lang)
        word2sw = self.word2subword_list[lang].to(Word_emb.device)  # V, max_n_sw
        sw_emb = self.sw_emb(word2sw)  # V, max_n_sw, sw_size
        sw_emb = self.emb_dropout(sw_emb)
        if self.share_vocab == 3:
            sw_emb = self.sum_embedding(sw_emb)
        elif self.share_vocab == 4:
            sw_emb = self.CNN_embedding(sw_emb, lang)
        else:
            raise Exception
        sw_emb = sw_emb/self.word2subword_len_list[lang].to(sw_emb.device).float().view(-1, 1).expand_as(sw_emb)
        if self.use_pretrained_emb and lang == 1:
            self.mask_vec_list = self.mask_vec_list.to(sw_emb.device)
            sw_emb = sw_emb * self.mask_vec_list
            OOV_wordemb = self.OOV_emb.weight * self.mask_vec_list
            Word_emb = (self.a * Word_emb + self.b) * (1 - self.mask_vec_list)
            Word_emb = self.emb_dropout(Word_emb + OOV_wordemb)  # * self.alpha.weight.reshape(-1)
            return Word_emb + sw_emb
        else:
            return self.emb_dropout(Word_emb) + sw_emb

    def sum_embedding(self, emb):
        emb = torch.sum(emb, dim=1)
        return emb

    def CNN_embedding(self, emb, lang):
        #V, max_n_sw, emb_size = emb.size()
        emb = self.CNN(emb.transpose(1, 2)).transpose(1, 2)  # V, max_n_sw, sw_size
        emb = emb * self.CNNmask[lang].to(emb.device)
        emb = torch.sum(emb,dim=1)
        return emb

class Shared_MT(nn.Module):
    def __init__(self, encoder, decoder, vocab_dict, emb_size, swemb_size, share_vocab, subword_file, dr_rate, pretrained_emb):
        super().__init__()
        self.encoder = encoder
        if encoder == None:
            self.is_unsupervised = True
        else:
            self.is_unsupervised = False
        self.cross_entropy = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
        self.decoder = decoder
        self.emb_dropout = nn.Dropout(p=dr_rate)
        self.lang_size = len(vocab_dict.V_size)
        self.emb_size = emb_size
        self.swemb_size = swemb_size
        self.vocab2id = vocab_dict.vocab2id
        self.id2vocab = vocab_dict.id2vocab
        self.init_Embeddings(vocab_dict, share_vocab, subword_file, dr_rate,pretrained_emb)

    def init_Embeddings(self, vocab_dict, share_vocab, subword_file, dr_rate, pretrained_emb):
        if share_vocab == 0:
            logger.info("word embedding")
            self.embedding_weight = Word_Embedding(self.lang_size, vocab_dict, self.emb_size, dr_rate, pretrained_emb)

        elif share_vocab in [3, 4]:
            logger.info("subword embedding")
            self.embedding_weight = SubWord_Embedding(self.lang_size, vocab_dict, self.emb_size, share_vocab, subword_file, dr_rate,pretrained_emb)

    def encode(self, s_id, s_lengths):
        s_id_emb, _ = self.embedding(self.torch.LongTensor(s_id), self.src_lang)  # bs, s_len, emb
        hs, hs_list = self.encoder(s_id_emb, s_lengths)
        return s_id_emb, hs, hs_list

    def decode(self, t_id, hs, s_lengths, s_id_emb):
        t_id = self.torch.LongTensor(t_id)
        t_id_emb, embedding_weight = self.embedding(t_id, self.tgt_lang)
        ht, attn_matix = self.decoder._decode(t_id_emb, hs, s_lengths, s_id_emb)
        score_sep = self.Output_layer(ht, embedding_weight)
        return score_sep, t_id_emb, ht.view(len(t_id_emb),len(t_id_emb[0]),-1)

    def decode_LM(self, t_id, *args):
        t_id_emb, embedding_weight = self.embedding(self.torch.LongTensor(t_id), self.tgt_lang)
        ht = self.decoder._decode_LM(t_id_emb)
        score = self.Output_layer(ht, embedding_weight)
        return score, t_id_emb

    def forward(self, s_id, s_lengths, t_id, t_lengths, t_id_EOS):
        # only used for multi-GPU training
        # all inputs need to be in GPU
        if s_id != None:
            s_id_emb, hs, hs_list = self.encode(s_id, s_lengths.cpu().numpy())
            score, _, _ = self.decode(t_id, hs, s_lengths, s_id_emb)
        else:
            score, _ = self.decode_LM(t_id)
        loss = self.cross_entropy(score, t_id_EOS.view(-1))  # (bs * maxlen_t,)
        return loss.sum().unsqueeze(0)

    def word_align(self, cossim, threshold):
        cossim_sorted = -1 * np.sort(- cossim, axis=1)  # s_len, t_len
        cossim_sorted_T = -1 * np.sort(- cossim.T, axis=1)  # t_len, s_len
        K = 3
        rT = np.array([np.mean(x[0:K]) for x in cossim_sorted]).reshape(-1, 1)  # x,1
        rS = np.array([np.mean(x[0:K]) for x in cossim_sorted_T]).reshape(1, -1)  # 1, y
        CSLS_score = 2 * cossim - np.broadcast_to(rT, cossim.shape) - np.broadcast_to(rS, cossim.shape)
        max_list = []
        for i in range(len(CSLS_score)):
            maxidx = np.argmax(CSLS_score[i])
            max_score = CSLS_score[i][maxidx]
            if max_score > threshold:
                max_list.append([i, maxidx])  # s_len, N
        return max_list

    def Encoder_align(self, src_lang, tgt_lang, s_id, s_lengths, t_id, t_lengths, null_align):
        self.Set_SrcLang(src_lang)  # reverse src and tgt langs
        s_emb, hs, _ = self.encode(s_id, s_lengths)
        self.Set_SrcLang(tgt_lang)  # reverse src and tgt langs
        t_emb, ht, _ = self.encode(t_id, t_lengths)
        cos_similarity = cossim(hs[0], ht[0])
        if not null_align:
            max_list = self.word_align(cos_similarity, -100)
        else:
            cos_similarity_emb = cossim(s_emb[0], t_emb[0])
            max_list_tmp = self.word_align(cos_similarity, 0)
            src_BEOS = self.embedding_weight(src_lang)[2:3]  # 1, emb_size (BOS EMB)
            tgt_BEOS = self.embedding_weight(tgt_lang)[2:3]  # 1, emb_size (BOS EMB)
            cos_similarity_Semb_BEOS = cossim(s_emb[0], tgt_BEOS)
            cos_similarity_Temb_BEOS = cossim(t_emb[0], src_BEOS)
            max_list = []
            for l in range(len(max_list_tmp)):  # [[0,1],[0,2]]
                src_tgt = max_list_tmp[l]
                embsim = cos_similarity_emb[src_tgt[0], src_tgt[1]]
                thre = max(cos_similarity_Semb_BEOS[src_tgt[0]].max(), cos_similarity_Temb_BEOS[src_tgt[1]].max())
                if embsim > thre:
                    max_list.append(src_tgt)
                else:
                    max_list.append([src_tgt[0],-1])
        return max_list

    def Output_layer(self, ht, embedding_weight):
        score = ht.mm(embedding_weight[2:].t())  # omit <PAD> and <MASK>
        return score

    def embedding(self, s_id, lang):
        bs, max_len = s_id.size()
        Emb_Weight = self.embedding_weight(lang)[1:] #omit Padding emb
        Emb_Weight = torch.cat([self.torch.FloatTensor(np.zeros((1,self.emb_size))),Emb_Weight],dim =0) #concat padding emb
        s_id_emb = Emb_Weight.index_select(0, s_id.view(-1)).view(bs,max_len,-1)
        return s_id_emb, Emb_Weight

    def Set_SrcLang(self, src):
        self.src_lang = src

    def Set_TgtLang(self,tgt, direction):
        self.tgt_lang = tgt
        self.direction = direction
        self.tgt_lang_class = self.lang_class[self.tgt_lang]
        self.decoder.Set_TgtLang(direction,self.tgt_lang_class) #decoder type

    def Set_lang_class(self,lang_class):
        assert len(lang_class)== self.lang_size
        self.lang_class = lang_class

    def set_device(self, is_cuda):
        if is_cuda:
            self.torch = torch.cuda
        else:
            self.torch = torch
        if not self.is_unsupervised:
            self.encoder.torch =  self.torch
        self.decoder.torch =  self.torch

class Shared_Decoder(nn.Module):

    def __init__(self, class_number, n_layer, emb_size, h_size, dr_rate, vocab_dict, is_unsupervised):
        super().__init__()
        self.lang_size  = len(vocab_dict.id2vocab)
        self.dr_rate = dr_rate
        self.emb_dropout = nn.Dropout(p=dr_rate)
        self.h_size = h_size
        self.n_layer = n_layer
        if is_unsupervised:
            layer = []
            for _ in range(2): #forward and backward decoders
                layer.append(nn.LSTM(
                    input_size=emb_size,
                    hidden_size=h_size,
                    num_layers=n_layer,
                    batch_first=True
                ))
            self.LM_LSTM = nn.ModuleList(layer)
        else:
            layer = []
            for _ in range(class_number):
                layer.append(nn.LSTM(
                input_size=emb_size,
                hidden_size=h_size,
                num_layers=n_layer,
                batch_first=True
                ))
            self.lstm = nn.ModuleList(layer)
        self.W_emb_h = nn.Linear(emb_size, h_size, bias=False)
    #t_id_emb, t_lengths, hs, h_last, c_last, s_lengths, s_id_emb
    def _decode(self, t_id_emb, hs, s_lengths, s_id_emb):
        self.lstm[self.lstm_class].flatten_parameters()
        ht, (_, _) = self.lstm[self.lstm_class](t_id_emb)
        context_vector_sep, attn_matix = self.attention(hs, s_lengths, ht)
        context_vector_emb = torch.bmm(attn_matix, s_id_emb)  # (bt *  maxlen_t * maxlen_s) * (#bt * maxlen_s * demb) = bt * maxlen_t * demb
        all_h = context_vector_sep + ht + context_vector_emb
        ht_attn = self.emb_dropout(all_h)
        ht_attn = ht_attn.view(-1, self.h_size)  # bs*s_len, h_size
        ht_attn = ht_attn.mm(self.W_emb_h.weight)
        return ht_attn, attn_matix

    def _decode_LM(self, input_id_emb):
        self.LM_LSTM[self.direction].flatten_parameters()
        ht, (_, _) = self.LM_LSTM[self.direction](input_id_emb)  # ht: bt * len_t * demb(最上位層の出力 from each hidden state)
        ht = self.emb_dropout(ht)
        ht = ht.contiguous().view(-1, ht.size()[-1])
        ht = ht.mm(self.W_emb_h.weight)
        return ht.contiguous()

    def attention(self, hs_padded, s_length, ht_padded):
        #hs_padded: bs, seqlen, hsize
        #(ht_padded, mask): bs, seqlen, hsize
        hs_swap = hs_padded.transpose(1, 2)  # bt *  demb * maxlen_s
        attn_matix = torch.bmm(ht_padded, hs_swap)  # bt *  maxlen_t * maxlen_s
        # create mask based on the sentence lengths
        mask = np.zeros(attn_matix.size())
        for i, l in enumerate(s_length): #for each batch
            mask[i, :, l:] = 1
        attn_matix.masked_fill_(self.torch.BoolTensor(mask), - float('inf')) #GPU
        attn_matix_sm = F.softmax(attn_matix, dim=2)  # bt *  maxlen_t * maxlen_s
        context_vector = torch.bmm(attn_matix_sm, hs_padded)  # (bt *  maxlen_t * maxlen_s) * (#bt * maxlen_s * demb) = bt * maxlen_t * demb
        return context_vector, attn_matix_sm

    def Register_vocab(self,vocab2id,id2vocab):
        self.vocab2id = vocab2id
        self.id2vocab = id2vocab

    def Set_TgtLang(self, direction, tgt_lang_class):
        # self.Append_BOS = self.Append_BOS
        self.direction = direction
        self.tgt_lang_class = tgt_lang_class
        self.lstm_class = 2 * self.tgt_lang_class + direction

class Shared_Encoder(nn.Module):
    def __init__(self, n_layer, emb_size, h_size, dr_rate, vocab_dict, *args):
        super().__init__()
        if n_layer==1:
            self.shared_lstm = nn.LSTM(
                input_size=emb_size,
                hidden_size=int(h_size / 2),
                num_layers=n_layer,
                batch_first=True,
                bidirectional=True
            )
        else:
            layer = []
            for _ in range(n_layer):
                layer.append(nn.LSTM(
                input_size=emb_size,
                hidden_size=int(h_size / 2),
                num_layers=1,
                batch_first=True,# ,dropout=dr_rate
                bidirectional=True))
            self.shared_lstm = nn.ModuleList(layer)
        self.n_layer = n_layer
        self.lang_size = len(vocab_dict.vocab2id)
        self.emb_dropout = nn.Dropout(p=dr_rate)

    def __call__(self, s_id_emb, s_lengths, *args):
        lstm_input = pack(s_id_emb, batch_first=True, lengths=s_lengths)
        if self.n_layer == 1:
            self.shared_lstm.flatten_parameters()
            hs, (h_last, c_last) = self.shared_lstm(lstm_input)
            hs, _ = unpack(hs, batch_first=True, total_length=s_id_emb.size()[1])  # (bs, s_len, demb)
            hs_list = hs.unsqueeze(0) #
        else:
            hs_list = []
            self.shared_lstm[0].flatten_parameters()
            hs_next, (h_last, c_last) = self.shared_lstm[0](lstm_input)
            hs, _ = unpack(hs_next, batch_first=True, total_length=s_id_emb.size()[1])  # (bs, s_len, demb)
            hs_list.append(hs.unsqueeze(0))
            for i in range(1, self.n_layer):
                self.shared_lstm[i].flatten_parameters()
                hs_next, (h_last, c_last) = self.shared_lstm[i](hs_next)
                hs, _ = unpack(hs_next, batch_first=True, total_length=s_id_emb.size()[1])  # (bs, s_len, demb)
                hs_list.append(hs.unsqueeze(0))
            hs_list = torch.cat(hs_list,dim=0) #N_layer, bs, s_len, dim
        _, bs, h_size = h_last.shape
        return hs, hs_list
