import numpy as np
import warnings

def Extract_vocab_dict(file):
    src_word_list = []
    tgt_word_list = []
    for line in open(file):
        src_word_list.append(line.split()[0])
        tgt_word_list.append(line.split()[1])
    src_word_list = np.array(src_word_list)
    tgt_word_list = np.array(tgt_word_list)

    print("dict size",len(src_word_list))
    return src_word_list, tgt_word_list

def Extract_vocab_emb(file):
    vocab2emb = {}
    words =[]
    emb =[]
    for line in open(file,errors='ignore'):
        words.append(line.split()[0])
        emb.append(np.array(line.split()[1:]).astype(np.float32))

    for i in range(1,len(words)):
        vocab2emb[words[i]] = emb[i]

    dim=int(emb[0])
    print("Vocab",words[0])
    print("dim",dim)

    return vocab2emb,dim

def Calc_CSLS(src_emb_list, tgt_emb_list):
    src_emb_list = np.array(src_emb_list)
    tgt_emb_list = np.array(tgt_emb_list)
    src_norm = np.linalg.norm(src_emb_list, axis=1)  # src_len
    tgt_norm = np.linalg.norm(tgt_emb_list, axis=1)  # tgt_len
    src_emb_list = src_emb_list / src_norm.reshape(-1, 1)  # normalize
    tgt_emb_list = tgt_emb_list / tgt_norm.reshape(-1, 1)  # normalize

    cossim = np.matmul(src_emb_list, tgt_emb_list.T)  # src_len , tgt_len
    cossim_sorted = -1 * np.sort(-cossim, axis=1)  # src_len, tgt_len
    cossim_sorted_T = -1 * np.sort(-cossim.T, axis=1)  # tgt_len , src_len
    K = 10
    rT = np.array([np.mean(x[0:K]) for x in cossim_sorted]).reshape(-1, 1)  # src_len,1
    rS = np.array([np.mean(x[0:K]) for x in cossim_sorted_T]).reshape(1, -1)  # 1,tgt_len

    CSLS_score = 2 * cossim - np.broadcast_to(rT, cossim.shape) - np.broadcast_to(rS, cossim.shape)
    idx_sort_by_csls = np.argsort(-CSLS_score, axis=1)  # src_len * tgt_len
    return idx_sort_by_csls

def Eval_Matching(idx_sort_by_csls):
    #src_emb_list: src_data_len * demb
    #tgt_emb_list: tgt_data_len * demb

    acc_top10 = 0
    acc_top5 = 0
    acc_top1 = 0
    matrix_len = len(idx_sort_by_csls)
    for i in range(len(idx_sort_by_csls)):
        top10 = idx_sort_by_csls[i][0:10] #top 10 tgt_idx
        if (i in top10):
            acc_top10 += 1
            if (i in top10[0:5]):
                acc_top5 += 1
                if (i == top10[0]):
                    acc_top1 += 1
    acc_top10 = acc_top10 / matrix_len
    acc_top5 = acc_top5 / matrix_len
    acc_top1 = acc_top1 / matrix_len

    return str(acc_top1),str(acc_top5),str(acc_top10)

def Matching(dict_data, src_vocab2emb, tgt_vocab2emb):

    src_word_list, tgt_word_list = Extract_vocab_dict(dict_data)

    src_emb_list = []
    tgt_emb_list = []
    assert len(tgt_word_list) == len(src_word_list)
    word_dim = 300
    UNK_count = 0
    for i in range(len(src_word_list)):
        if (src_word_list[i] in src_vocab2emb and tgt_word_list[i] in tgt_vocab2emb):
            src_emb = np.array(src_vocab2emb[src_word_list[i]])
            tgt_emb = np.array(tgt_vocab2emb[tgt_word_list[i]])
            src_emb_list.append(src_emb)
            tgt_emb_list.append(tgt_emb)
        else:  #if a word in a dictionary is unknown
            UNK_count += 1

    if UNK_count != 0:
        print("number of unknown words:", UNK_count)
        warnings.warn("UNK words are included in a dictionary")
    print("align " + str(len(src_word_list)-UNK_count) + " pairs of words")
    idx_sort_by_csls = Calc_CSLS(src_emb_list, tgt_emb_list)
    return Eval_Matching(idx_sort_by_csls)



