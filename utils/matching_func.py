import numpy as np
import warnings
from logging import getLogger, StreamHandler, FileHandler, basicConfig, INFO
basicConfig(level = INFO)
import torch
import os
import io
logger = getLogger("Log").getChild("sub")

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
    eps = 0.0000001
    src_norm = np.linalg.norm(src_emb_list, axis=1) +eps # src_len
    tgt_norm = np.linalg.norm(tgt_emb_list, axis=1) +eps  # tgt_len
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

def CSLS_topk(k, emb1, emb2, average_dist1, average_dist2, src_idx):
    query = emb1[src_idx]  # dico, emb
    bs = 128
    scores = []
    cossim = []
    for i in range(0, len(emb2), bs):
        scores_tmp = query.mm(emb2[i:i + bs].transpose(0, 1))  # dico, bs
        cossim.append(scores_tmp)
        scores_tmp.mul_(2)
        scores_tmp.sub_(average_dist1[src_idx][:, None])  # dico,1
        scores_tmp.sub_(average_dist2[None, i:i + bs])  # 1, bs
        scores.append(scores_tmp)  # dico, bs
    scores = torch.cat(scores, dim=1)  # dico, V
    cossim = torch.cat(cossim, dim=1)  # dico, V
    top_matches = scores.topk(k, 1, True)[1]  # dico, 10
    cossim = [cossim[i,top_matches[i]] for i in range(len(top_matches))]
    return top_matches, cossim

def Cossim_topk(k, emb1, emb2, average_dist1, average_dist2, src_idx):
    query = emb1[src_idx]  # dico, emb
    bs = 128
    scores = []
    cossim = []
    for i in range(0, len(emb2), bs):
        scores_tmp = query.mm(emb2[i:i + bs].transpose(0, 1))  # dico, bs
        cossim.append(scores_tmp)
        scores.append(scores_tmp)  # dico, bs

    scores = torch.cat(scores, dim=1)  # dico, V
    cossim = torch.cat(cossim, dim=1)  # dico, V
    top_matches = scores.topk(k, 1, True)[1]  # dico, 10
    cossim = [cossim[i,top_matches[i]] for i in range(len(top_matches))]
    return top_matches, cossim

def MUSE_get_word_translation_accuracy(dico, emb1, emb2, method = "csls"):
    """
    Given source and target word embeddings, and a dictionary,
    evaluate the translation accuracy using the precision@k.
    based on code at https://github.com/facebookresearch/MUSE
    """
    dico = dico.cuda() if emb1.is_cuda else dico

    assert dico[:, 0].max() < emb1.size(0)
    assert dico[:, 1].max() < emb2.size(0)

    # normalize word embeddings
    emb1 = emb1 / emb1.norm(2, 1, keepdim=True).expand_as(emb1) #V, emb
    emb2 = emb2 / emb2.norm(2, 1, keepdim=True).expand_as(emb2)#V, emb

    # nearest neighbors
    if method == 'nn':
        print("nearest neighbors")
        query = emb1[dico[:, 0]]
        scores = query.mm(emb2.transpose(0, 1)) #V,V
    # contextual dissimilarity measure
    elif method == 'csls':
        print("csls neighbors")
    # average distances to k nearest neighbors
        knn = 10
        average_dist1 = get_nn_avg_dist(emb2, emb1, knn) #emb_V1
        average_dist2 = get_nn_avg_dist(emb1, emb2, knn) #emb_V2
        average_dist1 = torch.from_numpy(average_dist1).type_as(emb1)
        average_dist2 = torch.from_numpy(average_dist2).type_as(emb2)
        # queries / scores
        top_matches, _ = CSLS_topk(5, emb1, emb2, average_dist1, average_dist2, dico[:, 0])

    results = []
    # print(top_1_matches)
    #dico: dico_size, 2
    for k in [1,5]:#1, 5, 10
        top_k_matches = top_matches[:, :k] ##dico, k
        _matching = (top_k_matches == dico[:, 1][:, None].expand_as(top_k_matches)).sum(1).cpu().numpy()
        # allow for multiple possible translations
        matching = {}
        for i, src_id in enumerate(dico[:, 0].cpu().numpy()):
            matching[src_id] = min(matching.get(src_id, 0) + _matching[i], 1)
        # evaluate precision@k
        precision_at_k = 100 * np.mean(list(matching.values()))
        logger.critical("%i source words - %s - Eval Dict Precision at k = %i: %f" %
                    (len(matching), 'csls', k, precision_at_k))
        results.append(precision_at_k)

    return results

def Get_CSLS(emb1, emb2, k = 10):

    # normalize word embeddings
    emb1 = emb1 / emb1.norm(2, 1, keepdim=True).expand_as(emb1) #V, emb
    emb2 = emb2 / emb2.norm(2, 1, keepdim=True).expand_as(emb2)#V, emb

    knn = 10
    #get_nn_avg_dist: query, emb, K
    average_dist1 = get_nn_avg_dist(emb2, emb1, knn) #embV1
    average_dist2 = get_nn_avg_dist(emb1, emb2, knn) #embV2
    average_dist1 = torch.from_numpy(average_dist1).type_as(emb1)
    average_dist2 = torch.from_numpy(average_dist2).type_as(emb2)
    # queries / scores
    top_matches, cossim = CSLS_topk(k, emb1, emb2, average_dist1, average_dist2, np.arange(len(emb1)))

    return top_matches, cossim #|emb1|, 10

def Get_Cossim(emb1, emb2, k = 10):

    # normalize word embeddings
    emb1 = emb1 / emb1.norm(2, 1, keepdim=True).expand_as(emb1) #V1, emb
    emb2 = emb2 / emb2.norm(2, 1, keepdim=True).expand_as(emb2)#V2, emb
    cossim = emb1.mm(emb2.transpose(0, 1))  # V1, V2
    sortidx = emb1.topk(k, 1, True)[1].data.tolist() #N_words, 5
    logger.info(sortidx)
    top_matches = []
    for i in range(len(sortidx)):
        top_matches.append([x for x in sortidx[i] if cossim[i][x] > 0.3])
    logger.info(np.array(top_matches))
    return np.array(top_matches)#|emb1|, 10



def Word_Translation(N_words, emb1, emb2, method = "csls"):
    """
    Given source and target word embeddings, and a dictionary,
    evaluate the translation accuracy using the precision@k.
    """

    # normalize word embeddings
    emb1 = emb1 / emb1.norm(2, 1, keepdim=True).expand_as(emb1) #V, emb
    emb2 = emb2 / emb2.norm(2, 1, keepdim=True).expand_as(emb2)#V, emb

    # nearest neighbors
    if method == 'nn':
        print("nearest neighbors")
        query = emb1[:N_words]#N_words,V
        scores = query.mm(emb2.transpose(0, 1)) #N_words, V
    # contextual dissimilarity measure
    elif method == 'csls':
        print("csls neighbors")
    # average distances to k nearest neighbors
        knn = 10
        emb1 = emb1[: N_words] #N_words, emb
        #get_nn_avg_dist: query, emb, K
        average_dist1 = get_nn_avg_dist(emb2, emb1, knn) #N_words
        average_dist2 = get_nn_avg_dist(emb1, emb2, knn) #embV2
        average_dist1 = torch.from_numpy(average_dist1).type_as(emb1)
        average_dist2 = torch.from_numpy(average_dist2).type_as(emb2)
        # queries / scores
        query = emb1 #N_words, emb
        scores = query.mm(emb2.transpose(0, 1)) #N_words, embV2
        scores.mul_(2)
        scores.sub_(average_dist1[:, None])
        scores.sub_(average_dist2[None, :])

    results = []
    top_matches = scores.topk(5, 1, True)[1] #N_words, 5

    return top_matches

def get_nn_avg_dist(emb, query, knn):
    """
    Compute the average distance of the `knn` nearest neighbors
    for a given set of embeddings and queries.
    Use Faiss if available.
    """
    # if FAISS_AVAILABLE:
    #     emb = emb.cpu().numpy()
    #     query = query.cpu().numpy()
    #     if hasattr(faiss, 'StandardGpuResources'):
    #         # gpu mode
    #         res = faiss.StandardGpuResources()
    #         config = faiss.GpuIndexFlatConfig()
    #         config.device = 0
    #         index = faiss.GpuIndexFlatIP(res, emb.shape[1], config)
    #     else:
    #         # cpu mode
    #         index = faiss.IndexFlatIP(emb.shape[1])
    #     index.add(emb)
    #     distances, _ = index.search(query, knn)
    #     return distances.mean(1)
    # else:
    bs = 1024
    all_distances = []
    emb = emb.transpose(0, 1).contiguous() #emb, V
    for i in range(0, query.shape[0], bs):
        distances = query[i:i + bs].mm(emb) #bs, V
        best_distances, _ = distances.topk(knn, dim=1, largest=True, sorted=True) #bs,
        all_distances.append(best_distances.mean(1).cpu())
    all_distances = torch.cat(all_distances) #query_N
    return all_distances.numpy()

def load_dictionary(path, word2id1, word2id2, init_idx = 0):
    assert os.path.isfile(path)
    pairs = []
    not_found = 0
    not_found1 = 0
    not_found2 = 0
    with io.open(path, 'r', encoding='utf-8') as f:
        for index, line in enumerate(f):
            assert line == line.lower()
            parts = line.rstrip().split()
            if len(parts) < 2:
                logger.warning("Could not parse line %s (%i)", line, index)
                continue
            word1, word2 = parts
            if word1 in word2id1 and word2 in word2id2:
                pairs.append((word1, word2))
            else:
                not_found += 1
                not_found1 += int(word1 not in word2id1)
                not_found2 += int(word2 not in word2id2)

    logger.critical("Found %i pairs of words in the dictionary (%i unique). "
                "%i other pairs contained at least one unknown word "
                "(%i in lang1, %i in lang2)"
                % (len(pairs), len(set([x for x, _ in pairs])),
                   not_found, not_found1, not_found2))
    # sort the dictionary by source word frequencies
    pairs = sorted(pairs, key=lambda x: word2id1[x[0]])
    dico = torch.LongTensor(len(pairs), 2) #
    for i, (word1, word2) in enumerate(pairs):
        dico[i, 0] = word2id1[word1] - init_idx
        dico[i, 1] = word2id2[word2] - init_idx
    return dico


def get_candidates(emb1, emb2, N_words, method = 'csls', dico_max_size =0 , dico_threshold = 0):
    """
    Get best translation pairs candidates.
    """
    bs = 128

    all_scores = []
    all_targets = []

    # number of source words to consider
    n_src = emb1.size(0)
    if N_words > 0:
        n_src = min(N_words, n_src)

    # nearest neighbors
    if method == 'nn':

        # for every source word
        for i in range(0, n_src, bs):

            # compute target words scores
            scores = emb2.mm(emb1[i:min(n_src, i + bs)].transpose(0, 1)).transpose(0, 1)
            best_scores, best_targets = scores.topk(2, dim=1, largest=True, sorted=True)

            # update scores / potential targets
            all_scores.append(best_scores.cpu())
            all_targets.append(best_targets.cpu())

        all_scores = torch.cat(all_scores, 0)
        all_targets = torch.cat(all_targets, 0)

    # # inverted softmax
    # elif params.dico_method.startswith('invsm_beta_'):
    #
    #     beta = float(params.dico_method[len('invsm_beta_'):])
    #
    #     # for every target word
    #     for i in range(0, emb2.size(0), bs):
    #
    #         # compute source words scores
    #         scores = emb1.mm(emb2[i:i + bs].transpose(0, 1))
    #         scores.mul_(beta).exp_()
    #         scores.div_(scores.sum(0, keepdim=True).expand_as(scores))
    #
    #         best_scores, best_targets = scores.topk(2, dim=1, largest=True, sorted=True)
    #
    #         # update scores / potential targets
    #         all_scores.append(best_scores.cpu())
    #         all_targets.append((best_targets + i).cpu())
    #
    #     all_scores = torch.cat(all_scores, 1)
    #     all_targets = torch.cat(all_targets, 1)
    #
    #     all_scores, best_targets = all_scores.topk(2, dim=1, largest=True, sorted=True)
    #     all_targets = all_targets.gather(1, best_targets)

    # contextual dissimilarity measure
    elif method == 'csls':

    # knn = params.dico_method[len('csls_knn_'):]
    # assert knn.isdigit()
        knn = min(10, min(len(emb1),len(emb2)))

        # average distances to k nearest neighbors
        average_dist1 = torch.from_numpy(get_nn_avg_dist(emb2, emb1, knn))
        average_dist2 = torch.from_numpy(get_nn_avg_dist(emb1, emb2, knn))
        average_dist1 = average_dist1.type_as(emb1)
        average_dist2 = average_dist2.type_as(emb2)

        # for every source word
        for i in range(0, n_src, bs):

            # compute target words scores
            scores = emb2.mm(emb1[i:min(n_src, i + bs)].transpose(0, 1)).transpose(0, 1)
            # CSLS_score = 2*scores - average_dist1[i:min(n_src, i + bs)][:, None] - average_dist2[None, :]
            # CSLS_score = 2*scores - average_dist1[i:min(n_src, i + bs)][:, None] - average_dist2[None, :]
            scores.mul_(2)
            scores.sub_(average_dist1[i:min(n_src, i + bs)][:, None] + average_dist2[None, :])
            best_scores, best_targets = scores.topk(2, dim=1, largest=True, sorted=True)
            #best_scores: bs, 2
            #best_targets: bs, 2
            # best_scores = scores[best_targets]
            # update scores / potential targets
            all_scores.append(best_scores.cpu())
            all_targets.append(best_targets.cpu())

        all_scores = torch.cat(all_scores, 0)
        all_targets = torch.cat(all_targets, 0)

    all_pairs = torch.cat([
        torch.arange(0, all_targets.size(0)).long().unsqueeze(1),
        all_targets[:, 0].unsqueeze(1)
    ], 1)

    # sanity check
    assert all_scores.size() == all_pairs.size() == (n_src, 2)

    # sort pairs by score confidence (by diff of scores between top1 and top2)
    diff = all_scores[:, 0] - all_scores[:, 1] #(n_src, 2)
    reordered = diff.sort(0, descending=True)[1]
    all_scores = all_scores[reordered]
    all_pairs = all_pairs[reordered] #n_src, 2

    # max dico words rank
    # if params.dico_max_rank > 0:
    #     selected = all_pairs.max(1)[0] <= params.dico_max_rank
    #     mask = selected.unsqueeze(1).expand_as(all_scores).clone()
    #     all_scores = all_scores.masked_select(mask).view(-1, 2)
    #     all_pairs = all_pairs.masked_select(mask).view(-1, 2)
    #
    # max dico size
    if dico_max_size > 0:
        all_scores = all_scores[:dico_max_size]
        all_pairs = all_pairs[:dico_max_size]
    #
    # # min dico size
    # diff = all_scores[:, 0] - all_scores[:, 1]
    # if params.dico_min_size > 0:
    #     diff[:params.dico_min_size] = 1e9

    # confidence threshold
    if dico_threshold > 0:
        mask = all_scores[:, 0]  > dico_threshold
        logger.info("Selected %i / %i pairs above the confidence threshold." % (mask.sum(), diff.size(0)))
        mask = mask.unsqueeze(1).expand_as(all_pairs).clone()
        all_pairs = all_pairs.masked_select(mask).view(-1, 2)

    return all_pairs

def build_dictionary(src_emb, tgt_emb, N_words = 0, method = 'csls', dico_max_size = 0, dico_threshold = 0, dico_top = 0):
    dico_build = 'S2T&T2S'
    """
    Build a training dictionary given current embeddings / mapping.
    """
    logger.info("Building the train dictionary ...")
    s2t = 'S2T' in dico_build
    t2s = 'T2S' in dico_build

    s2t_candidates = get_candidates(src_emb, tgt_emb, N_words, method, dico_max_size , dico_threshold) #n_src, 2
    t2s_candidates = get_candidates(tgt_emb, src_emb, N_words, method, dico_max_size, dico_threshold) #n_tgt, 2
    t2s_candidates = torch.cat([t2s_candidates[:, 1:], t2s_candidates[:, :1]], 1)

    # if dico_build == 'S2T':
    #     dico = s2t_candidates
    # elif dico_build == 'T2S':
    #     dico = t2s_candidates
    # else:
    s2t_candidates = set([(a, b) for a, b in s2t_candidates.numpy()])
    t2s_candidates = set([(a, b) for a, b in t2s_candidates.numpy()])
    # if dico_build == 'S2T|T2S':
    #         final_pairs = s2t_candidates | t2s_candidates
    #     else:
    #         assert dico_build == 'S2T&T2S'
    final_pairs = s2t_candidates & t2s_candidates #common, 2

    if len(final_pairs) == 0:
        logger.warning("Empty intersection ...")
        return [[],[]]
        # dico = torch.LongTensor(list([[int(a), int(b)] for (a, b) in final_pairs]))
    else:
        dico = np.array(list(final_pairs))
        if dico_top>0:
        # return dico[:, 0], dico[:, 1]
            emb1 =src_emb[dico[:,0]] #N_aligned, emb
            emb2 =tgt_emb[dico[:,1]] #N_aligned, emb
            scores = (emb1*emb2).sum(1).cpu().numpy() #N_aligned (cossim)
            idx = np.argsort(-1*scores)[:dico_top]
            return dico[idx,0], dico[idx,1]
        else:
            return dico[:, 0], dico[:, 1]
