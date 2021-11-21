import os
import io
from logging import getLogger, StreamHandler, FileHandler, basicConfig, INFO
basicConfig(level = INFO)
import numpy as np
import torch
logger = getLogger('Log')
handler = StreamHandler()
logger.setLevel(INFO)
handler.setLevel(INFO)
logger.addHandler(handler)
def get_nn_avg_dist(emb, query, knn):
    """
    Compute the average distance of the `knn` nearest neighbors
    for a given set of embeddings and queries.
    Use Faiss if available.
    """
    bs = 1024
    all_distances = []
    emb = emb.transpose(0, 1).contiguous()
    for i in range(0, query.shape[0], bs):
        distances = query[i:i + bs].mm(emb)
        best_distances, _ = distances.topk(knn, dim=1, largest=True, sorted=True)
        all_distances.append(best_distances.mean(1).cpu())
    all_distances = torch.cat(all_distances)
    return all_distances.numpy()

def Extract_vocab_emb(file, vocab, logging = True):
    vocab2id = {}
    id2vocab = {}
    vocab2emb = {}
    vocab_list = []
    words =[]
    emb =[]
    for line in open(vocab,encoding="utf-8"):
        vocab_list.append(line.split()[0])

    first_line = True
    for line in open(file,encoding="utf-8"):
        word  = line.split()[0]
        word_emb = np.array(line.split()[1:]).astype(np.float32)
        if first_line:
            words.append(word)
            emb.append(word_emb)
            first_line = False
        elif word in vocab_list:
            words.append(word)
            emb.append(word_emb)

    assert len(emb[0]) == 1
    if logging:
        print (len(words))
        print (len(vocab_list) + 1)
        if len(vocab_list) + 1 != len(words):
            print (set(vocab_list).difference(set(words)))

    assert len(vocab_list) + 1 == len(words)

    for i in range(1,len(words)):
        vocab2id[words[i]] = i - 1
        id2vocab[i-1]= words[i]
        vocab2emb[words[i]] = emb[i]

    dim = int(emb[0])
    if logging:
        print("Vocab", words[0])
        print("dim", dim)

    return vocab2id, id2vocab, vocab2emb, emb[1:], dim

def load_dictionary(path, word2id1, word2id2):
    """
    Return a torch tensor of size (n, 2) where n is the size of the
    loader dictionary, and sort it by source word frequency.
    """
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

    logger.info("Found %i pairs of words in the dictionary (%i unique). "
                "%i other pairs contained at least one unknown word "
                "(%i in lang1, %i in lang2)"
                % (len(pairs), len(set([x for x, _ in pairs])),
                   not_found, not_found1, not_found2))

    # sort the dictionary by source word frequencies
    pairs = sorted(pairs, key=lambda x: word2id1[x[0]])
    dico = torch.LongTensor(len(pairs), 2)
    for i, (word1, word2) in enumerate(pairs):
        dico[i, 0] = word2id1[word1]
        dico[i, 1] = word2id2[word2]

    return dico

def get_nearest_words(emb1, emb2, dico):
    method = 'csls'
    K = 5
    assert dico[:, 0].max() < emb1.size(0)
    assert dico[:, 1].max() < emb2.size(0)

    # normalize word embeddings
    emb1 = emb1 / emb1.norm(2, 1, keepdim=True).expand_as(emb1)
    emb2 = emb2 / emb2.norm(2, 1, keepdim=True).expand_as(emb2)

    # nearest neighbors
    if method == 'nn':
        query = emb1[dico[:, 0]]
        scores = query.mm(emb2.transpose(0, 1))

    # contextual dissimilarity measure
    elif method == 'csls':
        # average distances to k nearest neighbors
        knn = 10
        average_dist1 = get_nn_avg_dist(emb2, emb1, knn)
        average_dist2 = get_nn_avg_dist(emb1, emb2, knn)
        average_dist1 = torch.from_numpy(average_dist1).type_as(emb1)
        average_dist2 = torch.from_numpy(average_dist2).type_as(emb2)
        # queries / scores
        query = emb1[dico[:, 0]]
        scores = query.mm(emb2.transpose(0, 1))
        scores.mul_(2)
        scores.sub_(average_dist1[dico[:, 0]][:, None])
        scores.sub_(average_dist2[None, :])

    top_matches = scores.topk(K, 1, True)[1] ##dico, 5
    return top_matches##dico, k

def get_word_translation_accuracy(dico, top_matches):
    #top_matches : dico, k
    results  = []
    top_matches = torch.LongTensor(top_matches)
    for k in [1, 5]:
        top_k_matches = top_matches[:, :k] #dico, k
        _matching = (top_k_matches == dico[:, 1][:, None].expand_as(top_k_matches)).sum(1).cpu().numpy()
        # dico
        # allow for multiple possible translations
        matching = {}
        for i, src_id in enumerate(dico[:, 0].cpu().numpy()):
            matching[src_id] = min(matching.get(src_id, 0) + _matching[i], 1)
        # evaluate precision@k
        precision_at_k = round(100 * np.mean(list(matching.values())),1)

        logger.info("%i source words - csls - Precision at k = %i: %f" %
                    (len(matching), k, precision_at_k))

        results.append([k,precision_at_k])

    return results

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-src',
        type=str,
        required=True,
        help='source train data path')
    parser.add_argument(
        '-srcV',
        type=str,
        required=True,
        help='source train data path')
    parser.add_argument(
        '-tgt',
        type=str,
        required=True,
        help='source train data path')
    parser.add_argument(
        '-tgtV',
        type=str,
        required=True,
        help='source train data path')
    parser.add_argument(
        '-dict',
        type=str,
        required=True,
        help='source train data path')
    parser.add_argument(
        '-save',
        type=str,
        required=True,
        help='source train data path')

    opt = parser.parse_args()
    vocab2id_src, id2vocab_src, vocab2emb_src, emb1, dim1 = Extract_vocab_emb(opt.src, opt.srcV)
    vocab2id_tgt, id2vocab_tgt, vocab2emb_tgt, emb2, dim2 = Extract_vocab_emb(opt.tgt, opt.tgtV)
    assert dim1 == dim2

    emb1=torch.FloatTensor(emb1)
    emb2=torch.FloatTensor(emb2)
    dico  = load_dictionary(opt.dict, vocab2id_src, vocab2id_tgt)
    dico1 = len(dico)
    dim11 = dim1
    emb11 = len(emb1)
    emb21 = len(emb2)
    with open(opt.save, 'w',encoding="utf-8") as f:
        f.write("dico: "+ str(len(dico)) + "\n")
        f.write("dim: "+ str(dim1) + "\n")
        f.write("V1: "+ str(len(emb1)) + "\n")
        f.write("V2: "+ str(len(emb2)) + "\n")
    dico = dico.cuda() if emb1.is_cuda else dico
    top_matches = get_nearest_words(emb1, emb2, dico)
    results = get_word_translation_accuracy(dico, top_matches)
    with open(opt.save, 'a',encoding="utf-8") as f:
        for i in range(len(results)):
            print(str(round(results[i][1], 1)))
            f.write("P" + str(results[i][0]) + ": " + str(round(results[i][1],1)) + "\n")

