from collections import Counter
import argparse
import numpy as np
from tqdm import tqdm
from utils.preprocess_func_new import Read_Corpus

def gen_dice_dict(lines, save, max_dict_size, min_dice = 0.8, min_count = 3, max_sent_len = 2000000):
    assert len(lines)==2
    sent_len = min(len(lines[0]), max_sent_len)
    vocab = [{} for _ in range(2)]
    cooccur = {}
    for i in tqdm(range(sent_len)):
        words_list = []
        for lang in range(2):
            words = list(set(lines[lang][i]))
            for w in words:
                try:
                    vocab[lang][w] += 1
                except KeyError:
                    vocab[lang][w] = 1
            words_list.append(words)
        for src_w in words_list[0]:
            for tgt_w in words_list[1]:
                dictkey = tuple([src_w, tgt_w])
                try:
                    cooccur[dictkey] += 1
                except KeyError:
                    cooccur[dictkey] = 1
    print("Saving a file...")
    src_vocab2count = vocab[0]
    tgt_vocab2count = vocab[1]
    cooccur = {key: freq for key, freq in cooccur.items()
               if min_count <= freq
               and min_dice <= 2 * freq/(src_vocab2count[key[0]] + tgt_vocab2count[key[1]])}
    cooccur = sorted(cooccur.items(), key=lambda v: -1 * v[1])
    dict_size = 0
    with open(save, "w") as f:
        for x in cooccur: #x = ((src,tgt),count)
            srcword = x[0][0]
            tgtword = x[0][1]
            if dict_size >= max_dict_size:
                break
            else:
                dict_size += 1
                f.write(srcword + " " + tgtword+"\n")
        print("Dict Size: " + str(dict_size))

parser = argparse.ArgumentParser()

parser.add_argument(
    '-files',
    type=str,
    nargs='+',
    default=None,
    help='corpus data path')
parser.add_argument(
    '-save',
    type=str)

opt = parser.parse_args()
lines = Read_Corpus(opt.files)
shuf_idx = np.random.permutation(len(lines[0]))
lines_new = []
for lang in range(len(lines)):
    lines_new.append([lines[lang][j] for j in shuf_idx])

gen_dice_dict(lines_new, opt.save, max_dict_size = 5000, min_dice = 0.8, min_count = 3, max_sent_len = 3000000)

