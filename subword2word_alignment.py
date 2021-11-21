import argparse
import unicodedata
import pickle
import unicodedata
import pickle
from transformers import AutoTokenizer


def generate_subword2word(file):
    S_ID = -1
    idx_subword2word_list = []
    for line in open(file, encoding='utf8'):
        S_ID += 1
        line = line.strip('\n').split()
        idx_subword2word = {}
        wcount = -1
        for i, w in enumerate(line):
            if "▁" in w:
                wcount += 1
            idx_subword2word[i] = wcount
        idx_subword2word_list.append(idx_subword2word)
    return idx_subword2word_list

parser = argparse.ArgumentParser()
parser.add_argument(
    '-alignment',
    type=str,
    default=None,
    help='source train data path')

parser.add_argument(
    '-src',
    type=str,
    default=None,
    help='source train data path')

parser.add_argument(
    '-tgt',
    type=str,
    default=None,
    help='source train data path')

opt = parser.parse_args()
alignment_file = opt.alignment

srcidx_subword2word_list= generate_subword2word(opt.src)
tgtidx_subword2word_list= generate_subword2word(opt.tgt)

S_ID = -1
with open(alignment_file + ".walign", "w",encoding="utf-8") as f_new:
    for line in open(alignment_file, encoding='utf8'):
        S_ID += 1
        srcidx_subword2word = srcidx_subword2word_list[S_ID]
        tgtidx_subword2word = tgtidx_subword2word_list[S_ID]
        line = line.strip('\n').split() #0-1, 0-2,...
        new_line = set([])
        max_ink_idx = 0
        for e_i in line:
            e_i = e_i.split('-')
            assert len(e_i)==2
            e_wordidx = int(e_i[0])
            i_wordidx = int(e_i[1])
            e_wordidx = srcidx_subword2word[e_wordidx]
            i_wordidx = tgtidx_subword2word[i_wordidx]
            max_ink_idx = max(max_ink_idx, i_wordidx)
            new_line.add(str(e_wordidx)+"-"+str(i_wordidx))
        assert max_ink_idx <= len(tgtidx_subword2word)-1
        f_new.write(" ".join(list(new_line))+"\n")
