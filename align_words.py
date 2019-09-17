import argparse
from utils.matching_func import Matching,Extract_vocab_emb
import os

parser = argparse.ArgumentParser()
parser.add_argument(
    '-dict',
    type=str,
    help='dicrionary data path')

parser.add_argument(
    '-src',
    type=str,
    help='source embedding path')
parser.add_argument(
    '-tgt',
    type=str,
    help='target embedding path')

parser.add_argument(
    '-save',
    type=str,
    help='')


if __name__ == '__main__':
    opt = parser.parse_args()
    if (os.path.isfile(opt.src) and os.path.isfile(opt.tgt)):
        print('src path: ' + opt.src)
        print('tgt path: ' + opt.tgt)
    else:
        raise Exception('invalid path')
    src_vocab2emb, srcdim = Extract_vocab_emb(opt.src)
    tgt_vocab2emb, tgtdim = Extract_vocab_emb(opt.tgt)

    if srcdim != tgtdim:
        raise Exception("word embbedding size is different between source and target")

    p1, p5, p10= Matching(opt.dict, src_vocab2emb, tgt_vocab2emb)

    print("P@1 " + p1)
    print("P@5 " + p5)
    print("P@10 " + p10)

    with open(opt.save, "w") as f:
        f.write("P@1 " + p1+ "\n")
        f.write("P@5 " + p5+ "\n")
        f.write("P@10 " + p10+ "\n")
        