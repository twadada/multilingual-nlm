import argparse
from itertools import groupby

def gb(collection):
    keyfunc = lambda x: x[0]
    groups = groupby(sorted(collection, key=keyfunc), keyfunc)
    return {k: set([v for k_, v in g]) for k, g in groups}

parser = argparse.ArgumentParser()
parser.add_argument(
    '-pred',
    type=str,
    help='source embedding path')

parser.add_argument(
    '-save',
    type=str,
    help='')

parser.add_argument(
    '-init_id',
    type=int,
    default=1)

parser.add_argument(
    '-R',
    action='store_true',
    help='backward training'
    )


parser.add_argument(
    '-parse',
    action='store_true',
    help='backward training'
    )

parser.add_argument(
    '-gold',
    type=str,
    default=None,
    help='backward training'
    )

parser.add_argument(
    '-line_by_line',
    action='store_true',
    help='backward training'
    )


if __name__ == '__main__':
    opt = parser.parse_args()
    print (opt.pred)
    if opt.line_by_line:
        poss_aligns = [[int(x) for x in l.strip().split()[:3]] for l in open(opt.gold).readlines()]
        sure_aligns = [[int(x) for x in l.strip().split()[:3]] for l in open(opt.gold).readlines() if l.strip().split()[-1] != "P"]
        if opt.R:
            poss_aligns = [(sid, twid, swid) for sid, swid, twid in poss_aligns]
            sure_aligns = [(sid, twid, swid) for sid, swid, twid in sure_aligns]
        poss_aligns = gb([(sid - opt.init_id, (swid - 1, twid - 1)) for sid, swid, twid in poss_aligns])
        sure_aligns = gb([(sid - opt.init_id, (swid - 1, twid - 1)) for sid, swid, twid in sure_aligns])
    else:
        sure_aligns = []
        poss_aligns = []
        for line in open(opt.gold, encoding="utf8"):
            line = line.rstrip('\n').split()  # 1-2, 3-4,..
            alignment = set()
            for j in range(len(line)):
                x, y = line[j].split('-')
                alignment.add((int(x), int(y)))
            sure_aligns.append(alignment)
            poss_aligns.append(alignment)
    AER=0
    P=0
    R=0
    pred_alignment_list = []
    for line in open(opt.pred, encoding="utf8"):
        line = line.rstrip('\n').split() #1-2, 3-4,..
        alignment = set()
        for j in range(len(line)):
            x, y = line[j].split('-')
            alignment.add((int(x), int(y)))
        pred_alignment_list.append(alignment)

    assert len(pred_alignment_list) == len(poss_aligns)

    size_a = 0.0
    size_s = 0.0
    size_a_and_s = 0.0
    size_a_and_p = 0.0
    for sid in range(len(sure_aligns)):  # for each batch
        alignment = pred_alignment_list[sid]
        sure = sure_aligns[sid]
        poss = poss_aligns[sid]
        size_a += float(len(alignment))
        size_s += float(len(sure))
        s_a = alignment & sure
        p_a = alignment & poss
        size_a_and_s += float(len(s_a))
        size_a_and_p += float(len(p_a))
    #print (float((size_a_and_s + size_a_and_p) / (size_a + size_s)))
    AER = round(100*float((size_a_and_s + size_a_and_p) / (size_a + size_s)),1)
    P = round(100*float(size_a_and_p / size_a),1)
    R = round(100*float((size_a_and_s) / size_s),1)
    print('Precision: {0:.1f}'.format(100 * size_a_and_p / size_a))
    print('Recall: {0:.1f}'.format(100 * (size_a_and_s) / size_s))
    print ('P',round(P,1))
    print ('R',round(R,1))
    print ('1-AER',round(AER,1))
    with open(opt.save + ".txt", "w") as f:
        f.write(str(round(AER,1)) + "\n")

