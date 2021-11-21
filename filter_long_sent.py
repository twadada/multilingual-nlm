import argparse
parser = argparse.ArgumentParser()
parser.add_argument(
    '-src',
    type=str,
    default=None)
parser.add_argument(
    '-tgt',
    type=str,
    default=None)
opt = parser.parse_args()

src = []
for line in open(opt.src, encoding = "utf8"):
    src.append(line.strip("\n"))
tgt = []
for line in open(opt.tgt, encoding = "utf8",errors = "ignore"):
    tgt.append(line.strip("\n"))

assert len(src) == len(tgt)

with open(opt.src+".filtered", "w", encoding='utf8') as f1:
    with open(opt.tgt+".filtered", "w", encoding='utf8') as f2:
            for i in range(len(src)):
                if src[i] != "" and tgt[i] !="":
                    if len(src[i].split())<=80 and len(tgt[i].split())<=80:
                        f1.write(src[i]+"\n")
                        f2.write(tgt[i]+"\n")
