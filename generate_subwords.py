import argparse
import sentencepiece as spm

parser = argparse.ArgumentParser()

parser.add_argument(
    '-vocab_file',
    type=str,
    required=True,
    help='vocab data path')

parser.add_argument(
    '-out',
    type=str,
    required=True,
    help='save path')

parser.add_argument(
    '-t',
    type=str,
    default=None,
    help='sentence piece model')

opt = parser.parse_args()

# V="1000"
# add_dummy="true"
# hard_vocab_limit="false"
# model = "model_name"
# input_text = '--input='+ input_file + ' --model_prefix='+model+' --vocab_size='+V +' --character_coverage=1.0 --add_dummy_prefix='+add_dummy + ' --hard_vocab_limit='+hard_vocab_limit
# spm.SentencePieceTrainer.Train(input_text)

sp = spm.SentencePieceProcessor()
sp.Load(opt.spm_model)
with open(opt.out, "w", encoding='utf8') as f:
    for line in open(opt.vocab_file, encoding='utf8'):
        word = line.strip('\n').split()[0]  #
        subwords = sp.EncodeAsPieces(word)
        subwords = ' '.join(subwords)
        f.write(word + " " + subwords + "\n")