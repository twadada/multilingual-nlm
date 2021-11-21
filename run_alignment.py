import argparse
from utils.preprocess_func_new import Convert_word2id,Read_Corpus
import numpy as np
import torch
from logging import getLogger, StreamHandler, INFO
from itertools import groupby
from utils.train_base_new import preprare_model, load_data, PAD_Sentences_Source

def gb(collection):
    keyfunc = lambda x: x[0]
    groups = groupby(sorted(collection, key=keyfunc), keyfunc)
    return {k: set([v for k_, v in g]) for k, g in groups}

parser = argparse.ArgumentParser()
parser.add_argument(
    '-model',
    type=str,
    help='model path')
parser.add_argument(
    '-src',
    type=str,
    help='source sentence path')
parser.add_argument(
    '-tgt',
    type=str,
    help='target sentence path')
parser.add_argument(
    '-GPU',
    action='store_true',
    help='use GPU')
parser.add_argument(
    '-src_lang',
    type=int,
    default=None,
    help='source lang id during training')
parser.add_argument(
    '-tgt_lang',
    type=int,
    default=None,
    help='target lang id during training')

parser.add_argument(
    '-save',
    type=str)

parser.add_argument(
    '-backward',
    action='store_true',
    help='backward alignment'
    )

parser.add_argument(
    '-null_align',
    action='store_true',
    help='generate NULL alignments for higher precision/lower recall'
    )

if __name__ == '__main__':
    opt = parser.parse_args()
    import pickle
    folder= opt.model.split("/")[0]
    with open(folder+"/options.pkl",'rb') as f:
        model_opt = pickle.load(f)
    file = open("data/"+ model_opt.data + ".vocab_dict", 'rb')
    vocab_dict = pickle.load(file)
    model_opt = vars(model_opt)
    for variable in model_opt: #read model options
        assert not hasattr(opt, variable)
        setattr(opt, variable, model_opt[variable])
    save_name = opt.save
    logger = getLogger('Log')
    handler = StreamHandler()
    logger.setLevel(INFO)
    handler.setLevel(INFO)
    logger.addHandler(handler)
    if opt.backward:
        src_lang, tgt_lang = opt.tgt_lang, opt.src_lang
        src, tgt = opt.tgt, opt.src
    else:
        src_lang, tgt_lang = opt.src_lang, opt.tgt_lang
        src, tgt = opt.src, opt.tgt
    dataset, vocab_dict = load_data("data/"+model_opt["data"],logger)
    model, dataset, vocab_dict = preprare_model(opt, dataset, vocab_dict, logger)
    model.load_state_dict(torch.load(opt.model, map_location=lambda storage, loc: storage))
    model.set_device(opt.GPU)
    if opt.GPU:
        model.to('cuda')
    srcvocab2id = vocab_dict.vocab2id[src_lang]
    tgtvocab2id = vocab_dict.vocab2id[tgt_lang]
    srcid2vocab = vocab_dict.id2vocab[src_lang]
    tgtid2vocab = vocab_dict.id2vocab[tgt_lang]
    test_corpus = Read_Corpus([src, tgt])
    src_lines_id, src_sent_length = Convert_word2id(test_corpus[0], srcvocab2id)
    src_sent_length = np.array(src_sent_length)
    tgt_lines_id, tgt_sent_length = Convert_word2id(test_corpus[1], tgtvocab2id)
    tgt_sent_length = np.array(tgt_sent_length)
    s_id_batch = []
    s_lengths_batch = []
    batch_size = 1
    model.eval()
    with torch.no_grad():
        with open(save_name + ".txt", "w",encoding="utf-8") as f:
            with open(save_name + ".words.txt", "w",encoding="utf-8") as f_word:
                for sid in range(0, len(src_sent_length), batch_size):  # for each batch
                    index = np.array(list(range(sid, sid + batch_size)))
                    s_lengths, s_id  = \
                        PAD_Sentences_Source(src_sent_length, src_lines_id, index, srcvocab2id["<PAD>"])
                    t_lengths, t_id = \
                        PAD_Sentences_Source(tgt_sent_length, tgt_lines_id, index, tgtvocab2id["<PAD>"])
                    align = model.Encoder_align(src_lang, tgt_lang, s_id, s_lengths, t_id, t_lengths, opt.null_align)
                    for k in range(batch_size):
                        for l in range(len(align)): #[[0,1],[0,2]]
                            src_tgt = align[l]
                            src_word = s_id[k][src_tgt[0]]
                            tgt_word = t_id[k][src_tgt[1]]
                            if src_tgt[1] == -1:
                                f_word.write(srcid2vocab[src_word] + "|||NULL" + " ")
                            else:
                                if opt.backward:
                                    f.write(str(src_tgt[1])+ "-" + str(src_tgt[0])  + " ")
                                else:
                                    f.write(str(src_tgt[0]) + "-" + str(src_tgt[1]) + " ")
                                f_word.write(srcid2vocab[src_word] + "|||" + tgtid2vocab[tgt_word] + " ")
                    f.write('\n')
                    f_word.write('\n')

