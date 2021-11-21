import os
import argparse
from utils.preprocess_func_new import load_vocab,load_corpus,shuffle_sort_para_data,shuffle_sort_mono_data, Read_Corpus,Dataset,oversampling,save_files,save_vocab

parser = argparse.ArgumentParser()
parser.add_argument(
    '-multi',
    type=str,
    nargs='+',
    default=None,
    help='multilingual data path; e.g. given parallel data en1-X, en2-Y and en3-Z, the input should be "en1 en2 en3 X Y Z"')

parser.add_argument(
    '-para',
    type=str,
    nargs='+',
    default=None,
    help='parallel data path')

parser.add_argument(
    '-mono',
    type=str,
    nargs='+',
    default=None,
    help='monolingual data path')

parser.add_argument(
    '-V_min_freq',
    nargs='+',
    type=int,
    help='minimum frequency of words in the vocabulary')

parser.add_argument(
    '-V',
    nargs='+',
    type=int,
    help='vocabulary size')

parser.add_argument(
    '-V_files',
    nargs='+',
    type=str,
    help='vocabulary file')

parser.add_argument(
    '-save_name',
    default="default",
    type=str,
    help='data name')

parser.add_argument(
    '-output_vocab',
    action='store_true',
    help='output vocabulary txt file'
)
parser.add_argument(
    '-max_dict_size',
    default=5000,
    type=int)
parser.add_argument(
    '-identical',
    action='store_true')


opt = parser.parse_args()

if __name__ == '__main__':
    N_data = 0
    current_dir = os.getcwd()
    if opt.para:
        opt.save_name+=".para"
    elif  opt.mono:
        opt.save_name += ".mono"
    elif  opt.multi:
        opt.save_name += ".multi"

    if not os.path.isdir("data"):
        os.mkdir("data")
    f = open("data/" + opt.save_name + "_inputs.txt", "w")
    f.write("data_dir: " + current_dir + "/data/" + opt.save_name + "\n")

    if opt.para: #only para data
        print("parallel data is given")
        para_corpus = Read_Corpus(opt.para)
        vocab_corpus = para_corpus
        N_lang = len(opt.para)
        for lang in range(N_lang):
            print("Para N sent: ", len(para_corpus[lang]))
            f.write("para train_file" + str(lang) + ": " + opt.para[lang] + "\n")

    elif opt.multi: #only mutli data
        print("multilingual data is given")
        assert len(opt.multi)%2==0
        multi_corpus = Read_Corpus(opt.multi)
        src_idx = int(len(multi_corpus)/2)
        vocab_corpus = []
        source_corpus = []
        for i in range(src_idx):
            source_corpus.extend(multi_corpus[i])
        vocab_corpus.append(source_corpus)
        for i in range(src_idx):
            vocab_corpus.append(multi_corpus[src_idx+i])
        N_lang = len(vocab_corpus)
        for lang in range(len(multi_corpus)):
            f.write("multi train_file" + str(lang) + ": " + opt.multi[lang] + "\n")
            print("multi N sent: ", len(multi_corpus[lang]))

    elif opt.mono: #only mono data
        print("monolingual data is given")
        mono_corpus = Read_Corpus(opt.mono)
        vocab_corpus = mono_corpus
        N_lang = len(opt.mono)
        for lang in range(N_lang):
            print("Mono N sent: ", len(mono_corpus[lang]))
            f.write("mono train_file" + str(lang) + ": " + opt.mono[lang] + "\n")
    else:
        assert Exception
    f.close()
    dataset = Dataset(N_lang)
    special_V = 5
    vocab_dict = load_vocab(opt, vocab_corpus, N_lang, opt.identical)
    if (opt.output_vocab):
        save_vocab(opt.save_name, vocab_corpus, vocab_dict)
    if opt.mono:
        sentence_num = max(len(x) for x in mono_corpus)
        lines_id, lengths = load_corpus(mono_corpus, vocab_dict.vocab2id)
        lines_id, lengths = shuffle_sort_mono_data(lines_id, lengths)
        lines_id, lengths = oversampling(lines_id, lengths)
        dataset.load_mono_data(lines_id, lengths, sentence_num)
    elif opt.para:
        sentence_num = len(para_corpus[0])
        assert all([sentence_num == len(x) for x in para_corpus[1:]])
        print ("loading corpus")
        lines_id, lengths = load_corpus(para_corpus, vocab_dict.vocab2id)
        lines_id, lengths = shuffle_sort_para_data(lines_id, lengths)
        dataset.load_para_data(lines_id, lengths, sentence_num)
    elif opt.multi:
        sentence_num =  max([len(x) for x in multi_corpus])
        lines_id_src, lengths_src = load_corpus(multi_corpus[:src_idx], [vocab_dict.vocab2id[0] for _ in range(src_idx)])
        lines_id_tgt, lengths_tgt = load_corpus(multi_corpus[src_idx:], vocab_dict.vocab2id[1:])
        lines_id = []
        lengths = []
        for pair in range(src_idx):
            lines_id_tmp, lengths_tmp = shuffle_sort_para_data([lines_id_src[pair], lines_id_tgt[pair]], [lengths_src[pair],lengths_tgt[pair]])
            rep = sentence_num // len(lines_id_tmp[0])
            remainder = sentence_num % len(lines_id_tmp[0])
            lines_id_src_oversmpl =  lines_id_tmp[0] * rep  + lines_id_tmp[0][:remainder]
            lines_id_tgt_oversmpl =  lines_id_tmp[1] * rep  + lines_id_tmp[1][:remainder]
            lengths_src_oversmpl  =  lengths_tmp[0] * rep  + lengths_tmp[0][:remainder]
            lengths_tgt_oversmpl   =  lengths_tmp[1] *rep  + lengths_tmp[1][:remainder]
            lines_id.append([lines_id_src_oversmpl, lines_id_tgt_oversmpl])
            lengths.append([lengths_src_oversmpl, lengths_tgt_oversmpl])
        dataset.load_multi_data(lines_id, lengths, sentence_num)
    save_files(opt.save_name, dataset, vocab_dict)

