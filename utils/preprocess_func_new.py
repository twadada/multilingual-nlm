from collections import Counter, OrderedDict
from itertools import chain
import warnings
import os
import numpy as np
from random import sample
import pickle

def Read_Corpus(file):
    lines_list =[]
    empty_index = [set() for _ in range(len(file))]
    for i in range(len(file)):
        lines = []
        idx = 0
        for line in open(file[i], encoding="utf-8",errors ="replace"):
            if len(line.strip('\n').split()) == 0:
                empty_index[i].add(idx)
            lines.append(line.strip('\n').split())
            idx += 1
        lines_list.append(lines)

    if any([len(x)>0 for x in empty_index]):
        for i in range(len(lines_list)):
            print("In " +str(file[i])+"," +" empty lines found at index " + " ".join(str(x) for x in list(empty_index[i])))
        raise Exception

    return lines_list

def read_conll(file, lower= False):
   root = (0,'*root*',-1,'rroot')
   tokens = [root]
   for line in open(file):
      if lower: line = line.lower()
      tok = line.strip().split()
      if not tok: #if empty line

         if len(tokens)>1: yield tokens
         tokens = [root]
      else:#tokens.append( (i, word_i, j, dep_ij) )
         try:
            tokens.append((int(tok[0]),tok[1],int(tok[-4]),tok[-3]))
         except ValueError:
            try: # when
               dep_idx = float(tok[-4])
               assert tok[-4][-2:] == ".1"
               dep_idx = round(dep_idx)
               print >> sys.stderr, tok
               tokens.append((int(tok[0]), tok[1], int(dep_idx), tok[-3]))
            except ValueError: #
               pass

   if len(tokens) > 1:
      yield tokens

def Read_CoNLL_Corpus(file):
    lines_list =[]
    for i in range(len(file)):
        lines = []
        idx = 0
        for line in open(file[i], encoding="utf-8"):
            lines.append(line.strip('\n').split())
            idx += 1
        lines_list.append(lines)

    return lines_list

def Extract_Vocab(lang_size, V_files, V, V_min_freq, train_corpus):
    vocab_all = []
    for i in range(lang_size):  # process train corpora
        if (V_files is not None):
            print("Vocabulary file is given")
            vocab = open(V_files[i], 'r').read().splitlines()
            while ('' in vocab):  # if empty element exists
                vocab.remove('')
            if '<\s>' in vocab:
                vocab.remove('<\s>')
            if '<unk>' in vocab:
                vocab.remove('<unk>')
            wordfreq = Counter(chain.from_iterable(train_corpus[i]))
            vocab = sorted(vocab,key= lambda x: 1/wordfreq[x])

        elif V is not None:
            print("Vocabulary size is given")
            print("Build vocablary with the size " + str(V[i]))
            vocab = Build_Vocab(train_corpus[i], V[i], None)

        else:
            print("frequency threshold is given")
            print("Build vocablary list with the size min freq: " + str(V_min_freq[i]))
            vocab = Build_Vocab(train_corpus[i], None, V_min_freq[i])
        vocab_all.append(vocab)

    return vocab_all

def Convert_word2id(corpus, vocab2id):
    lines_id = []
    sentence_len = []
    for line in corpus:
        line_tmp = []
        sentence_len.append(len(line))
        for i, w in enumerate(line):
            try:
                line_tmp.append(vocab2id[w])
            except KeyError:  # if not in vocab
                line_tmp.append(vocab2id["UNK"])
        lines_id.append(line_tmp)

    return lines_id, sentence_len


def Register_wordID(vocab_list, special_tokens):
    # vocab_list: list of vocabulary
    # return two dictionaries that convert word2id and vice versa

    unique_vocab = list(set(vocab_list) - set(special_tokens))
    unique_vocab = sorted(unique_vocab, key=lambda x: vocab_list.index(x))
    vocab2id = {}
    id2vocab = {}

    for word in special_tokens:
        vocab2id[word] = len(vocab2id)
        id2vocab[len(id2vocab)] = word

    for word in unique_vocab:
        vocab2id[word] = len(vocab2id)
        id2vocab[len(id2vocab)] = word

    V_size = len(unique_vocab)
    assert '<ignore_idx>' not in vocab2id
    # vocab2id['<ignore_idx>'] = -1
    # id2vocab[-1] = '<ignore_idx>'

    return vocab2id, id2vocab, V_size


def Build_Vocab(train_corpus, vocab_size, freq_threshold):
    # build a list of vocab from train corpus
    wordfreq = Counter(chain.from_iterable(train_corpus))
    wordfreq = OrderedDict(wordfreq.most_common())  # sort words by frequency
    if (vocab_size != None):
        print("vocab size is given: ", vocab_size)  # extract top K most frequent words
        if (len(wordfreq) < vocab_size):
            warnings.warn("Your Specified vocab size is larger than the total number of the vocabulary")
            vocab = list(wordfreq.keys())
        else:
            vocab = list(wordfreq.keys())[:vocab_size + 1]

    else:  # if vocab size is not given, use threshold
        print("vocab min freq is given", freq_threshold)
        words = np.array(list(wordfreq.keys()))
        freq = np.array(list(wordfreq.values()))
        idx = freq >= freq_threshold
        vocab = words[idx].tolist()

    return vocab

def gen_dice_dict(lines_id, init_idx, vocab2id, id2vocab, max_dict_size, min_dice = 0.8, min_count = 3):
    assert len(lines_id)==2
    sent_len = len(lines_id[0])
    vocab = [{} for _ in range(2)]
    #cooccur = {}
    cooccur = Counter([])
    for i in range(sent_len):
        words_list = []
        for lang in range(2):
            words = list(set(lines_id[lang][i]))
            for w in words:
                try:
                    vocab[lang][w] += 1
                except KeyError:
                    vocab[lang][w] = 1
            words_list.append(words)
        for src_w in words_list[0]:
            for tgt_w in words_list[1]:
                dictkey = tuple([src_w, tgt_w])
                cooccur.update([dictkey])
                # try:
                #     cooccur[dictkey] += 1
                # except KeyError:
                #     cooccur[dictkey] = 1
    # cooccur = sorted(cooccur.items(), key=lambda v: -1*v[1])
    cooccur = cooccur.most_common()
    src_vocab2count = vocab[0]
    tgt_vocab2count = vocab[1]
    dict_size = 0
    dice_dict = [[] for _ in range(2)]
    UNK_id = vocab2id[0]["UNK"]
    for x in cooccur: #x = ((src,tgt),count)
        srcword = x[0][0]
        tgtword = x[0][1]
        co_count = x[1]
        if co_count < min_count or dict_size >= max_dict_size:
            print ("min_count", co_count)
            break
        src_count = src_vocab2count[srcword]
        tgt_count = tgt_vocab2count[tgtword]
        total_count = src_count + tgt_count
        dice = 2 * co_count / total_count
        if dice >= min_dice and UNK_id not in [srcword,tgtword]:
            # print (dice)
            dict_size+=1
            dice_dict[0].append(srcword - init_idx)
            dice_dict[1].append(tgtword - init_idx)
            # print("SRC: " + id2vocab[0][srcword])
            # print("TRG: " + id2vocab[1][tgtword])
    print("Dict Size: " + str(dict_size))
    return dice_dict
def gen_dice_dict_old(lines_id, init_idx, vocab2id, id2vocab, max_dict_size, min_dice = 0.8, min_count = 3):
        # print (len(lines_id))
        assert len(lines_id)==2
        sent_len = len(lines_id[0])
        vocab = [[] for _ in range(2)]
        cooccur = []
        for i in range(sent_len):
            words_list = []
            for lang in range(2):
                words = list(set(lines_id[lang][i]))
                vocab[lang].extend(words)
                words_list.append(words)
            for src_w in words_list[0]:
                cooccur.extend([(src_w, tgt_w) for tgt_w in words_list[1]])

        #cooccur [(1,2),(3,5),...]
        cooccur = Counter(cooccur).most_common()
        src_vocab2count = Counter(vocab[0])
        tgt_vocab2count = Counter(vocab[1])
        dict_size = 0
        dice_dict = [[] for _ in range(2)]
        UNK_id = vocab2id[0]["UNK"]
        for x in cooccur: #x = ((src,tgt),count)
            srcword = x[0][0]
            tgtword = x[0][1]
            co_count = x[1]
            if co_count < min_count or dict_size >= max_dict_size:
                print ("min_count", co_count)
                break
            src_count = src_vocab2count[srcword]
            tgt_count = tgt_vocab2count[tgtword]
            total_count = src_count + tgt_count
            dice = 2 * co_count / total_count
            if dice >= min_dice and UNK_id not in [srcword,tgtword]:
                # print (dice)
                dict_size+=1
                dice_dict[0].append(srcword - init_idx)
                dice_dict[1].append(tgtword - init_idx)
                # print("SRC: " + id2vocab[0][srcword])
                # print("TRG: " + id2vocab[1][tgtword])
        print("Dict Size: " + str(dict_size))
        return dice_dict


class Dataset():

    def __init__(self, lang_size):
        self.lines_id = [[] for _ in range(lang_size)]
        self.lengths = [[] for _ in range(lang_size)]
        self.lines_id_mono = [[] for _ in range(lang_size)]
        self.lengths_mono = [[] for _ in range(lang_size)]
        self.lines_id_multi = [[] for _ in range(lang_size)]
        self.lengths_multi = [[] for _ in range(lang_size)]
        self.lang_size = lang_size


    def load_para_data(self,lines_id, sentence_len, N_sent):
        self.lines_id = lines_id
        self.lengths = np.array(sentence_len)
        self.N_sent = N_sent

    def load_multi_data(self,lines_id, sentence_len, N_sent):
        self.lines_id_multi = lines_id
        self.lengths_multi = np.array(sentence_len)
        self.N_sent = N_sent

    def load_mono_data(self,lines_id, sentence_len,N_sent_dev):
        self.lines_id_mono = lines_id
        self.lengths_mono = np.array(sentence_len)
        self.N_sent_mono = N_sent_dev


class Vocab_dict():
    def __init__(self):
        self.vocab2id = []
        self.id2vocab = []
        self.V_size = []

    def register_dict(self, vocab2id,id2vocab, V_size):
        self.vocab2id.append(vocab2id)
        self.id2vocab.append(id2vocab)
        self.V_size.append(V_size)


def shuffle_sort_para_data(lines_id, lengths):
    shuf_idx = np.random.permutation(len(lines_id[0]))
    lines_id_new = []
    lengths_new = []
    for lang in range(len(lines_id)):
        lines_id_new.append([lines_id[lang][j] for j in shuf_idx])
        lengths_new.append([lengths[lang][j] for j in shuf_idx])

    # idx = np.argsort(lengths[0]) #sort by src len
    # for lang in range(len(lines_id)):
    #     lines_id[lang] = [lines_id[lang][j] for j in idx]
    #     lengths[lang] = lengths[lang][idx]

    return lines_id_new, lengths_new

def shuffle_sort_mono_data(lines_id, lengths):
    lines_id_new = []
    lengths_new = []
    for lang in range(len(lines_id)):
        shuf_idx = np.random.permutation(len(lines_id[lang]))
        lines_id_new.append([lines_id[lang][j] for j in shuf_idx])
        lengths_new.append([lengths[lang][j] for j in shuf_idx])
        # idx = np.argsort(lengths[lang])  # sort by src len
        # lines_id[lang] = [lines_id[lang][j] for j in idx]
        # lengths[lang] = lengths[lang][idx]

    return lines_id_new,  np.array(lengths_new)

def load_vocab(opt, train_corpus, lang_size, identical):
    vocab_dict = Vocab_dict()
    count = 0
    for V_option in [opt.V, opt.V_min_freq, opt.V_files]:
        if V_option is not None:
            count += 1
            assert len(V_option) == lang_size

    assert count== 1,"Provide one of the 'V', 'V_min_freq' or 'V_files options'"

    vocab_all = Extract_Vocab(lang_size, opt.V_files, opt.V, opt.V_min_freq, train_corpus)
    vocab_dict.special_tokens = ["<PAD>", "<MASK>", "<BOS>", "<EOS>","UNK"]  # specail tokens shared among all langs
    if identical:
        opt.save_name+='.identical'
        identical_words = set(vocab_all[0])
        for i in range(1, len(vocab_all)):
            identical_words = identical_words.intersection(set(vocab_all[i]))
        vocab_dict.shared_V = vocab_dict.special_tokens + list(identical_words)
    else:
        vocab_dict.shared_V = vocab_dict.special_tokens

    print('share_V: ' + str(len(vocab_dict.shared_V)))
    print(vocab_dict.shared_V)

    for i in range(lang_size):  # process train corpora
        vocab =  vocab_all[i] #prepare UNK for each lang
        for word in vocab_dict.special_tokens:
            assert word not in vocab, word +" should not appear in a corpus"
        vocab2id, id2vocab, V_size = Register_wordID(vocab, vocab_dict.shared_V)
        vocab_dict.register_dict(vocab2id, id2vocab, V_size)

    if not os.path.exists("data"):
        os.makedirs("data")

    with open("data/" + opt.save_name + "_inputs.txt", "a") as f:
        f.write("shared_Vsize: " + str(len(vocab_dict.shared_V)) + "\n")
        print("shared_V: ", str(len(vocab_dict.shared_V)))
        for lang in range(lang_size):
            f.write("Vsize" + str(lang) + ": " + str(vocab_dict.V_size[lang]) + "\n")
            if opt.V_files:
                f.write("V_files" + str(lang) + ": " + str(opt.V_files[lang]) + "\n")
            elif opt.V:
                f.write("V" + str(lang) + ": " + str(opt.V[lang]) + "\n")
            else:
                f.write("V_min_freq"+ str(lang) + ": " + str(opt.V_min_freq[lang]) + "\n")
            print("V: ", str(vocab_dict.V_size[lang]))

        f.close()

    return vocab_dict

def load_corpus(corpus, vocab2id):
    lines_id = []
    sentence_len = []
    for i in range(len(corpus)):
        lines_id_tmp, sentence_len_tmp = Convert_word2id(corpus[i], vocab2id[i])
        lines_id.append(lines_id_tmp)
        sentence_len.append(sentence_len_tmp)
    return lines_id, np.array(sentence_len)


def augment_data(lines, rep, ramdom_idx):
    out = lines.copy()
    out = out * rep
    out += [lines[idx] for idx in ramdom_idx]
    return out

def oversampling(lines_id, lengths):
    lang_size = len(lines_id)
    largest_corpus = np.argmax([len(lines_id[i]) for i in range(lang_size)])
    max_sentence_num = len(lines_id[largest_corpus])
    lines_id_new = []
    lengths_new = []
    for i in range(lang_size):
        sentence_num = len(lines_id[i])
        if max_sentence_num == sentence_num:
            augmented_lines_id = lines_id[i]
            augmented_lengths = lengths[i]
        else:
            print("Perform oversampling")
            print("max_sentence_num" + ": ", max_sentence_num)
            print("src lang" + str(i) + ": ", sentence_num)
            rep = max_sentence_num // sentence_num
            remainder = max_sentence_num % sentence_num
            ramdom_idx = sample(range(sentence_num), remainder)
            augmented_lines_id = augment_data(lines_id[i], rep, ramdom_idx)
            augmented_lengths = augment_data(lengths[i], rep, ramdom_idx)

        lines_id_new.append(augmented_lines_id)
        lengths_new.append(augmented_lengths)

    # dataset.train_data_size = max_sentence_num
    return lines_id_new, lengths_new

def oversampling_multi(lines_id, lengths):
    lang_size = len(lines_id)
    largest_corpus = np.argmax([len(lines_id[i]) for i in range(lang_size)])
    max_sentence_num = len(lines_id[largest_corpus])
    lines_id_new = []
    lengths_new = []
    for i in range(lang_size):
        sentence_num = len(lines_id[i])
        if max_sentence_num == sentence_num:
            augmented_lines_id = lines_id[i]
            augmented_lengths = lengths[i]
        else:
            print("Perform oversampling")
            print("max_sentence_num" + ": ", max_sentence_num)
            print("src lang" + str(i) + ": ", sentence_num)
            rep = max_sentence_num // sentence_num
            remainder = max_sentence_num % sentence_num
            ramdom_idx = sample(range(sentence_num), remainder)
            augmented_lines_id = augment_data(lines_id[i], rep, ramdom_idx)
            augmented_lengths = augment_data(lengths[i], rep, ramdom_idx)

        lines_id_new.append(augmented_lines_id)
        lengths_new.append(augmented_lengths)

    # dataset.train_data_size = max_sentence_num
    return lines_id_new, lengths_new

def save_files(save_name, dataset, vocab_dict):
    print("saving files")
    with open("data/" + save_name + ".data", mode='wb') as f:
        pickle.dump(dataset, f)
        f.close()

    with open("data/" + save_name + ".vocab_dict", mode='wb') as f:
        pickle.dump(vocab_dict, f)
        f.close()

def save_vocab(save_name, train_corpus, vocab_dict):
    for lang in range(len(train_corpus)):
        wordfreq = Counter(chain.from_iterable(train_corpus[lang]))
        with open("data/" + save_name + ".vocab" + str(lang) + ".txt", "w",encoding="utf-8") as f:
            for id in vocab_dict.id2vocab[lang].keys():
                word = vocab_dict.id2vocab[lang][id]
                if word not in ["<MASK>","<BOS>","<EOS>","<PAD>",'UNK']:
                    #f.write((vocab_dict.id2vocab[lang][id]) + ' '+str(wordfreq[word])+"\n")
                    f.write((vocab_dict.id2vocab[lang][id]) + "\n")