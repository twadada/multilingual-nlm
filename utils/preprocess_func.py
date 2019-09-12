from collections import Counter, OrderedDict
from itertools import chain
import warnings
import os
import numpy as np
from random import sample
import pickle

def Read_Corpus(file):
    lines_list =[]
    for i in range(len(file)):
        lines = []
        for line in open(file[i]):
            lines.append(line.strip('\n').split())
        lines_list.append(lines)
    return lines_list

def Extract_Vocab(lang_size, V_files, V, V_min_freq, train_corpus):
    V_size_param = []
    vocab_all = []
    vocab_list = []
    for i in range(lang_size):  # process train corpora
        if (V_files is not None):
            print("Vocabulary file is given")
            V_size_param.append("Vocab_file")
            V_size_param.append(V_files[i])
            vocab = open(V_files[i], 'r').read().splitlines()
            while ('' in vocab):  # if empty element exists
                vocab.remove('')
            if '<\s>' in vocab:
                vocab.remove('<\s>')
            if '<unk>' in vocab:
                vocab.remove('<unk>')

        elif V is not None:
            print("Vocabulary size is given")
            print("Build vocablary with the size " + str(V[i]))
            V_size_param.append("V_size_max")
            V_size_param.append(V[i])
            vocab = Build_Vocab(train_corpus[i], V[i], None)

        else:
            print("frequency threshold is given")
            print("Build vocablary list with the size min freq: " + str(V_min_freq[i]))
            V_size_param.append("V_min_freq")
            V_size_param.append(V_min_freq[i])
            vocab = Build_Vocab(train_corpus[i], None, V_min_freq[i])
        vocab_all.append(vocab)
        vocab_list.extend(vocab)

    return vocab_all, vocab_list, V_size_param

def Convert_word2id(corpus, vocab2id_input, vocab2id_output):
    lines_id_input = []
    lines_id_output = []
    sentence_len = []
    for line in corpus:
        line_input = []
        line_output = []
        sentence_len.append(len(line))
        for i, w in enumerate(line):
            try:
                line_input.append(vocab2id_input[w])
                line_output.append(vocab2id_output[w])
            except KeyError:  # if not in vocab
                line_input.append(vocab2id_input["UNK"])
                line_output.append(vocab2id_output["UNK"])

        lines_id_input.append(line_input)
        lines_id_output.append(line_output)

    return lines_id_input, lines_id_output, sentence_len


def Register_wordID(vocab_list, special_tokens, initial_vocab_id):
    # vocab_list: list of vocabulary
    # return two dictionaries that convert word2id and vice versa
    for word in special_tokens:
        assert word not in vocab_list

    unique_vocab = list(set(vocab_list))
    unique_vocab = sorted(unique_vocab, key=lambda x: vocab_list.index(x))

    vocab2id_input = {}
    id2vocab_input = {}
    vocab2id_output = {}
    id2vocab_output = {}

    for word in special_tokens:
        vocab2id_input[word] = len(vocab2id_input)
        id2vocab_input[len(id2vocab_input)] = word

    vocab2id_output["<EOS>"] = 0
    id2vocab_output[0] = "<EOS>"

    for word in unique_vocab:
        vocab2id_input[word] = initial_vocab_id
        id2vocab_input[initial_vocab_id] = word
        initial_vocab_id += 1

        vocab2id_output[word] = len(vocab2id_output)
        id2vocab_output[len(id2vocab_output)] = word

    V_size = len(vocab2id_output)

    assert '<ignore_idx>' not in vocab2id_output

    vocab2id_output['<ignore_idx>'] = -1
    id2vocab_output[-1] = '<ignore_idx>'

    return vocab2id_input, vocab2id_output, id2vocab_input, id2vocab_output, V_size


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


class Dataset():

    def __init__(self, lang_size):
        self.V_size = []
        self.lines_id_input = []
        self.lines_id_output = []
        self.lengths = []
        self.lang_size = lang_size


class Vocab_dict():
    def __init__(self):
        self.vocab2id_input = []
        self.vocab2id_output = []
        self.id2vocab_input = []
        self.id2vocab_output = []
        self.V_size = []

    def register_dict(self, vocab2id_input, vocab2id_output, id2vocab_input, id2vocab_output, V_size):
        self.vocab2id_input.append(vocab2id_input)
        self.vocab2id_output.append(vocab2id_output)
        self.id2vocab_input.append(id2vocab_input)
        self.id2vocab_output.append(id2vocab_output)
        self.V_size.append(V_size)


class Preprocesser():
    def shuffle_train_data(self,comparable):
        if comparable:
            shuf_idx = np.random.permutation(len(self.dataset.lines_id_input[0]))
            for lang in range(self.dataset.lang_size):
                self.dataset.lines_id_input[lang] = [self.dataset.lines_id_input[lang][j] for j in shuf_idx]
                self.dataset.lines_id_output[lang] = [self.dataset.lines_id_output[lang][j] for j in shuf_idx]
                self.dataset.lengths[lang] = self.dataset.lengths[lang][shuf_idx]
        else:
            for lang in range(self.dataset.lang_size):
                shuf_idx = np.random.permutation(len(self.dataset.lines_id_input[lang]))
                self.dataset.lines_id_input[lang] = [self.dataset.lines_id_input[lang][j] for j in shuf_idx]
                self.dataset.lines_id_output[lang] = [self.dataset.lines_id_output[lang][j] for j in shuf_idx]
                self.dataset.lengths[lang] = self.dataset.lengths[lang][shuf_idx]

        return self.dataset

    def load_vocab(self, opt, train_corpus, lang_size):
        self.lang_size = lang_size
        self.dataset = Dataset(self.lang_size)
        self.vocab_dict = Vocab_dict()

        count = 0
        for V_option in [opt.V, opt.V_min_freq, opt.V_files]:
            if V_option is not None:
                count += 1
                assert len(V_option) == self.lang_size

        if count != 1:
            raise Exception("Provide one of the 'V', 'V_min_freq' or 'V_files options'")

        vocab_all, vocab_list, V_size_param = Extract_Vocab(self.lang_size, opt.V_files, opt.V, opt.V_min_freq, train_corpus)
        special_tokens = ["<BOS_fwd>", "<BOS_bkw>", "<PAD>"]  #  specail tokens shared among all langs
        initial_vocab_id = len(special_tokens)
        for i in range(self.lang_size):  # process train corpora
            vocab = ["UNK"] + vocab_all[i] #prepare UNK for each lang
            vocab2id_input, vocab2id_output, id2vocab_input, id2vocab_output, V_size = Register_wordID(vocab,
                                                                                                       special_tokens,
                                                                                                       initial_vocab_id)
            self.vocab_dict.register_dict(vocab2id_input, vocab2id_output, id2vocab_input, id2vocab_output, V_size)
            initial_vocab_id = max(id2vocab_input.keys()) + 1

        if not os.path.exists("data"):
            os.makedirs("data")

        with open("data/" + opt.save_name + "_inputs.txt", "w") as f:
            f.write("save_name" + ": " + "data/" + opt.save_name + "\n")

            for lang in range(self.lang_size):
                f.write("train_file" + str(lang)+": " + opt.train[lang] + "\n")
                f.write(V_size_param[2 * lang] + str(lang) + ": " + str(V_size_param[2 * lang + 1]) + "\n")
                f.write("Vsize" + str(lang) + ": " + str(self.vocab_dict.V_size[lang]) + "\n")
                print("V + <EOS> and <BOS_fwd/bkw>: ", str(self.vocab_dict.V_size[lang]))
            f.close()

    def load_corpus(self,corpus):
        lines_id_input = []
        lines_id_output = []
        sentence_len = []
        for i in range(self.dataset.lang_size):
            lines_id_input_tmp, lines_id_output_tmp, sentence_len_tmp = Convert_word2id(corpus[i], self.vocab_dict.vocab2id_input[i], self.vocab_dict.vocab2id_output[i])
            lines_id_input.append(lines_id_input_tmp)
            lines_id_output.append(lines_id_output_tmp)
            sentence_len.append(sentence_len_tmp)
        return lines_id_input, lines_id_output, sentence_len

    def augment_data(self,lines, rep, ramdom_idx):
        out = lines.copy()
        out = out * rep
        out += [lines[idx] for idx in ramdom_idx]
        return out

    def oversampling(self):

        largest_corpus = np.argmax([len(self.dataset.lines_id_input[i]) for i in range(self.lang_size)])
        max_sentence_num = len(self.dataset.lines_id_input[largest_corpus])

        for i in range(self.lang_size):
            sentence_num = len(self.dataset.lines_id_input[i])
            if max_sentence_num != sentence_num:
                print("Perform oversampling")
                print("max_sentence_num" + ": ", max_sentence_num)
                print("src lang" + str(i) + ": ", sentence_num)
                rep = max_sentence_num // sentence_num
                remainder = max_sentence_num % sentence_num
                ramdom_idx = sample(range(sentence_num), remainder)
                self.dataset.lines_id_input[i] = self.augment_data(self.dataset.lines_id_input[i], rep, ramdom_idx)
                self.dataset.lines_id_output[i] = self.augment_data(self.dataset.lines_id_output[i], rep, ramdom_idx)
                self.dataset.lengths[i] = self.augment_data(self.dataset.lengths[i], rep, ramdom_idx)

            self.dataset.lengths[i] = np.array(self.dataset.lengths[i]) # list -> numpy

        self.dataset.train_data_size = max_sentence_num

    def save_files(self, save_name, output_vocab):


        print("saving files")

        with open("data/" + save_name + ".data", mode='wb') as f:
            pickle.dump(self.dataset, f)
            f.close()

        with open("data/" + save_name + ".vocab_dict", mode='wb') as f:
            pickle.dump(self.vocab_dict, f)
            f.close()

        if (output_vocab):
            for lang in range(self.lang_size):
                with open("data/" + save_name + ".vocab" + str(lang) + ".txt", "w") as f:
                    for id in self.vocab_dict.id2vocab_input[lang].keys():
                        if self.vocab_dict.id2vocab_input[lang][id] not in ["<BOS_fwd>", "<BOS_bkw>", "<PAD>",'UNK']:
                            f.write((self.vocab_dict.id2vocab_input[lang][id]) + "\n")

