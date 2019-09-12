# About
This repository provides the code for [‘Unsupervised Multilingual Word Embedding with Limited Resources using Neural Language Models’](https://www.aclweb.org/anthology/P19-1300). 
# Dependencies
* Python 3
* numpy
* torch (>=1.0.1)

# Usage

## Preprocess
**First, you can run preprocess.py to preprocess data before training**. It is **highly recommended to tokenize and lowercase your corpora** before running this code. You can input any number of languages, i.e., train corpora, to obtain multilingual word embeddings. For instance, you can pre-process train files `'train.fr', 'train.de' and 'train.en'` as follows. 

```
python preprocess.py -train train.fr train.de train.en -V_min_freq 5 5 3  -save_name frdeen
```

**It builds vocabularies that include words used at least 5, 5, and 3 times in 'train.fr', 'train.de', and 'train.en', respectively**. This code generates `'frdeen_inputs.txt', 'frdeen.data' and 'frdeen.vocab'` in the 'data' directory, and these files are used for training models. Instead of the 'V_min_freq' option, you may set vocabulary sizes (-V) or feed vocabulary files (-V_files) for each language, especially when vocabulary sizes are very large. **We recommend that you keep the vocabulary sizes at most 70000 for each language**. Alternatively, you may reduce the vocabulary sizes using subword segmentation methods such as Byte-Pair-Encoding (BPE) [2], although our model is not tested on this condition and may not work well. 

## Train
**After preprocessing, you can run train.py to obtain multilingual embeddings**. Use the name of the data saved in preprocessing (frdeen) for the '-data' argument. In our paper, we used the following options for the low-resource conditions (50k sentences for each language). 

```
python train.py -data frdeen -gpuid 1 -save_dir result -stop_threshold 0.99 -batch_size 64 -epoch_size 10 -opt_type SGD -learning_rate 1.0 -n_layer 2 -emb_size 300 -h_size 300 -remove_models
```

However, **the following options empirically yield better embeddings** at the expense of the training speed.

```
python train.py -data frdeen -gpuid 1 -save_dir save_dir_path -batch_size 32 -epoch_size 30 -opt_type ASGD -learning_rate 5.0 -n_layer 2 -emb_size 300 -h_size 300 -remove_models 
```

For the different-domain conditions (1M sentences for each language), we set the 'h_size' as 1024 in our paper. 
 

This code produces `'frdeen_params.txt', 'frdeen_epochX.model' (X = epoch size), and 'frdeen.lang{0,1,..,N-1}.vec (N = number of languages)'` in the specified directory by '-save_dir'. The first text file saves the options used in train.py and preprocess.py. The second file saves trained Multilingual Neural Language Models, and **the last files are multilingual word embeddings, e.g., for lang0 (fr), 1 (de), and 2 (en).**


## Evaluation

**You can run align_words.py to evaluate multilingual embeddings on a word alignment task.**

```
python align_words.py -dict dict_path -src src_vec_path -tgt tgt_vec_path -save save_name
```

**This code aligns pairs of words in a dictionary at 'dict_path' using CSLS** and saves the result as 'save_name'. **_Note that this evaluation is different from another evaluation method called 'Bilingual Lexicon Induction'_**, which extracts the most similar target words to each source word from the whole target vocabulary. 


# Reference
[1] Takashi Wada, Tomoharu Iwata, Yuji Matsumoto, Unsupervised Multilingual Word Embedding with Limited Resources using Neural Language Models, The 57th Annual Meeting of the Association for Computational Linguistics (ACL 2019)
```
@inproceedings{wada-etal-2019-unsupervised,
    title = "Unsupervised Multilingual Word Embedding with Limited Resources using Neural Language Models",
    author = "Wada, Takashi  and Iwata, Tomoharu  and Matsumoto, Yuji",
    booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics",
    month = jul,
    year = "2019",
    address = "Florence, Italy",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/P19-1300",
    pages = "3113--3124"}
 ```
[2] Rico Sennrich, Barry Haddow and Alexandra Birch (2016): Neural Machine Translation of Rare Words with Subword Units Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2016)