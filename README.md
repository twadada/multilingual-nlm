# About
This repository contains the implementations for the following papers:

[1] [Unsupervised Multilingual Word Embedding with Limited Resources using Neural Language Models](https://www.aclweb.org/anthology/P19-1300) (ACL 19)

[2] [Learning Contextualised Cross-lingual Word Embeddings and Alignments for Extremely Low-Resource Languages Using Parallel Corpora](https://arxiv.org/abs/2010.14649) (1st Workshop on Multilingual Representation Learning (MRL) at EMNLP 21 **[Best Paper Award]**)

**(NOTE) The performance of the unsupervised method in [1] has improved with the techniques used in [2] (e.g. weight tying, shared subword embeddings)**

# Dependencies
* Python 3
* numpy
* torch (>=1.0.1); tested on 1.7.1/1.8.1
* tqdm
* sentencepiece (optional)

# Usage

## Preprocess
### Build Vocab 
You first run preprocess.py to preprocess data. It is **highly recommended to tokenize and lowercase your corpora (+ subword segmentation if the data is large and your goal is to obtain word alignments)** before running this code. You can input two or more languages to obtain multilingual word embeddings. 

* **If you have monolingual data only (of syntacticaly similar languages)**, e.g. train.en/train.fr/train.de, run the following command:
```
en_minfreq=1
fr_minfreq=1
de_minfreq=1
save_name=enfrde

python preprocess.py -mono train.en train.fr train.de -V_min_freq ${en_minfreq} ${fr_minfreq} ${de_minfreq} -save_name ${save_name} -output_vocab
```
* **If you have one parallel data**, e.g. train.en-fr.{en/fr}, run the following command:
```
python preprocess.py -para train.en-fr.en train.en-fr.fr -V_min_freq ${en_minfreq} ${fr_minfreq} -save_name ${save_name} -output_vocab
```
* **If you have two or more parallel data where one language (e.g. English) is aligned with the other langauges**, e.g. train.en-fr.{en/fr}, train.en-de.{en/de}, run the following command:
```
python preprocess.py -multi train.en-fr.en train.en-de.en train.en-fr.fr train.en-de.de -V_min_freq ${en_minfreq} ${fr_minfreq} ${de_minfreq} -save_name ${save_name} -output_vocab
```
(Similarly, if you have three parallel data: en1-X, en2-Y, and en3-Z, the input of -multi should be "en1 en2 en3 X Y Z"')

**These commands generate the files whose names start with "${save_name}.{mono/para/multi}", which are used in the subsequent steps.** The option "-V_min_freq 1" means all the words in the data are included in the vocabulary, but if the data is very large and not segmented into subwords in advance, you should set -V_min_freq higher to adjust the vocab size (< 40~50k). You can also use '-V' option to directly set the vocabulary size, or feed the vocabulary files (-V_files) for each language. 

### Extract Pseudo Dictionaries from Parallel Data (used for [2])
If you use parallel data, you can generate pseudo dictionaries based on Dice coefficient, and use them for model selection. 
```
python generate_dice_dict.py -files train.en-fr.en train.en-fr.fr -save en-fr_dict.txt
python generate_dice_dict.py -files train.en-de.en train.en-de.de -save en-de_dict.txt
```
However, if you have bilingual dicrionaries that you can use for model validation, it's probably better to use them rather than using those pseudo dictionaries.

### Prepare Subword Files (Optional)

To learn subword embeddings, prepare files where **each line is "word + list of its subwords" separated by space** for each word in the vocabulary file (data/${save_name}.mono/para/multi.vocab{0,1,...}.txt).

(e.g.)

understandable ▁under stand able" <br>
plays ▁play s 

In this example, the embeddings of the subwords (e.g. "▁under", "stand", "able") are shared among all the input languages. To obtain subwords, you may learn a SentencePiece model [3] (https://github.com/google/sentencepiece) on training corpora (either jointly or separately for each language) and apply the model to the vocabulary file as follows (data/enfrde.para.vocab0.txt.):

```
python generate_subwords.py -vocab_file data/enfr.para.vocab0.txt -spm_model path_to_spm_model -out en_subwords.txt
```

## Train
### Bilingual Model in [2]
To obtain cross-lingual word embeddings using "enfr.para", run the following command (the hyper-parameters are set to the ones used in low-resource experiments in [2]). **Note that the model performance can be somewhat unstable when trained on very small data; this is because the training is performed in an "unsupervised" way in that the model does not employ any cross-lingual supervision at a word level.**

```
save_dir=Result
CUDA_VISIBLE_DEVICES=0 python train.py -gpu -data enfr.para -share_vocab 0 -eval_dict en-fr_dict.txt -seed 0 -dr_rate 0.5 -epoch_size 200 -opt_type Adam -save_dir ${save_dir} -batch_size 16 -enc_dec_layer 1 1 -emb_size 500 -h_size 500  -remove_models -save_point 10
```
 
This command produces cross-lingual word embeddings "enfr.para.lang{0,1}.vec" in ${save_dir} directory. It also produces the best model "${data}_epochX.bestmodel", which you can use to perform word alignments or generate contextualised word embeddings. To use multiple GPUs, specify multiple GPU ids at CUDA_VISIBLE_DEVICES (but the implementation gets computationally less efficient). To share the decoders among different languages (which may work better for syntactically close languages such as English and French), set "-class_number 0 0".

#### Learn Subword-Aware Embeddings

To train subword-aware word embeddings, add "-share_vocab" and "-subword" options as follows:
```
CUDA_VISIBLE_DEVICES=0 python train.py -gpu -data enfr.para -share_vocab 3 -subword en_subwords.txt fr_subwords.txt -eval_dict en-fr_dict.txt -seed 2 -dr_rate 0.5 -epoch_size 200 -opt_type Adam -save_dir ${save_dir} -batch_size 16 -enc_dec_layer 1 1 -emb_size 500 -h_size 500  -remove_models -save_point 10
```
where "en_subwords.txt" and "fr_subwords.txt" denote the files that contain subword information for each word in the vocabulary. The option "-share_vocab 3" denotes training the average-pooling model; for the CNN model, use "-share_vocab 4" instead (however, this model is scalable only on extremely low-resource data). If you have applied subword segmentaion to the training corpora, you do not have to learn subword-aware embeddings (but you can learn "subsubword" embeddings in the same way, which would be effective for some languages, e.g. Japanese, Chinese).

#### Use Pre-trained Embeddings

To use pre-trained word embeddings, **set "-pretrained embedding_file.txt", where embedding_file.txt denotes a word embedding file in a word2vec format** (the first line is "vocab_size  embedding_dimension", and the sebseqnet lines are "word word_vector" ). **Note that the input should be the embedding file for the targert language (e.g. French in the example above), and the embedding dimension should be the same as emb_size/h_size**. During training, the pre-trained embedding weights E are freezed and words are represented by a⊙E+b, where a and b are trainable vectors. After training, the code outputs the numpy vectors for a and b as "\*param_a.npy" and "\*.param_b.npy", which you may use to convert pre-trained embeddings of OOV words.


### Multilingual Model in [2]
To generate multilingual word embeddings using "enfrde.multi", feed N-1 (psuedo) dictionaries for "-eval_dict" (and N subword files for -subword if you use subwords) where N is the number of languages. Note that the order of the language pairs should be consistent between "-eval_dict" in train.py and "-multi" in preprocess.py (e.g. en-fr, en-de). The following command trains the model that learns subword-aware multilingual word embeddings using average pooling (to disable subwords embeddings, remove -share_vocab and -subword options).

```
CUDA_VISIBLE_DEVICES=0 python train.py -gpu -data enfrde.multi -share_vocab 3 -subword en_subwords.txt fr_subwords.txt de_subwords.txt -eval_dict en-fr_dict.txt en-de_dict.txt -seed 2 -dr_rate 0.5 -epoch_size 100 -opt_type Adam -save_dir Result -batch_size 16 -enc_dec_layer 1 1 -emb_size 500 -h_size 500  -remove_models -save_point 10
```

### Fully Unsupervised Model in [1]
**To generate multilingual word embeddings using "enfrde.mono", simply replace "-data enfrde.multi" above with "-data enfrde.mono"; set -class_number 0 0 0 (this means sharing one decoder among three languages); and omit "-eval_dict"** (but if you have dictionaries, you can still use them in the same way as described above). You can also train subword-aware embeddings using the subword option and that may yield better performance, although this is not proposed in the original paper [1]. Note that this model does not have an encoder.

### Tips for Hyper-parameters
The most important hyper-parameters that significantly affect the performance are -epoch_size, -emb_size/h_size, -batch_size, -dr_rate, and -enc_dec_layer. One rule of thumb is that **the larger the vocabulary size, the larger the embedding size should be**. In our paper [2], we set the embedding size to 500 for small data (300 ~ 300k parallel sents) with the vocabulary size < 20k, and to 768 for large data (~ 2M parallel sents) with the vocabulary size < 35k. Also, **larger training data requires the larger batch_size/enc_dec_layer and smaller epoch_size/save_point. (refer to [2] for the details)**. If you want to reduce the embedding size, you should reduce the dropout rate to 0.1 ~ 0.3, although this may lead to poorer performance on BLI/word alignments.

To know whether the training of the model in [2] has been successful, you can check the BLI performance on psuedo dictionaries — if P@1 is below 80~90%, it is very likely that either the hyper-parameters are not optimal, or the training data is very noisy. Also, if current_loss/previous_loss is below 0.99 when the training is done, you should probably increase the epoch size and train the model longer to ensure convergence.

## Evaluation

**You can perform Bilingual Lexicon Induction (BLI) as follows.** Here, srcV and tgtV are the vocabulary files that contain a list of words in each line — you can use data/*vocab0.txt and data/*vocab1.txt, which are generated by preprocess.py. A part of the implementation is based on https://github.com/facebookresearch/MUSE.

```
src=path_to_src_embedding (e.g. ${save_dir}/de-en.para.lang0.vec )
tgt=path_to_tgt_embedding (e.g. ${save_dir}/de-en.para.lang1.vec )
srcV=path_to_srcVocab (e.g. data/de-en.para.vocab0.txt)
tgtV=path_to_tgtVocab (e.g. data/de-en.para.vocab1.txt)
dict=path_to_gold_dictionary
save_name=result_BLI
python run_BLI.py -src ${src} -tgt ${tgt} -srcV ${srcV} -tgtV ${tgtV} -dict ${dict} -save ${save_name}
```

**You can perform word alignment as follows.** Here, src and tgt are parallel sentences in which you want to align words cross-lingually.
```
src=path_to_source_sent
tgt=path_to_target_sent
model=path_to_model
save_name=result
CUDA_VISIBLE_DEVICES=0 python run_alignment.py -null_align -model ${model} -GPU -src_lang 0 -tgt_lang 1 -src ${src} -tgt ${tgt} -save ${save_name}
CUDA_VISIBLE_DEVICES=0 python run_alignment.py -null_align -backward -model ${model} -GPU -src_lang 0 -tgt_lang 1 -src ${src} -tgt ${tgt} -save ${save_name}.bkw
```

This produces forward and backward word alignments. **To combine them to generate bidirectinal alignments, use ./atools at https://github.com/clab/fast_align [4].** Omit the "-null_align" option to disable NULL alignments and acheive higher recall and lower precision (with usually lower F1). **If you align subwords, you may want to convert the subword alignments into word alignments (if you use sentencepiece to segment words, you can use subword2word_alignment.py as shown in the next section below.).**

To evaluate word alignment, use the following command if the human annotation does not distinguish sure and possible alignments and its file format is the same as the one of the prediction (e.g. 1-1 2-3 4-5).
```
python calc_AER.py  -pred ${prediction} -gold ${gold}  -save save_name
```
If the format of the human annotation is "sent_id src_word_id tgt_word_id Sure/Possible" in each line (e.g. 1 9 8 S), and sent_id/src_word_id/tgt_word_id starts from 1, use this command.
```
python calc_AER.py  -pred ${prediction} -gold ${gold}  -save save_name -line_by_line
```

## Reproduce the De-En word alignment experiment in [2]

1. Download and preprocess de-en training and test data using the code at https://github.com/lilt/alignment-scripts (alignment-scripts)
2. Run the commands below

```
data_dir=path_to_alignment-scripts
de=${data_dir}/train/deen.lc.plustest.src.bpe
en=${data_dir}/train/deen.lc.plustest.tgt.bpe

# remove long sentences to avoid out of memory, and then preprocess data
python filter_long_sent.py -src ${de} -tgt ${en}
save_name=de-en
python preprocess.py -para ${de}.filtered ${en}.filtered -V_min_freq 1 1 -save_name ${save_name} -output_vocab
python generate_dice_dict.py -files $de.filtered $en.filtered -save de-en_dict.txt

# train a de-en model
save_dir=de-en_result
CUDA_VISIBLE_DEVICES=0 python train.py -gpu -data enfr.para -eval_dict de-en_dict.txt -seed 0 -dr_rate 0.5 -epoch_size 10 -opt_type Adam -save_dir ${save_dir} -batch_size 80 -enc_dec_layer 3 2 -emb_size 768 -h_size 768  -remove_models -save_point 1

# Perform alignments
src=${data_dir}/test/deen.lc.src.bpe
tgt=${data_dir}/test/deen.lc.tgt.bpe
model=de-en_result/*bestmodel
save_name=result
CUDA_VISIBLE_DEVICES=0 python run_alignment.py -null_align -model ${model} -GPU -src_lang 0 -tgt_lang 1 -src ${src} -tgt ${tgt} -save ${save_name}
CUDA_VISIBLE_DEVICES=0 python run_alignment.py -null_align -backward -model ${model} -GPU -src_lang 0 -tgt_lang 1 -src ${src} -tgt ${tgt} -save ${save_name}.bkw

# Convert SentencePiece-token alignments to word alignmenbts
python subword2word_alignment.py -alignment ${save_name}.txt -src ${src} -tgt ${tgt}
python subword2word_alignment.py -alignment ${save_name}.bkw.txt -src ${src} -tgt ${tgt}

# Combine forward and backward alignments using atools at https://github.com/clab/fast_align
./atools  -i ${save_name}.txt.walign -j ${save_name}.bkw.txt.walign -c grow-diag-final-and > result_birdir

# Evaluation (This should be around 86.0 in the F-measure)
${data_dir}/scripts/aer.py ${data_dir}/test/deen.talp result_birdir --fAlpha 0.5 --oneRef

```


# Reference
[1] Takashi Wada, Tomoharu Iwata, Yuji Matsumoto, "Unsupervised Multilingual Word Embedding with Limited Resources using Neural Language Models", Proceedings of  the 57th Annual Meeting of the Association for Computational Linguistics (ACL 2019)
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

[2] Takashi Wada, Tomoharu Iwata, Yuji Matsumoto, Timothy Baldwin, Jey Han Lau, "Learning Contextualised Cross-lingual Word Embeddings and Alignments for Extremely Low-Resource Languages Using Parallel Corpora", Proceedings of the 1st Workshop on Multilingual Representation Learning, colocated with EMNLP 2021
 ```
@inproceedings{wada-etal-2021-learning,
    title = "Learning Contextualised Cross-lingual Word Embeddings and Alignments for Extremely Low-Resource Languages Using Parallel Corpora",
    author = "Wada, Takashi  and
      Iwata, Tomoharu  and
      Matsumoto, Yuji  and
      Baldwin, Timothy  and
      Lau, Jey Han",
    booktitle = "Proceedings of the 1st Workshop on Multilingual Representation Learning",
    month = nov,
    year = "2021",
    address = "Punta Cana, Dominican Republic",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.mrl-1.2",
    pages = "16--31",
}
```
[3] Taku Kudo, John Richardson "SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing"
```
@inproceedings{kudo-richardson-2018-sentencepiece,
    title = "{S}entence{P}iece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing",
    author = "Kudo, Taku  and
      Richardson, John",
    booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing: System Demonstrations",
    month = nov,
    year = "2018",
    address = "Brussels, Belgium",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/D18-2012",
    doi = "10.18653/v1/D18-2012",
    pages = "66--71"
}
```

[4] Chris Dyer, Victor Chahuneau, Noah A. Smith, "A Simple, Fast, and Effective Reparameterization of IBM Model 2"
 ```
@inproceedings{dyer-etal-2013-simple,
    title = "A Simple, Fast, and Effective Reparameterization of {IBM} Model 2",
    author = "Dyer, Chris  and
      Chahuneau, Victor  and
      Smith, Noah A.",
    booktitle = "Proceedings of the 2013 Conference of the North {A}merican Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jun,
    year = "2013",
    address = "Atlanta, Georgia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/N13-1073",
    pages = "644--648",
}
 ```
