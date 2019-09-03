# About
This repository provides the code for ‘Unsupervised Multilingual Word Embedding with Limited Resources using Neural Language Models’. 

# Usage

## Preprocess
First, run preprocess.py to preprocess data before training. For instance, you can preprocess train files 'train.fr', 'train.de' and 'train.en' as follows:

python -train train.fr train.de train.en -V_min_freq 5 5 3  -save_name frdeen 

This code generates 'data/frdeen_inputs.txt', 'data/frdeen.data' and 'data/frdeen.vocab', which are used for training models. What it does is to build vocabularies that include words used at least 5, 5, and 3 times in 'train.fr', 'train.de', and 'train.en', respectively. Instead of the 'V_min_freq' option, you may set vocabulary sizes (-V) or feed vocabulary files (-V_files) for each language. 

## Train
Run train.py to obtain multilingual embeddings. Set the name of the saved preprocessed data (frdeen) for the '-data' argument. In our paper, we used the following options for the low-resource conditions. 

python train.py -data frdeen -gpuid 1 -n_layer 2  -emb_size 300 -h_size 300 -batch_size 64 -epoch_size 10  -opt_type SGD  -learning_rate 1.0 -remove_models -stop_threshold 0.99 -save_dir dir_name

However, the following options empirically yield better embeddings at the expense of the training speed. 

`python train.py -data frdeen -gpuid 1 -n_layer 2  -emb_size 300 -h_size 300 -batch_size 32 -epoch_size 30  -opt_type ASGD  -learning_rate 5.0 -remove_models -save_dir dir_name`

For the different-domain conditions (1M sentences for each language), we set the 'h_size' as 1024. 
 
This code produces 'fren_params.txt', 'frdeen_epochX.model' (X = epoch size), and 'frdeen.lang{0,1,2}.vec' in the 'dir_name' directory. The first txt file describes the options used in train.py and preprocess.py. The second file is a trained model, and the last files are multilingual word embeddings for lang0 (fr), 1 (de), and 2 (en). 


## Evaluation

Run align_words.py to evaluate multilingual embeddings on a word alignment task. 

python align_words.py -dict fr-de -src dir_name/frdeen.lang0.vec -tgt dir_name/frdeen.lang1.vec -save save_name

This code aligns pairs of words in a 'fr-de' dictionary using CSLS. Note that this evaluation is different from another evaluation method called 'Bilingual Word Induction', which extracts target words from the target vocabulary table for each source word.  


# Reference
Takashi Wada, Tomoharu Iwata, Yuji Matsumoto, Unsupervised Multilingual Word Embedding with Limited Resources using Neural Language Models, The 57th Annual Meeting of the Association for Computational Linguistics (ACL), 2019



