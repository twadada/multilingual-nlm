# About
This repository provides the code for ‘Unsupervised Multilingual Word Embedding with Limited Resources using Neural Language Models’. 

##Preprocess
First, run preprocess.py to preprocess data before training. For instance, you can preprocess train files 'train.fr', 'train.de' and 'train.en' as follows:

python -train train.fr train.de train.en -V_min_freq 5 5 3  -save_name frdeen 

This code generates 'data/frdeen_inputs.txt', 'data/frdeen.data' and 'data/frdeen.vocab', which are used for training models. What it does is to build vocabularies that include words used at least 5, 5, and 3 times in 'train.fr', 'train.de', and 'train.en', respectively. Instead of the 'V_min_freq' option, you may set vocabulary sizes (-V) or feed vocabulary files (-V_files) for each language. 

##Train
Run train.py to obtain multilingual embeddings. Set the name of the saved preprocessed data (frdeen) for the '-data' argument. In our paper, we used the following options for the low-resource conditions. 

python train.py -data frdeen -gpuid 1 -n_layer 2  -emb_size 300 -h_size 300 -batch_size 64 -epoch_size 10  -opt_type SGD  -learning_rate 1.0 -remove_models -stop_threshold 0.99 -save_dir dir_name

However, the following options empirically yield better embeddings at the expense of the training speed. 

python train.py -data frdeen -gpuid 1 -n_layer 2  -emb_size 300 -h_size 300 -batch_size 32 -epoch_size 30  -opt_type ASGD  -learning_rate 5.0 -remove_models -save_dir dir_name

For the different-domain conditions (1M sentences for each language), we set the 'h_size' as 1024. 
 
##Evaluation

Run Matching_Words.py to obtain multilingual embeddings


# Reference
Takashi Wada, Tomoharu Iwata, Yuji Matsumoto, Unsupervised Multilingual Word Embedding with Limited Resources using Neural Language Models, The 57th Annual Meeting of the Association for Computational Linguistics (ACL), 2019



