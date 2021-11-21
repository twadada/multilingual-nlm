import torch
from logging import getLogger, StreamHandler, INFO
from utils.train_base_new import load_data, save_emb, out_wordemb, preprare_model

opt = parser.parse_args()
import pickle
model_name = "path_to_model"
dirname="dir"

folder = model_name.split("/")[0]
with open(folder+"/options.pkl",'rb') as f:
    model_opt = pickle.load(f)

file = open("data/"+ model_opt.data + ".vocab_dict", 'rb')
vocab_dict = pickle.load(file)
logger = getLogger('Log')
dataset, vocab_dict = load_data("data/"+model_opt.data,logger)
model, dataset, vocab_dict = preprare_model(model_opt, dataset, vocab_dict, logger)
model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
with torch.no_grad():  # save word embeddings
    for lang in range(model.lang_size):
        emb_weight = model.embedding_weight(lang)
        vocab2emb = out_wordemb(vocab_dict.id2vocab[lang], emb_weight)
        save_name = dirname + 'lang' + str(lang) + '.vec'
        save_emb(vocab2emb, model_opt.emb_size, save_name)
