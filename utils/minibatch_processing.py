import numpy as np

def Generate_bacth_idx(dataset, batch_size):
    #dataset: sorted by src length
    batch_idx_list = []
    for i in range(0, dataset.train_data_size, batch_size):
        batch_idx = list(range(i, min(i + batch_size,dataset.train_data_size)))  # batch_size
        batch_idx_list.append(batch_idx)  # n_batch, batch_size
    return batch_idx_list

def Sort_data_by_sentlen(dataset):
    for lang in range(dataset.lang_size):
        idx = np.argsort(dataset.lengths[lang])[::-1]
        dataset.lines_id_input[lang] = [dataset.lines_id_input[lang][j] for j in idx]
        dataset.lines_id_output[lang] = [dataset.lines_id_output[lang][j] for j in idx]
        dataset.lengths[lang] = dataset.lengths[lang][idx]
    return dataset

