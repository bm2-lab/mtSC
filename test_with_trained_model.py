import h5py
import os
from torch.utils.data import TensorDataset, DataLoader, Dataset, Sampler
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import argparse
import time

class Net(nn.Module):
    def __init__(self, feature_num):
        super(Net, self).__init__()
        self.layer_1 = nn.Linear(feature_num, 500)
        self.layer_2 = nn.Linear(500, 20)

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = self.layer_2(x)
        return x


def process_data_to_same_gene(gene, root_dir, output_dir, mode):
    dataset_list = os.listdir(root_dir)
    for l in range(len(dataset_list)):
        dataset = dataset_list[l]
        if '.txt' in dataset:
            all_data = pd.read_csv(root_dir + dataset, sep='\t')
        elif '.csv' in dataset:
            all_data = pd.read_csv(root_dir + dataset)
        elif '.h5' in dataset and '_processed' not in dataset:
            all_data = pd.read_hdf(root_dir + dataset)
        else:
            continue
        add_gene = []
        all_data_columns = []
        for all_data_gene in all_data.columns:
            all_data_columns.append(all_data_gene.lower())
        all_data.columns = all_data_columns
        for gene_ in gene:
            if gene_ not in all_data_columns:
                add_gene.append(gene_)
        df = pd.DataFrame(
            np.zeros((all_data.shape[0], len(add_gene))), index=all_data.index, columns=add_gene)
        df = pd.concat([all_data, df], axis=1)
        df = df.loc[df.index.tolist(), gene]
        if mode == 'train':
            df['cell_label'] = df['cell_label'].astype(str)
            train_indices = np.array([]).astype(int)

            cell_label = df['cell_label']
            df.drop(labels=['cell_label'], axis=1, inplace=True)
            df.insert(len(df.columns), 'cell_label', cell_label)

        df.to_hdf(output_dir+dataset.split('.')
                  [0]+'_processed.h5', key='data')


def test(model, test_data, metrics_list, labels_list):
    test_data = torch.tensor(test_data.values,
                        dtype=torch.float32)
    max_likelihood_lists = []
    max_likelihood_classes = []
    for l in range(len(metrics_list)):
        max_likelihood_list, max_likelihood_class = one_model_predict(
            test_data, model, metrics_list[l], labels_list[l])
        max_likelihood_lists.append(max_likelihood_list)
        max_likelihood_classes.append(max_likelihood_class)
        # calculate f1_score
    pred_class = []
    max_likelihood_indices = np.argmax(max_likelihood_lists, axis=0)
    for k in range(len(max_likelihood_indices)):
        max_likelihood_indice = max_likelihood_indices[k]
        pred_class.append(max_likelihood_classes[max_likelihood_indice][k])
    return pred_class


def one_model_predict(test_data, model, metrics, labels):
    test_embedding = model(test_data).detach().cpu().numpy()
    max_likelihood_class = []
    max_likelihood_list = []
    for i in test_embedding:
        predict_pearsonr = []
        for k in metrics:
            predict_pearsonr.append(pearsonr(i, k)[0])
        pred = np.argmax(predict_pearsonr)
        max_likelihood_list.append(predict_pearsonr[pred])
        max_likelihood_class.append(labels[pred])
    return max_likelihood_list, max_likelihood_class


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', required=True, type=str,
                    help="you can choose one of 'brain, pancreas, PBMC-Ding, PBMC-Mereu, new_model",
                    choices=['brain', 'pancreas', 'PBMC-Ding', 'PBMC-Mereu', 'new_model'])
out_args = parser.parse_args()
mode = out_args.mode

testset_dir = 'test_set/'


with open('pre_trained/gene/'+mode+'.txt') as gene_file:
    gene = gene_file.readline().split(', ')
feature_num = len(gene)

test_dataset_list = []

if '_processed.h5' not in str(os.listdir(testset_dir)):
    process_data_to_same_gene(gene, testset_dir, testset_dir, mode='test')

for filename in os.listdir(testset_dir):
    if '_processed.h5' in filename:
        test_dataset_list.append(filename)



model = Net(feature_num)

model.load_state_dict(torch.load(
    'pre_trained/model/'+mode+'.pth'))

metrics_list = []
labels_list = []
for filename in os.listdir('pre_trained/metrics_and_labels/'+mode):
    if 'npz' in filename:
        npzfile = np.load('pre_trained/metrics_and_labels/'+mode+'/'+filename, allow_pickle=True)
        metrics_list.append(npzfile['metrics'])
        labels_list.append(npzfile['labels'])

for i in range(len(test_dataset_list)):
    test_set = test_dataset_list[i]
    test_data = pd.read_hdf(testset_dir+test_set)
    pred_class = test(model, test_data, 
                      metrics_list, labels_list)
    #check
    with open('result.txt','a') as result_file:
        for pred_class_indice in range(len(pred_class)):
            result_file.write(str(test_data.index[pred_class_indice])+'\t'+str(pred_class[pred_class_indice])+'\n')


