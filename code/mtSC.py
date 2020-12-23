import os
from torch.utils.data import TensorDataset, DataLoader, Dataset, Sampler
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from tqdm import tqdm


class Setting:
    """Parameters for training"""

    def __init__(self):
        self.epoch = 300
        self.lr = 0.0005


class Net(nn.Module):
    def __init__(self, feature_num):
        super(Net, self).__init__()
        self.layer_1 = nn.Linear(feature_num, 500)
        self.layer_2 = nn.Linear(500, 20)

    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = self.layer_2(x)
        return x


class CellDataset(Dataset):
    def __init__(self, hdf_file, root_dir):
        self.data_frame = pd.read_hdf(root_dir + hdf_file)

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        train_x = self.data_frame.iloc[idx, :-1]
        train_x = torch.tensor(train_x, dtype=torch.float32)
        return train_x


class NPairSampler(Sampler):
    def __init__(self, labels):
        self.labels = labels

    def generate_npairs(self, labels):
        if isinstance(labels, torch.Tensor):
            labels = labels.detach().cpu().numpy()
        label_set, count = np.unique(labels, return_counts=True)
        label_set = label_set[count >= 2]
        pos_pairs = np.array([np.random.choice(np.where(labels == x)[
                             0], 2, replace=False) for x in label_set])
        neg_tuples = []
        for idx in range(len(pos_pairs)):
            neg_tuples.append(
                pos_pairs[np.delete(np.arange(len(pos_pairs)), idx), 1])
        neg_tuples = np.array(neg_tuples)
        sampled_npairs = [[a, p, *list(neg)]
                          for (a, p), neg in zip(pos_pairs, neg_tuples)]
        return iter(sampled_npairs)

    def __iter__(self):
        """
        This methods finds N-Pairs in a batch given by the classes provided in labels in the
        creation fashion proposed in 'Improved Deep Metric Learning with Multi-class N-pair Loss Objective'.
        Args:
            batch:  np.ndarray or torch.Tensor, batch-wise embedded training samples.
            labels: np.ndarray or torch.Tensor, ground truth labels corresponding to batch.
        Returns:
            list of sampled data tuples containing reference indices to the position IN THE BATCH.
        """
        sampled_npairs = self.generate_npairs(self.labels)
        while True:
            try:
                yield next(sampled_npairs)
            except StopIteration:
                sampled_npairs = self.generate_npairs(self.labels)
                yield next(sampled_npairs)


class NPairLoss(torch.nn.Module):
    def __init__(self, l2=0.05):
        """
        Basic N-Pair Loss as proposed in 'Improved Deep Metric Learning with Multi-class N-pair Loss Objective'
        Args:
            l2: float, weighting parameter for weight penality due to embeddings not being normalized.
        Returns:
            Nothing!
        """
        super(NPairLoss, self).__init__()
        self.l2 = l2

    def npair_distance(self, anchor, positive, negatives):
        """
        Compute basic N-Pair loss.
        Args:
            anchor, positive, negative: torch.Tensor(), resp. embeddings for anchor, positive and negative samples.
        Returns:
            n-pair loss (torch.Tensor())
        """
        return torch.log(1+torch.sum(torch.exp(anchor.reshape(1, -1).mm((negatives-positive).transpose(0, 1)))))

    def weightsum(self, anchor, positive):
        """
        Compute weight penalty.
        NOTE: Only need to penalize anchor and positive since the negatives are created based on these.
        Args:
            anchor, positive: torch.Tensor(), resp. embeddings for anchor and positive samples.
        Returns:
            torch.Tensor(), Weight penalty
        """
        return torch.sum(anchor**2+positive**2)

    def forward(self, batch):
        """
        Args:
            batch:   torch.Tensor() [(BS x embed_dim)], batch of embeddings
        Returns:
            n-pair loss (torch.Tensor(), batch-averaged)
        """
        loss = torch.stack([self.npair_distance(npair[0],
                                                npair[1], npair[2:]) for npair in batch])
        loss = loss + self.l2*torch.mean(torch.stack([self.weightsum(
            npair[0], npair[1]) for npair in batch]))
        return torch.mean(loss)


def get_all_gene(trainset_dir):
    dataset_list = os.listdir(trainset_dir)
    gene = []
    for dataset in dataset_list:
        if '.txt' in dataset:
            df = pd.read_csv(trainset_dir + dataset, sep='\t')
        elif '.csv' in dataset:
            df = pd.read_csv(trainset_dir + dataset)
        elif '.h5' in dataset and '_processed' not in dataset:
            df = pd.read_hdf(trainset_dir + dataset)
        else:
            continue
        file_gene = df.columns.tolist()
        for i in file_gene:
            if i == 'cell_label':
                continue
            if i not in gene:
                gene.append(i)

    with open('pre_trained/gene/new_model.txt', 'w') as gene_file:
        gene_ = ''
        for i in gene[:-2]:
            gene_ = gene_ + i + ', '
        gene_ = gene_ + gene[-2]
        gene_ = gene_.lower()
        gene_file.write(gene_)
    gene = gene_.split(', ')
    gene.append('cell_label')
    return gene


def process_data_to_same_gene(gene, root_dir, output_dir, mode):
    dataset_list = os.listdir(root_dir)
    for l in tqdm(range(len(dataset_list))):
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


def get_dataloaders_and_LabelNumList_and_FeatureNum(train_dataset_list, trainset_dir):
    dataloaders = []
    label_num_list = []
    for train_dataset in train_dataset_list:
        cell_dataset = CellDataset(
            train_dataset, trainset_dir)
        tr_y = cell_dataset.data_frame.iloc[:, -1].values
        npair_sampler = NPairSampler(tr_y)
        dataloader = DataLoader(
            cell_dataset, batch_sampler=npair_sampler, num_workers=5)
        label_num = len(np.unique(tr_y))
        dataloaders.append(dataloader)
        label_num_list.append(label_num)
    feature_num = cell_dataset.data_frame.shape[1]-1
    return dataloaders, label_num_list, feature_num


def train(model, dataloaders, optimizer, epoch, loss_function, device, label_num_list):
    model.train()
    for _epoch in tqdm(range(epoch)):
        loss_list = []
        for dataloader_id in range(len(dataloaders)):
            count = 0
            batch_data = []
            for i, data in enumerate(dataloaders[dataloader_id]):
                data = data.to(device)
                output = model(data)
                batch_data.append(output)
                count = count + 1
                if count % label_num_list[dataloader_id] == 0:
                    optimizer.zero_grad()
                    loss = loss_function(batch_data).view(1, -1)
                    loss_list.append(loss)
                    break
        total_loss = torch.sum(torch.cat(loss_list, dim=1))
        total_loss.backward()
        optimizer.step()
    return model


def get_MetricsList_and_LabelsList(model, train_dataset_list, trainset_dir):
    metrics_list = []
    labels_list = []
    for l in range(len(train_dataset_list)):
        train_set = train_dataset_list[l]
        train_data = pd.read_hdf(trainset_dir+train_set)
        tr_x = train_data.iloc[:, :-1].values
        tr_y = train_data.iloc[:, -1].values
        labels = np.unique(tr_y)
        
        metrics = calculate_metrics(
            tr_x, tr_y, model, labels)
        metrics_list.append(metrics)
        labels_list.append(labels)
        np.savez('pre_trained/metrics_and_labels/new_model/'+train_set.split('.')[0],
                 metrics=metrics, labels=labels)
    return metrics_list, labels_list


def calculate_metrics(tr_x, tr_y, model, labels):
    metrics = []
    for i in labels:
        #classify embedding data according to classes and calculate metrics
        class_indices = np.where(tr_y == i)[0]
        class_data = torch.tensor(
            tr_x[class_indices, :], dtype=torch.float32)
        class_embedding = model(class_data)
        class_embedding = class_embedding.detach().numpy()
        class_metric = np.median(class_embedding, axis=0)
        metrics.append(class_metric)
    return metrics


def test(model, test_data, train_dataset_list, metrics_list, labels_list):
    test_data = torch.tensor(test_data.values,
                        dtype=torch.float32)
    max_likelihood_lists = []
    max_likelihood_classes = []
    for l in range(len(train_dataset_list)):
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
    test_embedding = model(test_data).detach().numpy()
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

def main(trainset_dir, testset_dir):
    if trainset_dir[-1] != '/':
        trainset_dir = trainset_dir+'/'
    if testset_dir[-1] != '/':
        testset_dir = testset_dir+'/'
    print('Get all gene')
    train_gene = get_all_gene(trainset_dir)
    test_gene = train_gene[:-1]
    print('Process the training data set into the same gene format')
    process_data_to_same_gene(train_gene, trainset_dir, trainset_dir, mode='train')
    print('Process the test data set into the same gene format')
    if os.listdir(testset_dir):
        process_data_to_same_gene(test_gene, testset_dir, testset_dir, mode='test')
    args = Setting()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epoch = args.epoch
    train_dataset_list = []
    test_dataset_list = []
    for filename in os.listdir(trainset_dir):
        if '_processed.h5' in filename:
            train_dataset_list.append(filename)
    for filename in os.listdir(testset_dir):
        if '_processed.h5' in filename:
            test_dataset_list.append(filename)

    dataloaders, label_num_list, feature_num = get_dataloaders_and_LabelNumList_and_FeatureNum(
        train_dataset_list, trainset_dir)


    model = Net(feature_num).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr
    )
    criterion = NPairLoss()
    print('train')
    model = train(model, dataloaders, optimizer, epoch, criterion, device, label_num_list)

    #test
    print('test')
    model.cpu().eval()
    metrics_list, labels_list = get_MetricsList_and_LabelsList(
        model, train_dataset_list, trainset_dir)
    if test_dataset_list:
        for i in range(len(test_dataset_list)):
            test_set = test_dataset_list[i]
            test_data = pd.read_hdf(testset_dir+test_set)
            pred_class = test(model, test_data, train_dataset_list,
                            metrics_list, labels_list)
            with open('result.txt', 'a') as result_file:
                for pred_class_indice in range(len(pred_class)):
                    result_file.write(str(test_data.index[pred_class_indice])+'\t'+pred_class[pred_class_indice]+'\n')

    torch.save(model.state_dict(), 'pre_trained/model/new_model.pth')
