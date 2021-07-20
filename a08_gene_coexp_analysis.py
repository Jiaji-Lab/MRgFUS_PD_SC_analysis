# %%
# gene co-expression analysis
import os
import csv
import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
import copy

import subnetwork_extract
from statistical_analysis.correlation import correlation

type = 'whole_brain'

# load gene expression
gene_df = pd.read_csv(gene_path, index_col=0)
gene_names = list(gene_df)
corr = gene_df.corr()

if type == 'whole_brain':
    all_nodes_unique = np.arange(1, 247)
elif type == 'subnetwork':
    # load subnetwork
    subnetwork = subnetwork_extract.readin_NBS_subnetwork(r'./NBS.mat')
    all_nodes_unique, *_ = subnetwork_extract.extract_subnetwork_nodes(subnetwork)
    # python index to label index
    all_nodes_unique = all_nodes_unique + 1

# load features t-values
metric_path = r'./degree_t.csv'
t_values_df = pd.read_csv(metric_path, index_col=0)
nodal_dict = dict(zip(t_values_df.index, t_values_df['t-value'].values))

out_path = r'./result/'

n_clusters = [10]

with open(os.path.join(out_path, 'pearson_r.txt'), 'w') as f:
    for n_cluster in n_clusters:
        cluster_model = AgglomerativeClustering(n_clusters=n_cluster).fit(corr)

        with open(os.path.join(out_path, 'labels_{}_cluster.csv'.format(n_cluster)), 'w', newline='') as file:
            fieldnames = ['gene_name', 'label']
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
                    
            for gene_name, label in zip(gene_names, cluster_model.labels_):
                writer.writerow({'gene_name': gene_name, 'label': label})

        ## perform correlation
        i = 0

        eigengenes = []
        for label in np.unique(cluster_model.labels_):
            tmp_labels = copy.deepcopy(cluster_model.labels_)
            tmp_labels[tmp_labels==label] = -1
            tmp_labels[tmp_labels!=-1] = 0
            nonzeroind = np.nonzero(tmp_labels)[0]
            names = []
            for ind in nonzeroind:
                names.append(gene_names[ind])
            result = gene_df[names].values.T
            eigengene = PCA(1).fit(result).components_
            eigengenes.append(eigengene.reshape(-1))

        for eigengene in eigengenes:
            gene_dict = dict(zip(gene_df.index, eigengene))
            x = []
            y = []
            for node in all_nodes_unique:
                try:
                    x.append(gene_dict[node])
                    y.append(nodal_dict[node])
                except KeyError:
                    pass
            r,p = correlation(x, y)
            if p < 0.05:
                print('cluster:{}, center:{}, r:{:.2f}, p:{:.2e}'.format(n_cluster, i, r, p), file=f)
            i += 1
        print('\n', file=f)
