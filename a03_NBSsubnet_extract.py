#%%
from scipy.io import loadmat
import numpy as np
import os

def readin_NBS_subnetwork(mat_path, index=0):
    mat = loadmat(mat_path)
    subnetwork = mat['nbs']['NBS'][0][0]['con_mat'][0][0][0][index]
    return subnetwork

def extract_subnetwork_nodes(subnetwork):
    row_nodes, col_nodes = subnetwork.nonzero()
    all_nodes = np.concatenate((row_nodes, col_nodes), axis=0)
    all_nodes_unique = np.unique(all_nodes)
    return all_nodes_unique, row_nodes, col_nodes

def extract_submatrix(matrix, nodes):
    return matrix[np.ix_(nodes, nodes)]

def whole_brain_for_brant(subjects, n=500, out_path='./brant'):
    #normalization of whole-brain structural networks for Brant (graph theory metrics calculation)
    for subject in subjects:
        for observation in subject.get_all_observation():
            dti = observation.dti
            network = dti.get_network_num()
            network = network / n
            np.fill_diagonal(network, 0)
            to_path = os.path.join(out_path, '{}_{}_num.txt'.format(subject.name, observation.name))
            np.savetxt(to_path, network, fmt='%10f')

def subnetwork_for_brant(subjects, n=500, out_path='./brant',
                          subnetwork_path='./subnetwork/NBS.mat'):
    #normalization of subnetworks for Brant (graph theory metrics calculation)
    subnetwork = readin_NBS_subnetwork(subnetwork_path)
    all_nodes_unique, *_ = extract_subnetwork_nodes(subnetwork)

    for subject in subjects:
        for observation in subject.get_all_observation():
            dti = observation.dti
            network = dti.get_network_num()
            submatrix = extract_submatrix(network, all_nodes_unique)
            submatrix = submatrix / n
            np.fill_diagonal(submatrix, 0)
            to_path = os.path.join(out_path, '{}_{}_num.txt'.format(subject.name, observation.name))
            np.savetxt(to_path, submatrix, fmt='%10f')