#%%
# Juspace PET correlation with delta degree
## whole brain
import numpy as np
import pandas as pd
from statistical_analysis.correlation import correlation, hexplot
from scipy.stats import ttest_ind
import a03_NBSsubnet_extract

obs_name1 = '030d'
obs_name2 = '360d'
threshold_value = 0.006
nodal_metric_names = ['degree']

juspace_dir = r'G:\006pd_DTI\08_Juspace_PET\masked_mean'
juspace_files = os.listdir(juspace_dir)


subnetwork = a03_NBSsubnet_extract.readin_NBS_subnetwork(r'G:\006pd_DTI\NBS.mat')
all_nodes_unique, *_ = a03_NBSsubnet_extract.extract_subnetwork_nodes(subnetwork)

out_path = r'G:\006pd_DTI\08_Juspace'

for metric_name in nodal_metric_names:
    a, b = a03_NBSsubnet_extract.load_subjects_nodal_metric(subjects, obs_name1, obs_name2,
                                      metric_name, threshold_value)
    x, p = ttest_ind(a.T, b.T)
    sub_x = x[np.ix_(all_nodes_unique)]
    for f in juspace_files:
        path = os.path.join(juspace_dir, f)
        df = pd.read_csv(path, index_col=0)
        y = df['Volume'].array
        sub_y = y[np.ix_(all_nodes_unique)]
        r, p = correlation(x, y, show=False)
        if p < 0.05:
            r, p = hexplot(x, y, save=True, x_label=metric_name, y_label=f[:-4], out_path = os.path.join(out_path, 'w_{}_{}.png'.format(metric_name, f[:-4])))
        r, p = correlation(sub_x, sub_y, show=False)
        if p < 0.05:
            r, p = hexplot(sub_x, sub_y, save=True, x_label=metric_name, y_label=f[:-4], out_path = os.path.join(out_path, 's_{}_{}.png'.format(metric_name, f[:-4])))


