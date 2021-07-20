#%%
# multi-modal correlation
obs_name1 = '360d'
obs_name2 = '030d'

graph_metric_ttest(subjects, t1_mask, 
                    out_csv_path_ske = r'G:\006pd_DTI\04_structural_network_node\07_newanalysis_360_30\{}.csv',
                    out_nii_path_ske = r'G:\006pd_DTI\04_structural_network_node\07_newanalysis_360_30\{}.nii',
                    obs_name1=obs_name1, obs_name2=obs_name2)
gmv_ttest(subjects, t1_mask,
            out_csv_path=r'G:\006pd_DTI\04_structural_network_node\07_newanalysis_360_30\gmv.csv',
            out_nii_path=r'G:\006pd_DTI\04_structural_network_node\07_newanalysis_360_30\gmv.nii',
            obs_name1=obs_name1, obs_name2=obs_name2)
cbf_ttest(subjects, t1_mask,
            out_csv_path=r'G:\006pd_DTI\04_structural_network_node\07_newanalysis_360_30\cbf.csv',
            out_nii_path=r'G:\006pd_DTI\04_structural_network_node\07_newanalysis_360_30\cbf.nii',
            obs_name1=obs_name1, obs_name2=obs_name2)

#%%
# mutlimodal correlation
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
csv_path = r'G:\degree.csv'
df = pd.read_csv(csv_path, index_col=0)

cols = ['gmvt', 'cbft']

x = []
y = []
x = df['degreet'].values
for col in cols:
    y = df[col].values
    r, p = spearmanr(x, y)
    sns.jointplot(x, y, kind="hex")
    ax = sns.regplot(x=x,y=y, robust=True, scatter=False)
    ax.set_xlabel('spearman r: {:.2f}, p-value: {:.2e}'.format(r, p))
    plt.show()
    plt.close()
