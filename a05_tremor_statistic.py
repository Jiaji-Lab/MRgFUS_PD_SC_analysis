#%%
# violin plot for graph metrics across timepoints
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from datasets.subject import load_subjects

subjects = load_subjects(r'G:\006pd_DTI\subject_info.csv')

threshold_value = 0.006
out_path = r'G:\006pd_DTI\04_structural_network_analysis\01_whole_brain\04_python_all_comparison'

metrics = ['assortative_mixing', 'neighbor_degree',
            'betweenness_centrality', 'degree',
            'fault_tolerance', 'shortest_path_length',
            'global_efficiency', 'clustering_coefficient',
            'local_efficiency',
            'vulnerability', 'transitivity']

for metric in metrics:
    subjects_values = []
    columns = []
    for subject in subjects:
        observations = subject.get_all_observation()
        values = []
        for obs in observations:
            degree = obs.dti.get_global_metric(metric, mat_name='brant.mat', threshold_value=threshold_value)
            columns.append(obs.name)
            subjects_values.append(float(degree))

    subjects_values = np.array(subjects_values)
    columns = np.array(columns)
    new_array = np.stack((subjects_values, columns), axis=-1)


    index = ['subject{}'.format(i//5) for i in range(5,50)]
    df = pd.DataFrame(data=new_array,
                        index=index,
                        columns=['Value', 'Observation'])
    df[['Value']] = df[['Value']].astype(float)

    ax = sns.violinplot(x='Observation', y="Value", data=df)
    ax.set_title('whole_network_{}'.format(metric))
    
    out_png_path = os.path.join(out_path, 'whole_network_{}.png'.format(metric))
    out_csv_path = os.path.join(out_path, 'whole_network_{}.csv'.format(metric))
    plt.savefig(out_png_path)
    plt.close()
    df.to_csv(out_csv_path)

#%%
# correlation between graph metrics and tremor assessments
import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from datasets import mask
from datasets.subject import load_subjects
from statistical_analysis.correlation import correlation
from a04_load_data import load_subjects_surgery_info, load_subjects_global_metric

t1_mask_path = r'G:\006pd_DTI\02_mask\brainnetome\rBN_Atlas_246_1mm.nii'
t1_mask = mask.NiiMask(t1_mask_path)

subjects = load_subjects(r'G:\006pd_DTI\subject_info.csv')

# seaborn relplot
obs_name1 = 'base'
obs_name2 = '030d'
mat_name = 'brant_30d_subnetwork.mat'

out_dir = r'D:\subnetwork_{}_vs_{}_surgury'.format(obs_name1, obs_name2)

if not os.path.exists(out_dir):
    os.mkdir(out_dir)

threshold_value = 0.006
graph_metric_names = ['assortative_mixing', 'neighbor_degree',
                    'betweenness_centrality', 'degree',
                    'fault_tolerance', 'shortest_path_length',
                    'global_efficiency', 'clustering_coefficient',
                    'local_efficiency',
                    'vulnerability', 'transitivity']

delta_info_names = ['handtremor', 'CRSTA_total', 'CRST b_total', 'CRST C', 'CRST TOTAL']

df_dict = {}
i = 0
x = 1
for info_name in delta_info_names:
    obs1_info = load_subjects_surgery_info(subjects, obs_name1, info_name)
    obs2_info = load_subjects_surgery_info(subjects, obs_name2, info_name)
    delta_info = obs2_info - obs1_info

    y = 1
    for graph_metric_name in graph_metric_names:
        obs1_graph_metric = load_subjects_global_metric(subjects, obs_name1, mat_name, graph_metric_name, threshold_value)
        obs2_graph_metric = load_subjects_global_metric(subjects, obs_name2, mat_name, graph_metric_name, threshold_value)
        delta_graph_metric = obs2_graph_metric - obs1_graph_metric

        r, p = correlation(delta_info, delta_graph_metric, show=False)
        if p < 0.05:
            correlation(delta_info, delta_graph_metric, show=False, save=True,
                        x_label=info_name, y_label=graph_metric_name,
                        out_path=os.path.join(out_dir ,r'Δ{}_Δ{}_{}.png'.format(info_name, graph_metric_name, mat_name[:-4])))
        if p > 0.05:
            r = 0
        row = [x, y, r, abs(r), info_name, graph_metric_name]
        y += 1
        i += 1
        df_dict[i] = row

    x += 1

df = pd.DataFrame(df_dict, index=['x','y', 'r', 'absr', 'f1','f2']).T

sns.set_theme(style="whitegrid")
g = sns.relplot(data=df, x="x", y="y", hue="r", size="absr",
    palette="vlag", hue_norm=(-1, 1), edgecolor=".7",
    height=10, sizes=(10, 250), size_norm=(0, .8))

# Tweak the figure to finalize
g.set(xlabel='', ylabel='', aspect="equal")
g.despine(left=True, bottom=True)

g.ax.set_xticks(np.arange(1, len(delta_info_names)+1))
delta_info_names = ['Δ'+xlabel for xlabel in delta_info_names]
xlabels = delta_info_names
g.ax.set_xticklabels(xlabels)

ylabels = ['Δ'+ylabel for ylabel in graph_metric_names]
g.ax.set_yticks(np.arange(1, len(graph_metric_names)+2))
g.ax.set_yticklabels(ylabels)
g.ax.set_title('{}_{}_{}'.format(obs_name1, obs_name2, mat_name))

for label in g.ax.get_xticklabels():
    label.set_rotation(90)
for artist in g.legend.legendHandles:
    artist.set_edgecolor(".7")
plt.show()