#%%
# correlation between graph metrics and surgical info and MRgFUS lesion measures
import numpy as np
import pandas as pd
import seaborn as sns

from statistical_analysis.correlation import correlation
from a04_load_data import load_subjects_surgery_info, load_subjects_global_metric

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

delta_info_names = ['injury']

base_info_names = ['Number of sonications','Mean maximal energy delivered (J)','Mean Power(w)',
                    'maximal temperature','Mean maximal temperature','Number of sonications above 54',
                    'Number of sonications','Mean duration(sec)']

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

    # Injury with info
    injury1_info = load_subjects_surgery_info(subjects, obs_name1, 'injury')
    injury2_info = load_subjects_surgery_info(subjects, obs_name2, 'injury')
    delta_injury_info = injury2_info - injury1_info
    r, p = correlation(delta_info, delta_injury_info, show=False)
    if p > 0.05:
            r = 0
    row = [x, y, r, abs(r), info_name, 'injury']
    y += 1
    i += 1
    df_dict[i] = row

    x += 1

for info_name in base_info_names:
    obs1_info = load_subjects_surgery_info(subjects, 'base', info_name)

    y = 1
    for graph_metric_name in graph_metric_names:
        obs1_graph_metric = load_subjects_global_metric(subjects, obs_name1, mat_name, graph_metric_name, threshold_value)
        obs2_graph_metric = load_subjects_global_metric(subjects, obs_name2, mat_name, graph_metric_name, threshold_value)
        delta_graph_metric = obs2_graph_metric - obs1_graph_metric

        r, p = correlation(obs1_info, delta_graph_metric, show=False)
        if p < 0.05:
            correlation(obs1_info, delta_graph_metric, show=False, save=True,
                        x_label=info_name, y_label=graph_metric_name,
                        out_path=os.path.join(out_dir ,r'{}_Δ{}_{}.png'.format(info_name, graph_metric_name, mat_name[:-4])))
        else:
            r = 0
        row = [x, y, r, abs(r), info_name, graph_metric_name]
        y += 1
        i += 1
        df_dict[i] = row

    # Injury with info
    injury1_info = load_subjects_surgery_info(subjects, obs_name1, 'injury')
    injury2_info = load_subjects_surgery_info(subjects, obs_name2, 'injury')
    delta_injury_info = injury2_info - injury1_info
    r, p = correlation(obs1_info, delta_injury_info, show=False)
    if p > 0.05:
            r = 0
    row = [x, y, r, abs(r), info_name, 'injury']
    y += 1
    i += 1
    df_dict[i] = row

    x += 1

df = pd.DataFrame(df_dict, index=['x','y', 'r', 'absr', 'f1','f2']).T
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")
g = sns.relplot(
    data=df,
    x="x", y="y", hue="r", size="absr",
    palette="vlag", hue_norm=(-1, 1), edgecolor=".7",
    height=10, sizes=(10, 250), size_norm=(0, .8),
)

# Tweak the figure to finalize
g.set(xlabel='', ylabel='', aspect="equal")
g.despine(left=True, bottom=True)


g.ax.set_xticks(np.arange(1, len(delta_info_names)+len(base_info_names)+1))
delta_info_names = ['Δ'+xlabel for xlabel in delta_info_names]
xlabels = delta_info_names+base_info_names
g.ax.set_xticklabels(xlabels)

ylabels = ['Δ'+ylabel for ylabel in graph_metric_names] + ['Δinjury']
g.ax.set_yticks(np.arange(1, len(graph_metric_names)+2))
g.ax.set_yticklabels(ylabels)
g.ax.set_title('{}_{}_{}'.format(obs_name1, obs_name2, mat_name))

for label in g.ax.get_xticklabels():
    label.set_rotation(90)
for artist in g.legend.legendHandles:
    artist.set_edgecolor(".7")
plt.show()
# %%
# correlation between injury delta and degree delta respectively 
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

mat_name = 'brant_30d_subnetwork.mat'
threshold_value = 0.006
obs1_graph_metric = load_subjects_global_metric(subjects, 'base', mat_name, 'degree', threshold_value)
obs2_graph_metric = load_subjects_global_metric(subjects, '030d', mat_name, 'degree', threshold_value)

obs1_info = load_subjects_surgery_info(subjects, 'base', 'injury')
obs2_info = load_subjects_surgery_info(subjects, '030d', 'injury')

x = obs2_graph_metric - obs1_graph_metric
y = obs2_info - obs1_info

r, p = correlation(x, y, show=True)
plt.show()