#%%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
# %%
# line graph for tremor
df = pd.read_csv('./CRST-T.csv', index_col=0)

x = np.arange(1,df.shape[1] + 1)

fig, ax = plt.subplots()
p = []
l = []
for index, row in df.iterrows():
    l.append(str(index))
    p.append(ax.plot(x, row.to_numpy(), marker='o'))
ax.grid(False)
ax.legend(p, labels=l, loc=(1.1, 0))
#%%
# line graph for degree
p = []
l = []

for subject in subjects:
    y = []
    x = np.arange(1, len(subject.get_observations()) + 1)
    for obs in subject.get_observations():
        if obs.name == 'ncnc':
            continue
        dti = obs.dti
        metric = dti.get_global_metric('degree', 'brant.mat', threshold_value=0.006)

        y.append(metric)
    l.append(str(subject.name))
    p.append(ax.plot(x, row.to_numpy(), marker='o'))
    
ax.grid(False)
ax.legend(p, labels=l, loc=(1.1, 0))


# %%
# plot top value bar
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

csv_path = r'D:\30vs360.csv'
out_path = r'D:\result'

df = pd.read_csv(csv_path, index_col=0)

value_col_name = 'gmvt'
sort_col_name = 'gmvp'

sorted_df = df.sort_values(sort_col_name, ascending=True)
head_values = sorted_df[value_col_name][:10]
head_roi_names = sorted_df['Fullname'][:10]


fig, ax = plt.subplots()
y_pos = np.arange(10)
ax.barh(y_pos, np.abs(head_values))
ax.set_yticks(y_pos)
ax.set_yticklabels(head_roi_names)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('t-value')

plt.show()
