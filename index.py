import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statistics as stats
import scipy.stats as sp
from itertools import chain

sns.set_theme(style="whitegrid")
df_demo = pd.read_excel("./xlsx/demo14.xlsx", sheet_name="Возр. группы", header=None, skiprows=lambda x: x in chain(range(11), range(20, 75)), usecols="AB", names=["Demo_count"]).sort_values(by=["Demo_count"], ascending=True).reset_index(drop=True)
skip = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 29, 33, 46, 53, 61, 76, 80, 81, 85, 98]
df = pd.DataFrame.dropna(pd.read_excel("./xlsx/1-адм_таб1.xlsx", sheet_name="Таблица 1", header=None, skiprows=skip, usecols=[1], names=["Municipal_count"])).sort_values(by=["Municipal_count"], ascending=True).reset_index(drop=True)
df = df.astype({"Municipal_count":"int16"})

def weighted_average(dataframe, value, weight):
	val = dataframe[value]
	wt = dataframe[weight]
	return (val * wt).sum() / wt.sum()

def properties(df, x):
	M_x = np.mean(df[x])
	var = stats.variance(df[x], M_x)
	symmetry = sp.skew(df[x], axis=0, bias=False)
	excess = sp.kurtosis(df[x], axis=0, bias=False)
	quant0_05 = np.quantile(df[x], .05)
	quant0_95 = np.quantile(df[x], .95)
	percent_point_2_5 = np.quantile(df[x], .975)
	return {"M_x": M_x,"var": var,"symmetry": symmetry,"excess": excess,"quants": [quant0_05, quant0_95],"percent_point": percent_point_2_5}

Municipal_properties = properties(df, "Municipal_count")
sns.displot(df, x="Municipal_count", kind="kde", bw_adjust=.9, cut=0, color="cyan")
sns.lineplot(df, x="Municipal_count", y=sp.norm.pdf(df["Municipal_count"], Municipal_properties["M_x"], np.sqrt(Municipal_properties["var"])), color="red")
plt.gcf().set_size_inches(10,6)
plt.show()
demo_properties = properties(df_demo, "Demo_count")
print(df_demo)