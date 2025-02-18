#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats

epa_data = pd.read_csv("c4_epa_air_quality.csv", index_col = 0)

epa_data.head(10)

epa_data.describe(include='all')


### YOUR CODE HERE ###

population_mean = epa_data['aqi'].mean()
population_mean

sampled_data.head(10)

sample_mean = sampled_data['aqi'].mean()
sample_mean

estimate_list = []
for i in range(10000):
    estimate_list.append(epa_data['aqi'].sample(n=50,replace=True).mean())


estimate_df = pd.DataFrame(data={'estimate': estimate_list})
estimate_df


mean_sample_means = estimate_df['estimate'].mean()
mean_sample_means

estimate_df['estimate'].hist()

standard_error = sampled_data['aqi'].std() / np.sqrt(len(sampled_data))
standard_error

### YOUE CODE HERE ###
plt.figure(figsize=(8,5))
plt.hist(estimate_df['estimate'], bins=25, density=True, alpha=0.4, label = "histogram of sample means of 10000 random samples")
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100) # generate a grid of 100 values from xmin to xmax.
p = stats.norm.pdf(x, population_mean, standard_error)
plt.plot(x, p, 'k', linewidth=2, label = 'normal curve from central limit theorem')
plt.axvline(x=population_mean, color='m', linestyle = 'solid', label = 'population mean')
plt.axvline(x=sample_mean, color='r', linestyle = '--', label = 'sample mean of the first random sample')
plt.axvline(x=mean_sample_means, color='b', linestyle = ':', label = 'mean of sample means of 10000 random samples')
plt.title("Sampling distribution of sample mean")
plt.xlabel('sample mean')
plt.ylabel('density')
plt.legend(bbox_to_anchor=(1.04,1));
