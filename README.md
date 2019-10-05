# redhook-analysis

This repository is for a data analysis project dealing with noise pollution in Red Hook. The goal is to model the relationship between the occurences of trucks passing by a SONYC sensor and the SPL level, since the residents of Red Hook believe there is a correlation between the high volume of truck activity and the high noise level in the neighborhood.

The structure of the repository is as follows:

notebooks: All of the jupyter notebooks used for data processing and analysis.

data: Various data files that were analyzed. Some are not included because they are too large, so the data files in this directory aren't representative of all the data needed to produce informative models.

modules: Modules containing the functions used in the notebooks to analyze data. These modules are divided up into different categories depending on their intended use.

scripts: Scripts used for cluster identification. These are scripts because the large amount of data being processed made the running time take a few hours.

output: Output for the scripts used for cluster identification.

visuals: Sample plots demonstrating the relationship between truck activity and SPL level in Red Hook. These were generated using the modeling functions.