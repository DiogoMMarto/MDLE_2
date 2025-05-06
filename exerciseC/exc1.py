"""
This script performs cluster analysis on a given dataset using Agglomerative Clustering. 
It computes various metrics for each cluster, such as radius, diameter, and density, 
and visualizes the results for different numbers of clusters (k).

Key functionalities:
1. Reads two input CSV files: one containing the data and another containing track information.
2. Filters and preprocesses the data based on specific criteria.
3. Applies Agglomerative Clustering for a range of cluster numbers (k = 8 to 16).
4. Computes metrics for each cluster, including:
   - Number of points in the cluster
   - Radius and diameter of the cluster
   - Density based on radius and diameter
5. Aggregates and averages the metrics for each value of k.
6. Outputs the results and visualizes the metrics (radius, diameter, and densities) 
   as a function of the number of clusters.

The script is designed to be run from the command line with the following arguments:
- `input_file`: Path to the main data file.
- `input_file_2`: Path to the track file.

The results are displayed in the console and visualized using matplotlib.

Author: Diogo Marto 
Date: 25-04-2025

Code written with help from automatic code generation tools namely github copilot.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import pdist
import argparse
import matplotlib.pyplot as plt

argparser = argparse.ArgumentParser(description="Cluster analysis")
argparser.add_argument("input_file", help="Path to data file")
argparser.add_argument("input_file_2", help="Path to track file") 
args = argparser.parse_args()
data = pd.read_csv(args.input_file, index_col=0, header=[0, 1, 2])
tracks = pd.read_csv(args.input_file_2, index_col=0, header=[0, 1])
subset = tracks["set"]["subset"]
data = data[tracks["set"]["subset"] == "small"]
cols_to_use = [col for col in data.columns if col[1] in ['mean']]
data = data[cols_to_use]
data

results = {}

def compute_metrics(data, labels, k):
    unique_labels = np.unique(labels)
    res = {}
    
    for label in unique_labels:
        cluster_points = data[labels == label]
        n_points = len(cluster_points)
        
        if n_points == 0:
            print(f"Warning: No points found for label {label} K={k}.")
            continue
        
        centroid = np.mean(cluster_points, axis=0)
        squared_distances = np.sum((cluster_points - centroid) ** 2, axis=1)
        r2 = np.mean(squared_distances)
        radius = np.sqrt(r2)
        
        if n_points == 1:
            print(f"Warning: Only one point in cluster {label} K={k}.")
            d2 = 0
            diameter = 0
        else:
            pairwise_distances = pdist(cluster_points, metric='euclidean')
            d2 = np.mean(pairwise_distances ** 2)
            diameter = np.sqrt(d2)
            
        if radius > 0:
            desnity_r = n_points / r2
        else:
            desnity_r = 0
            
        if diameter > 0:
            desnity_d = n_points / d2
        else:
            desnity_d = 0
            
        res[int(label)] = {
            'n_points': float(n_points),
            'radius': float(radius),
            'diameter': float(diameter),
            'density_r': float(desnity_r),
            'density_d': float(desnity_d)
        }
    
    return res
        

for k in range(8, 17):
    model = AgglomerativeClustering(n_clusters=k, linkage='ward')
    labels = model.fit_predict(data)
    m = compute_metrics(data, labels, k)
    
    avg_radius    = float(np.mean([v['radius'] for v in m.values()]))
    avg_diameter  = float(np.mean([v['diameter'] for v in m.values()]))
    avg_density_r = float(np.mean([v['density_r'] for v in m.values()]))
    avg_density_d = float(np.mean([v['density_d'] for v in m.values()]))
    results[k] = {
        'radius': avg_radius,
        'diameter': avg_diameter,
        'density_r': avg_density_r,
        'density_d': avg_density_d,
        'cluster': m
    }

# Output results for each k
for k, metrics in results.items():
    print(f"k={k}:{metrics}")
    
# Plotting the results over k 
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
ksall = [i+8 for i,k in enumerate(results.keys()) for j in range(i+8)]
ks = results.keys()
ds = [metrics['diameter'] for metrics in results.values()]
dsall = [i["diameter"] for i in results.values() for i in i["cluster"].values()]
plt.plot(ks, ds, marker='o')
plt.scatter(ksall, dsall, marker='o', color='red',alpha=0.5)
plt.title('Diameter vs. Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Diameter')
plt.grid()

plt.subplot(2, 2, 2)
rs = [metrics['radius'] for metrics in results.values()]
rsall = [i["radius"] for i in results.values() for i in i["cluster"].values()]
plt.plot(ks, rs, marker='o')
plt.scatter(ksall, rsall, marker='o', color='red',alpha=0.5)
plt.title('Radius vs. Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Radius')
plt.grid()

plt.subplot(2, 2, 3)
dds = [metrics['density_d'] for metrics in results.values()]
ddsall = [i["density_d"] for i in results.values() for i in i["cluster"].values()]
plt.plot(ks, dds, marker='o')
plt.scatter(ksall, ddsall, marker='o', color='red',alpha=0.5)
plt.title('D2 Density vs. Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('D2 Density')
plt.grid()

plt.subplot(2, 2, 4)
drs = [metrics['density_r'] for metrics in results.values()]
dsrall = [i["density_r"] for i in results.values() for i in i["cluster"].values()]
plt.plot(ks, drs, marker='o')
plt.scatter(ksall, dsrall, marker='o', color='red',alpha=0.5)
plt.title('R2 Density vs. Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('R2 Density')
plt.grid()

plt.tight_layout()
plt.show()
