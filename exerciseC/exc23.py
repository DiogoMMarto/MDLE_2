import argparse

import numpy as np
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession
from bfr import BFR
import pandas as pd

conf = None
sc = None

def main():
    argparser = argparse.ArgumentParser(description="Cluster analysis")
    argparser.add_argument("input_file_data", help="Path to data file")
    argparser.add_argument("input_file_track", help="Path to track file") 
    argparser.add_argument("-k","--num_clusters", type=int, default=11, help="Number of clusters")
    args = argparser.parse_args()
    
    data = pd.read_csv(args.input_file_data, index_col=0, header=[0, 1, 2])
    tracks = pd.read_csv(args.input_file_track, index_col=0, header=[0, 1])
    
    cols_to_use = [col for col in data.columns if col[1] == 'mean']
    data: pd.DataFrame = data[cols_to_use]
    
    ss = SparkSession(sc)
    spark_df = ss.createDataFrame(data.reset_index()) 
    
    rdd = spark_df.rdd
    rdd_ndarrays = rdd.map(lambda row: (int(row[0]),np.array(row[1:], dtype=np.float32) )).repartition(1000)
    
    bfr = BFR(sc,k=args.num_clusters)
    res = bfr.fit(rdd_ndarrays,data.shape)
    
    for i, cluster in enumerate(res):
        print(f"Cluster {i}:")
        print(f"  Centroid: {cluster['centroid']}")
        print(f"  Radius: {cluster['radius']}")
        print(f"  Diameter: {cluster['diameter']}")
        print(f"  Density (radius): {cluster['density_r']}")
        print(f"  Density (diameter): {cluster['density_d']}")
        print(f"  Number of points: {cluster['n_points']}")
        print()

if __name__ == "__main__":
    conf = SparkConf().setAppName("BFR Clsutering")
    sc = SparkContext(conf=conf)
    # sc.setLogLevel("ERROR")
    main()
    sc.stop()