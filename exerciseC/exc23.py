import argparse
import inspect
import numpy as np
from pyspark import Broadcast, SparkConf, SparkContext , RDD
from pyspark.sql import SparkSession
import pandas as pd
from collections.abc import Iterable
from numpy import ndarray
from pyspark.mllib.clustering import KMeans

class SummaryCluster:
    def __init__(self,N,SUM,SUMSQ,points_index):
        self.N: int = N
        self.SUM: ndarray = SUM
        self.SUMSQ: ndarray = SUMSQ
        self.points_index: list[int] = points_index
        
    @property
    def centroid(self) -> ndarray:
        return self.SUM / self.N
    
    @property
    def variance(self) -> ndarray:
        return self.SUMSQ / self.N - self.centroid ** 2

    def add_point(self, point: ndarray, index: int) -> None:
        self.N += 1
        self.SUM += point
        self.SUMSQ += point ** 2
        self.points_index.append(index)

    def add_points(self, points: list[ndarray], indexes: list[int]) -> None:
        self.N += len(points)
        self.SUM += np.sum(points, axis=0)
        self.SUMSQ += np.sum([p ** 2 for p in points], axis=0)
        self.points_index.extend(indexes)
        
    def merge(self, other: 'SummaryCluster') -> None:
        self.N += other.N
        self.SUM += other.SUM
        self.SUMSQ += other.SUMSQ
        self.points_index.extend(other.points_index)
        
    def mahalanobis_distance(self, point: ndarray) -> float:
        var = self.variance
        var[var == 0] = 1e-10
        d = ((point-self.centroid) / var)
        return d.dot(d) ** 0.5
    
    def mahalanobis_distance_another(self, other: 'SummaryCluster') -> float:
        avg_var = (self.variance + other.variance) / 2
        avg_var[avg_var == 0] = 1e-10
        d = ((self.centroid-other.centroid) / avg_var)
        return d.dot(d) ** 0.5

def mahalanobis_distance(point: ndarray, centroid: ndarray, var: ndarray) -> float:
    var[var == 0] = 1e-10
    d = ((point-centroid) / var)
    return d.dot(d) ** 0.5


class BFR:
    """
    BFR clustering algorithm implementation.
    """
    
    
    def __init__(self, sc: SparkContext, k:int, threshold:float=1.96,seed:int=42):
        """
        Initialize the BFR clustering algorithm.
        
        Parameters
        ----------
        sc : SparkContext
            Spark context for distributed computation.
            
        k : int
            Number of clusters to form.
            
        threshold : float
            Threshold for Mahalanobis distance.
            
        seed : int
            Random seed for reproducibility.
        """
        self.k: int = k
        self.threshold: float = threshold
        self.DSs: RDD[SummaryCluster] = []
        self.CSs: RDD[SummaryCluster] = []
        self.RS: RDD[tuple[int,ndarray]] = sc.parallelize([])
        self.seed: int = seed
        self.sc: SparkContext = sc
        self.centroid_mult:int = 4
        self.few_points_threshold: int = 3

    def fit(self,data: RDD[tuple[int,ndarray]], dim: int, n_points_per_iteration:int=2000) -> list["SummaryCluster"]:
        """
        Fit the BFR model to the data.
        
        Parameters
        ----------
        data : RDD[tuple[int,ndarray]]
            RDD of data points, where each point is a tuple of (index, features).
        dim : int
            Dimension of the data points.
        n_points_per_iteration : int
            Number of points to sample in each iteration.    
        """
        
        number_of_points = dim[0]
        fraction_of_points = n_points_per_iteration / number_of_points
        
        # 1. Load the data points from one file
        bag = data.sample(False, fraction_of_points ,self.seed)
        points = bag.map(lambda x: x[1])

        # 2. Run K-Means on the data points or a random subset of the data points
        kmeans = KMeans.train(points, self.k * self.centroid_mult, seed=self.seed, distanceMeasure="euclidean")
        labels = bag.map(lambda x: (kmeans.predict(x[1]),(x[0],x[1])))
        
        acc: dict[int,list[tuple[int,ndarray]]] = {}
        for predict , d in labels.collect():
            acc[predict] = acc.get(predict,[]) + [d]
        clusters = self.sc.parallelize(list(acc.items()))
        
        # 3. Among the result clusters from step 2, move all the clusters that contain only one or very few points
        # to RS as the outliers.
        few_b = self.sc.broadcast(self.few_points_threshold)
        outlier_clusters = clusters.filter(lambda x: len(x[1]) <= few_b.value)
        inlier_clusters = clusters.filter(lambda x: len(x[1]) > few_b.value)
        
        # 4. Run the clustering algorithm again to cluster the inlier data points into K clusters.
        inlier_points = inlier_clusters \
                        .flatMap(lambda x: x[1]) 
        inlier_points_real = inlier_points.map(lambda x: x[1])
        kmeans_inlier = KMeans.train(inlier_points_real, self.k, seed=self.seed, distanceMeasure="euclidean")
        labels_inlier = inlier_points.map(lambda x: (kmeans_inlier.predict(x[1]),(x[0],x[1])))
        
        acc2: dict[int,list[tuple[int,ndarray]]] = {}
        for predict , d in labels_inlier.collect():
            acc2[predict] = acc2.get(predict,[]) + [d]
        clusters_inlier = self.sc.parallelize(list(acc2.items()))
            
        # 5. Use these K clusters as DS. Discard these points and generate the DS statistics
        self.DSs = clusters_inlier.map(lambda x: SummaryCluster(
            len(x[1]),
            np.sum([point[1] for point in x[1]], axis=0),
            np.sum([point[1] ** 2 for point in x[1]], axis=0),
            [point[0] for point in x[1]]
        ))
        
        # 6. Run the clustering algorithm again to cluster the outlier data points using a large number of clusters
        outlier_points = outlier_clusters \
                        .flatMap(lambda x: x[1])
        outlier_points_real = outlier_points.map(lambda x: x[1])
        kmeans_outlier = KMeans.train(outlier_points_real, self.k * self.centroid_mult, seed=self.seed, distanceMeasure="euclidean")
        labels_outlier = outlier_points.map(lambda x: (kmeans_outlier.predict(x[1]),(x[0],x[1])))
        
        acc3: dict[int,list[tuple[int,ndarray]]] = {}
        for predict , d in labels_outlier.collect():
            acc3[predict] = acc3.get(predict,[]) + [d]
        clusters_outlier = self.sc.parallelize(list(acc3.items()))
    
        # 7. Generate CS and their statistics from the clusters with more than one data
        # point and use the remaining as new RS
        outlier_outliers_clusters = clusters_outlier.filter(lambda x: len(x[1]) < 2)
        outlier_inliers_clusters = clusters_outlier.filter(lambda x: len(x[1]) >= 2)
        
        self.CSs = outlier_inliers_clusters.map(lambda x: SummaryCluster(
            len(x[1]),
            np.sum([point[1] for point in x[1]], axis=0),
            np.sum([point[1] ** 2 for point in x[1]], axis=0),
            [point[0] for point in x[1]]
        ))
        
        new_rs = outlier_outliers_clusters.flatMap(lambda x: x[1])
        self.RS = new_rs
        # The above steps finish the initialization of DS. So far, we have K DS, some number
        # of CS , and some number of RS 
        ndim = dim[1]
        threshold_b = self.sc.broadcast((self.threshold) * (ndim ** 0.5))
        
        for iter in range(number_of_points // n_points_per_iteration):
            print(f"Iteration {iter+1:6d}/{number_of_points // n_points_per_iteration}", end="\r")
            # 8. Load the data points from another file
            bag = data.sample(False, fraction_of_points ,self.seed)
            
            # 9. For the new data points, compare them to each DS using the Mahalanobis Distance and assign them
            # to the nearest DS clusters if the distance is < α 𝑑^1/2 (e.g., 1.96 𝑑^1/2) -> 95% confidence
            
            ds_broadcast = self.sc.broadcast(self.DSs.map(lambda x: (x.centroid,x.variance,x.points_index[0],(x.N,x.SUM,x.SUMSQ,x.points_index))).collect())
            
            def find_closest_cluster(record,cl_b) -> tuple[int, tuple[int,ndarray]]:
                """
                For a single (id, point) record, scan all clusters to find the one
                with minimal Mahalanobis distance.  If that min distance < threshold,
                return that cluster, else None.
                """
                point_id, point = record
                min_dist = float("inf")
                best_cluster = None

                # scan all clusters locally
                for centroid , variance , fp , _ in cl_b.value:
                    dist = mahalanobis_distance(point, centroid, variance)
                    if dist < min_dist:
                        min_dist = dist
                        best_cluster = fp

                # apply threshold test
                if min_dist < threshold_b.value:
                    return (point_id, (best_cluster, point))
                else:
                    return (point_id, (None, point))
                
            def update_cluster(x: tuple[int, Iterable[tuple]],cl_b: Broadcast[list[tuple[ndarray, ndarray, int, tuple[int, ndarray, ndarray, list[int]]]]]):
                fp , it = x
                cur_cluster = [cdata for _ , _ , fp2 , cdata in cl_b.value if fp == fp2 ][0]
                cluster = SummaryCluster(
                    cur_cluster[0],
                    cur_cluster[1],
                    cur_cluster[2],
                    cur_cluster[3]
                )
                for s in it:
                    point_index , s2 = s
                    _ , point = s2
                    cluster.add_point(point, point_index)
                return cluster

            comparasion = bag.map(lambda x: find_closest_cluster(x,ds_broadcast))
            
            to_ds = comparasion.filter(lambda x: x[1][0] != None) 
            not_to_ds = comparasion.filter(lambda x: x[1][0] == None)
            
            new_ds = to_ds.groupBy(
                lambda x: x[1][0]
            ).map(
                lambda x: update_cluster(x,ds_broadcast)
            )
            self.DSs = new_ds
            # 10. For the new data points which are not assigned to any DS cluster, compare them to each of the CS
            # using the Mahalanobis Distance. For the new data points which are not assigned to any DS or CS cluster, add them to your RS
            
            cs_broadcast = self.sc.broadcast(self.CSs.map(lambda x: (x.centroid,x.variance,x.points_index[0],(x.N,x.SUM,x.SUMSQ,x.points_index))).collect())
            comparasion_cs = not_to_ds.map(lambda x: (x[0],x[1][1])).map(lambda x: find_closest_cluster(x, cs_broadcast))
            
            to_cs = comparasion_cs.filter(lambda x: x[1][0] != None)
            not_to_cs = comparasion_cs.filter(lambda x: x[1][0] == None)        
            
            new_cs = to_cs.groupBy(
                lambda x: x[1][0]
            ).map(
                lambda x: update_cluster(x,cs_broadcast)
            )
            
            new_rs = not_to_cs.map(
                lambda x: (x[0], x[1][1])
            )
            
            self.CSs = new_cs
            self.RS = self.RS.union(new_rs)
            # 11. Run the clustering algorithm on the RS with a large number of centroids (e.g., 3 or 5 time of K).
            # Generate CS and their statistics from the clusters with more than one data point and add them to
            # your existing CS list. Use the remaining points as your new RS.
            rs_points = self.RS.map(lambda x: x[1])
            kmeans_rs = KMeans.train(rs_points, self.k * self.centroid_mult, seed=self.seed, distanceMeasure="euclidean")
            labels_rs = self.RS.map(lambda x: (kmeans_rs.predict(x[1]),(x[0],x[1])))
            
            acc4: dict[int,list[tuple[int,ndarray]]] = {}
            for predict , d in labels_rs.collect():
                acc4[predict] = acc4.get(predict,[]) + [d]
            clusters_rs = self.sc.parallelize(list(acc4.items()))

            outliers_clusters_rs = clusters_rs.filter(lambda x: len(x[1]) < 2)
            inliers_clusters_rs = clusters_rs.filter(lambda x: len(x[1]) >= 2)
            
            new_cs = inliers_clusters_rs.map(lambda x: SummaryCluster(
                len(x[1]),
                np.sum([point[1] for point in x[1]], axis=0),
                np.sum([point[1] ** 2 for point in x[1]], axis=0),
                [point[0] for point in x[1]]
            ))
            new_rs = outliers_clusters_rs.flatMap(lambda x: x[1])
            self.CSs = self.CSs.union(new_cs)
            self.RS = new_rs
            
            # 12. Merge CS clusters that have a Mahalanobis Distance < α d ^ 1/2.
            cs_list = self.CSs.collect()
            merged_flags = [False] * len(cs_list)
            new_cs_list: list[SummaryCluster] = []

            for i, ci in enumerate(cs_list):
                if merged_flags[i]:
                    continue

                # try to merge any later cluster j (> i) into ci
                for j in range(i + 1, len(cs_list)):
                    if not merged_flags[j]:
                        cj = cs_list[j]
                        if ci.mahalanobis_distance_another(cj) < self.threshold * (ndim ** 0.5):
                            ci.merge(cj)
                            merged_flags[j] = True  
                new_cs_list.append(ci)
                
            self.CSs = self.sc.parallelize(new_cs_list)
                             
        # 13. Merge your CS and RS clusters into the closest DS clusters.
        ds_list = self.DSs.collect()
        for i in self.CSs.collect():
            dists = [ ds.mahalanobis_distance_another(i) for ds in ds_list ]
            min_dist = min(dists)
            index_of_closest = dists.index(min_dist)
            ds_list[index_of_closest].merge(i)
            
        for index, point in self.RS.collect():
            dists = [ ds.mahalanobis_distance(point) for ds in ds_list ]
            min_dist = min(dists)
            index_of_closest = dists.index(min_dist)
            ds_list[index_of_closest].add_points(point, [index])

        return ds_list
    

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
    rdd_ndarrays = rdd.map(lambda row: (int(row[0]),np.array(row[1:], dtype=np.float32) ))
    
    bfr = BFR(sc,k=args.num_clusters)
    res = bfr.fit(rdd_ndarrays,data.shape)
    
    with open("output.txt") as f:
        i = 0
        for c in res:
            i = i + 1
            f.write(f"{i}={str(c.points_index)}")
    
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
    conf = (SparkConf()
            .setAppName("BFR Clustering")
            .setMaster("local[16]")
            .set("spark.driver.memory", "8g")
    )
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")
    main()
    sc.stop()