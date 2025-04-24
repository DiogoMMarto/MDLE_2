import argparse
import inspect
import numpy as np
from pyspark import SparkConf, SparkContext , RDD
from pyspark.sql import SparkSession
import pandas as pd
from collections.abc import Iterable
from numpy import ndarray
from pyspark.mllib.clustering import KMeans

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
        self.few_points_threshold: int = 10

    def fit(self,data: RDD[tuple[int,ndarray]], dim: int, n_points_per_iteration:int=1000) -> list["SummaryCluster"]:
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
        
        print(f"[DEBUG] Line {inspect.currentframe().f_lineno}")
        print(f"[DEBUG] Line {inspect.currentframe().f_lineno}")
        print(f"[DEBUG] Line {inspect.currentframe().f_lineno}")
    
        # 2. Run K-Means on the data points or a random subset of the data points
        kmeans = KMeans.train(points, self.k * self.centroid_mult, seed=self.seed, distanceMeasure="euclidean")
        labels = bag.map(lambda x: (kmeans.predict(x[1]),(x[0],x[1])))
        
        clusters = labels.groupByKey()
        
        print(f"[DEBUG] Line {inspect.currentframe().f_lineno}")
        print(f"[DEBUG] Line {inspect.currentframe().f_lineno}")
        print(f"[DEBUG] Line {inspect.currentframe().f_lineno}")
        
        # 3. Among the result clusters from step 2, move all the clusters that contain only one or very few points
        # to RS as the outliers.
        few_b = self.sc.broadcast(self.few_points_threshold)
        outlier_clusters = clusters.filter(lambda x: len(x[1]) <= few_b.value)
        inlier_clusters = clusters.filter(lambda x: len(x[1]) > few_b.value)
        
        print(f"[DEBUG] Line {inspect.currentframe().f_lineno}")
        print(f"[DEBUG] Line {inspect.currentframe().f_lineno}")
        print(f"[DEBUG] Line {inspect.currentframe().f_lineno}")
        
        # 4. Run the clustering algorithm again to cluster the inlier data points into K clusters.
        inlier_points = inlier_clusters \
                        .flatMap(lambda x: x[1]) 
        inlier_points_real = inlier_points.map(lambda x: x[1])
        kmeans_inlier = KMeans.train(inlier_points_real, self.k, seed=self.seed, distanceMeasure="euclidean")
        labels_inlier = inlier_points.map(lambda x: (kmeans_inlier.predict(x[1]),(x[0],x[1])))
        clusters_inlier = labels_inlier.groupByKey()
        
        print(f"[DEBUG] Line {inspect.currentframe().f_lineno}")
        print(f"[DEBUG] Line {inspect.currentframe().f_lineno}")
        print(f"[DEBUG] Line {inspect.currentframe().f_lineno}")
        
        # 5. Use these K clusters as DS. Discard these points and generate the DS statistics
        self.DSs = clusters_inlier.map(lambda x: SummaryCluster(
            len(x[1]),
            np.sum([point[1] for point in x[1]], axis=0),
            np.sum([point[1] ** 2 for point in x[1]], axis=0),
            [point[0] for point in x[1]]
        ))
        
        print(f"[DEBUG] Line {inspect.currentframe().f_lineno}")
        print(f"[DEBUG] Line {inspect.currentframe().f_lineno}")
        print(f"[DEBUG] Line {inspect.currentframe().f_lineno}")
        
        # 6. Run the clustering algorithm again to cluster the outlier data points using a large number of clusters
        outlier_points = outlier_clusters \
                        .flatMap(lambda x: x[1])
        outlier_points_real = outlier_points.map(lambda x: x[1])
        kmeans_outlier = KMeans.train(outlier_points_real, self.k * self.centroid_mult, seed=self.seed, distanceMeasure="euclidean")
        labels_outlier = outlier_points.map(lambda x: (kmeans_outlier.predict(x[1]),(x[0],x[1])))
        clusters_outlier = labels_outlier.groupByKey()
        
        print(f"[DEBUG] Line {inspect.currentframe().f_lineno}")
        print(f"[DEBUG] Line {inspect.currentframe().f_lineno}")
        print(f"[DEBUG] Line {inspect.currentframe().f_lineno}")
        
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
        self.RS = self.sc.parallelize(self.RS.collect())
        
        print(f"[DEBUG] Line {inspect.currentframe().f_lineno}")
        print(f"[DEBUG] Line {inspect.currentframe().f_lineno}")
        print(f"[DEBUG] Line {inspect.currentframe().f_lineno}")
        # The above steps finish the initialization of DS. So far, we have K DS, some number
        # of CS , and some number of RS 
        ndim = dim[1]
        threshold_b = self.sc.broadcast((self.threshold) * (ndim ** 0.5))
        
        for iter in range(number_of_points // n_points_per_iteration):
            # 8. Load the data points from another file
            bag = data.sample(False, fraction_of_points ,self.seed)
            
            # 9. For the new data points, compare them to each DS using the Mahalanobis Distance and assign them
            # to the nearest DS clusters if the distance is < α 𝑑^1/2 (e.g., 1.96 𝑑^1/2) -> 95% confidence
            
            def get_min(x: Iterable[tuple[float,SummaryCluster]]):
                min_dist , cluster = min(x)
                if min_dist < threshold_b.value:
                    return cluster
                else:
                    return None
                
            def update_cluster(x: tuple[int, Iterable[tuple[int, tuple[SummaryCluster, ndarray]]]]) -> SummaryCluster:
                _ , it = x
                cur_cluster = None
                for s in it:
                    point_index , s2 = s
                    cluster , point = s2
                    if cur_cluster is None:
                        cur_cluster = cluster
                    cluster.add_point(point, point_index)
                return cur_cluster

            comparasion = bag.cartesian(self.DSs) 
            print(f"[DEBUG] Line {inspect.currentframe().f_lineno}")              
            comparasion = self.sc.parallelize(comparasion.collect())
            print(f"[DEBUG] Line {inspect.currentframe().f_lineno}")
            comparasion = comparasion.map(lambda x: (x[0][0], (x[1].mahalanobis_distance(x[0][1]), x[1])) )
            print(f"[DEBUG] Line {inspect.currentframe().f_lineno}")              
            comparasion = self.sc.parallelize(comparasion.collect())
            print(f"[DEBUG] Line {inspect.currentframe().f_lineno}")
            comparasion = comparasion.groupByKey()
            print(f"[DEBUG] Line {inspect.currentframe().f_lineno}")              
            comparasion = self.sc.parallelize(comparasion.collect())
            print(f"[DEBUG] Line {inspect.currentframe().f_lineno}")          
            comparasion = comparasion.map(lambda x: (x[0], get_min(x[1])))     
            print(f"[DEBUG] Line {inspect.currentframe().f_lineno}")              
            comparasion = self.sc.parallelize(comparasion.collect())
            print(f"[DEBUG] Line {inspect.currentframe().f_lineno}")
            comparasion = comparasion.join(bag) 
        
            to_ds = comparasion.filter(lambda x: x[1][0] != None) 
            not_to_ds = comparasion.filter(lambda x: x[1][0] == None)
            
            new_ds = to_ds.groupBy(
                lambda x: x[1][0].points_index[0]
            ).map(
                lambda x: update_cluster(x)
            )
            self.DSs = new_ds
            self.DSs = self.sc.parallelize(self.DSs.collect())
            print(f"[DEBUG] Line {inspect.currentframe().f_lineno}")
            print(f"[DEBUG] Line {inspect.currentframe().f_lineno}")
            print(f"[DEBUG] Line {inspect.currentframe().f_lineno}")
            # 10. For the new data points which are not assigned to any DS cluster, compare them to each of the CS
            # using the Mahalanobis Distance. For the new data points which are not assigned to any DS or CS cluster, add them to your RS
            
            comparasion_cs = not_to_ds.cartesian(self.CSs) \
                            .map(lambda x: 
                                (x[0][0], (x[1].mahalanobis_distance(x[0][1]), x[1])) \
                            ) \
                            .groupByKey() \
                            .map(lambda x: (x[0], get_min(x[1]))) \
                            .join(bag)
            
            to_cs = comparasion_cs.filter(lambda x: x[1][0] != None)
            not_to_cs = comparasion_cs.filter(lambda x: x[1][0] == None)        
            
            new_cs = to_cs.groupBy(
                lambda x: x[1][0].points_index[0]
            ).map(
                lambda x: update_cluster(x)
            )
            
            new_rs = not_to_cs.map(
                lambda x: (x[0], x[1][1])
            )
            
            self.CSs = new_cs
            self.RS = self.RS.union(new_rs)
            self.RS = self.sc.parallelize(self.RS.collect())
            
            print(f"[DEBUG] Line {inspect.currentframe().f_lineno}")
            print(f"[DEBUG] Line {inspect.currentframe().f_lineno}")
            print(f"[DEBUG] Line {inspect.currentframe().f_lineno}")
            # 11. Run the clustering algorithm on the RS with a large number of centroids (e.g., 3 or 5 time of K).
            # Generate CS and their statistics from the clusters with more than one data point and add them to
            # your existing CS list. Use the remaining points as your new RS.
            rs_points = self.RS.map(lambda x: x[1])
            print(f"[DEBUG] Line {inspect.currentframe().f_lineno}")
            print(f"[DEBUG] Rs_points length: {rs_points.countApprox(100)} , clusters: {self.k * self.centroid_mult}")
            print(f"[DEBUG] Line {inspect.currentframe().f_lineno}")
            kmeans_rs = KMeans.train(rs_points, self.k * self.centroid_mult, seed=self.seed, distanceMeasure="euclidean")
            print(f"[DEBUG] Line {inspect.currentframe().f_lineno}")
            labels_rs = self.RS.map(lambda x: (kmeans_rs.predict(x[1]),(x[0],x[1])))
            print(f"[DEBUG] Line {inspect.currentframe().f_lineno}")
            clusters_rs = labels_rs.groupByKey()
            print(f"[DEBUG] Line {inspect.currentframe().f_lineno}")
            outliers_clusters_rs = clusters_rs.filter(lambda x: len(x[1]) < 2)
            inliers_clusters_rs = clusters_rs.filter(lambda x: len(x[1]) >= 2)
            print(f"[DEBUG] Line {inspect.currentframe().f_lineno}")
            new_cs = inliers_clusters_rs.map(lambda x: SummaryCluster(
                len(x[1]),
                np.sum([point[1] for point in x[1]], axis=0),
                np.sum([point[1] ** 2 for point in x[1]], axis=0),
                [point[0] for point in x[1]]
            ))
            print(f"[DEBUG] Line {inspect.currentframe().f_lineno}")
            new_rs = outliers_clusters_rs.flatMap(lambda x: x[1])
            print(f"[DEBUG] Line {inspect.currentframe().f_lineno}")
            self.CSs = self.CSs.union(new_cs)
            self.RS = new_rs
            
            print(f"[DEBUG] Line {inspect.currentframe().f_lineno}")
            print(f"[DEBUG] Line {inspect.currentframe().f_lineno}")
            print(f"[DEBUG] Line {inspect.currentframe().f_lineno}")
            # 12. Merge CS clusters that have a Mahalanobis Distance < α d ^ 1/2.
            cs_list = self.CSs.collect()
            merged = set()
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
        print(f"[DEBUG] Line {inspect.currentframe().f_lineno}")
        print(f"[DEBUG] Line {inspect.currentframe().f_lineno}")
        print(f"[DEBUG] Line {inspect.currentframe().f_lineno}")
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
        print(f"[DEBUG] Line {inspect.currentframe().f_lineno}")
        print(f"[DEBUG] Line {inspect.currentframe().f_lineno}")
        print(f"[DEBUG] Line {inspect.currentframe().f_lineno}")
        return ds_list
    
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
            .setAppName("BFR Clsutering")
    )
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")
    main()
    sc.stop()