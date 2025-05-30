"""
This script analyzes clusters of tracks and identifies the top genres within each cluster 
using Spark for distributed data processing.

Usage:
    spark-submit exc3.py <input_file_out> <input_file_tracks> <input_file_genres>

Arguments:
    <input_file_out>    Path to the output file containing cluster assignments.
    <input_file_tracks> Path to the CSV file containing track data.
    <input_file_genres> Path to the CSV file containing genre data.

The script performs the following steps:
1. Loads cluster assignments from the output file.
2. Loads track data and extracts genre information for each track.
3. Loads genre metadata to map genre IDs to their names.
4. Joins cluster assignments with track and genre data.
5. Counts the occurrences of each genre within each cluster.
6. Ranks genres within each cluster and identifies the top 10 genres per cluster.
7. Outputs the results, showing the top genres for each cluster.

Output:
    The top genres for each cluster are displayed in the console.

Author: Diogo Marto
Date: 25-04-2025

Code written with help from automatic code generation tools namely GitHub Copilot.
"""
import pandas as pd
from pyspark.sql import SparkSession
import argparse
from pyspark.sql.functions import col, udf, explode, count, desc, row_number, struct, collect_list
from pyspark.sql.types import ArrayType, IntegerType
from pyspark.sql.window import Window
import ast

def main():
    # Initialize Spark session
    spark = SparkSession.builder.appName("Top genres per cluster").getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")

    # Define and parse command line arguments
    parser = argparse.ArgumentParser(description="Cluster analysis")
    parser.add_argument("input_file_out", help="Path to data file")
    parser.add_argument("input_file_tracks", help="Path to tracks data file")
    parser.add_argument("input_file_genres", help="Path to genres data file") # Added argument for genres.csv
    args = parser.parse_args()

    try:
        # Load cluster data
        clusters = []
        with open(args.input_file_out) as f:
            text = f.read()
            clusters = [
                (cn, int(i))
                for cn, line in enumerate(text.split("\n"))
                for i in line.split(",")
            ]
        clusters_df = (
            spark.sparkContext.parallelize(clusters)
            .toDF(["cluster_id", "track_id"])
        )

        # Load track data
        tracks_pd = pd.read_csv(args.input_file_tracks, index_col=0, header=[0, 1])
        tracks_subset = tracks_pd["track"]["genres"]
        tracks_subset_df = tracks_subset.to_frame()
        tracks_subset_df.index.name = 'track_id'
        tracks_subset_df.reset_index(inplace=True)

        # UDF to parse string lists
        def parse_string_list(x):
            try:
                return ast.literal_eval(x)
            except ValueError:
                return []
        parse_string_list_udf = udf(parse_string_list, ArrayType(IntegerType())) # Changed to IntegerType()

        # Convert to Spark DataFrame and apply UDF
        tracks_df = spark.createDataFrame(tracks_subset_df)
        tracks_df = tracks_df.withColumn("genres", parse_string_list_udf(col("genres")))

        # Load genres data
        genres_pd = pd.read_csv(args.input_file_genres)
        genres_df = spark.createDataFrame(genres_pd)
        genres_df = genres_df.withColumn("genre_id", col("genre_id").cast(IntegerType())) # Cast 'genre_id' to IntegerType

        # Join track and cluster data
        tracks_clusters_df = clusters_df.join(
            tracks_df,
            on="track_id",
            how="inner"
        )

        # Explode the genres array
        tracks_clusters_exploded = tracks_clusters_df.withColumn(
            "genre_id",  explode(col("genres"))
        )

        # Join with genres_df to get genre names
        tracks_clusters_exploded = tracks_clusters_exploded.join(genres_df, on="genre_id", how="left")

        # Count genre occurrences
        genre_counts = (
            tracks_clusters_exploded
            .groupBy("cluster_id", "title") # Changed from "genre" to "title"
            .agg(count("*").alias("count"))
        )

        # Rank genres within each cluster
        window = Window.partitionBy("cluster_id").orderBy(desc("count"))
        genre_ranked = (
            genre_counts
            .withColumn("rank", row_number().over(window))
            .filter(col("rank") <= 10)
        )

        # Collect top genres per cluster
        top_genres_per_cluster = (
            genre_ranked
            .groupBy("cluster_id")
            .agg(
                collect_list(
                    struct(
                        col("title"), # Changed from "genre" to "title"
                        col("count")
                    )
                ).alias("top_genres")
            )
        )

        # Show the results
        top_genres_per_cluster.orderBy("cluster_id").show(truncate=False)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Stop the Spark session
        if 'spark' in locals():
            spark.stop()

if __name__ == "__main__":
    main()