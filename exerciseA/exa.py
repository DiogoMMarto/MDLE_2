'''
Spark-submit Python script to implement a "People You Might Know" recommendation algorithm.

Usage:
    spark-submit people_you_might_know.py <input_file> <output_file>

<input_file> should be the path to soc-LiveJournal1Adj.txt.
<output_file> will contain the output: one line per user with top 10 recommendations.

Algorithm:
1. Read the input file and create an adjacency list.
2. Create a flat list of pairs (user, friend) and (friend, user) to find mutual friends.
3. Join the two lists to find mutual friends.
4. Count the number of mutual friends for each pair of users.
5. Filter out pairs that are already friends and self-pairs and count them.
6. Sort the pairs by the number of mutual friends and keep the top 10 for each user.
7. Write the results to the output file.

Author: Diogo Marto
Date: 17-04-2025

Code written with help from automatic code generation tools namely github copilot.
''' 
import os
from pyspark import SparkConf, SparkContext
import argparse

conf = None
sc = None

def main():
    argparser = argparse.ArgumentParser(description="People You Might Know")
    argparser.add_argument("input_file", help="Path to input file")
    argparser.add_argument("output_file", help="Path to output file")
    args = argparser.parse_args()
    
    output_path = os.path.abspath(args.output_file)
    if os.path.exists(output_path):
        print(f"Output path {output_path} already exists. Please remove it before running the script.")
        return
    if not os.path.exists(args.input_file):
        print(f"Input file {args.input_file} does not exist.")
        return
    
    in_file = sc.textFile(args.input_file)
    def parse_line(line):
        user, friends = line.split("\t")
        return user, friends.split(",")
    
    adj_list = in_file.map(parse_line)
    
    adj_list_flat = adj_list.flatMap(lambda x: [(x[0], friend) for friend in x[1]])
    reverse_adj_list = adj_list.flatMap(lambda x: [(friend, x[0]) for friend in x[1]])
    
    mutual_friends_join = adj_list_flat.join(reverse_adj_list)
    
    existing = adj_list_flat.map(lambda x: ((x[0],x[1]),1))
    mutual_friends_pairs = mutual_friends_join.map(lambda x: ((x[1][0],x[1][1]),1))
    
    counted_pairs = (mutual_friends_pairs
                     .filter(lambda x: x[0][0] != x[0][1])
                     .subtractByKey(existing)
                     .reduceByKey(lambda x,y: x+y) # i dont know why but this needs to be run after subtractByKey which is slower
    )
    
    topN = 10
    top_candidates = (counted_pairs
                    .map(lambda x: (x[0][0], (x[0][1], x[1])))
                    .groupByKey()
                    .mapValues(lambda x: sorted(x, reverse=True , key=lambda y: (y[1], y[0]))[:topN])
    )
    def gen_line(x):
        l = ','.join([str(i[0]) for i in x[1]]) if len(x[1]) > 0 else ''
        return f"{x[0]}\t{l}\n"
    
    print("Count:", top_candidates.count())
    res = top_candidates.map(gen_line).collect()
    with open(output_path,"w") as f:
        for line in res:
            f.write(line)
    
if __name__ == "__main__":
    conf = (SparkConf()
            .setAppName("PeopleYouMightKnow")
    )
    sc = SparkContext(conf=conf)
    sc.setLogLevel("WARN")
    main()
    sc.stop()