"""
Spark-submit Python script to implement a frequent itemset mining algorithm (Apriori-like)
to discover associations between medical conditions based on patient data.

Usage:
    spark-submit conditions_association.py <input_file> <output_file_1> <output_file_2> \
    [-t <topN>] [-s <support>] [-l <lift>] [-i <number_iterations>]

<input_file> should be a CSV file with the header:
    START,STOP,PATIENT,ENCOUNTER,CODE,DESCRIPTION

<output_file_1> will contain the top N most frequent itemsets of conditions
(for k=1, 2, and 3) along with their support counts.

<output_file_2> will contain the discovered association rules with their
lift, confidence, and interest values, filtered by the specified lift threshold.

Algorithm Overview:
1. Reads the input file and extracts patient-condition pairs.
2. Groups conditions by patient and finds unique conditions for each patient.
3. Implements a multi-pass approach to find frequent itemsets of size 1, 2, and 3
   based on the specified support threshold.
4. Generates association rules from the frequent itemsets of size 2 and 3.
5. Calculates lift, confidence, and interest for each rule.
6. Filters the rules based on the specified lift threshold.
7. Writes the top N frequent itemsets and the association rules to the output files.

Author: Diogo Marto 
Date: 20-04-2025

Code written with help from automatic code generation tools namely github copilot.
"""
import argparse
from itertools import combinations
import os
from pyspark import RDD, SparkConf, SparkContext

conf = None
sc = None

def calculate_association_rules(support_k: RDD, support_k_plus_1: RDD, items_count_dict:dict , k, lift_threshold):
    if support_k.isEmpty() or support_k_plus_1.isEmpty():
        return []
    
    total_items_count = sum(items_count_dict.values())
    tic = sc.broadcast(total_items_count)
    icd = sc.broadcast(items_count_dict)
    def calculate_metrics(x):
        antecedent , rest = x
        c , support_antecedent = rest
        consequent, support_antecedent_consequent = c
        support_consequent = icd.value[consequent]
        
        p_consequent = support_consequent / tic.value
        
        confidence = support_antecedent_consequent / support_antecedent
        interest = confidence - p_consequent
        lift = confidence / p_consequent
        
        return f"{antecedent}->{consequent}\t{'REPLACE'}\t{lift}\t{confidence}\t{interest}" , lift
        
    res = (support_k_plus_1
           .flatMap(
               lambda x: [
                   (antecedent , (consequent, x[1]))
                   for antecedent in combinations(x[0], k)
                     for consequent in set(x[0]) - set(antecedent)
               ]
           ).join(support_k)
           .map(
                lambda x: calculate_metrics(x)
              )
    )
    
    max_lift = res.map(lambda x: x[1]).max()
    min_lift = res.map(lambda x: x[1]).min()
    
    res = (res.map(lambda x: (x[0], (x[1] - min_lift)/(max_lift - min_lift)))
            .filter(lambda x: x[1] >= lift_threshold)
            .sortBy(lambda x: x[1], ascending=False)
            .map(lambda x: x[0].replace('REPLACE', str(x[1])))
    )
    
    return res.collect()

def main():
    argparser = argparse.ArgumentParser(description="People You Might Know")
    argparser.add_argument("input_file", help="Path to input file")
    argparser.add_argument("output_file_1", help="Path to output file for top most frequent itemsets of conditions")
    argparser.add_argument("output_file_2", help="Path to output file for association rules")
    argparser.add_argument("-t", "--topN", type=int, default=10, help="Number of top items to consider")
    argparser.add_argument("-s", "--support", type=int, default=1000, help="Support threshold")
    argparser.add_argument("-l", "--lift", type=float, default=0.2, help="Lift threshold")
    args = argparser.parse_args()
    
    in_file = sc.textFile(args.input_file)
    header = "START,STOP,PATIENT,ENCOUNTER,CODE,DESCRIPTION"
    in_file = in_file.filter(lambda x: x != header)
    in_file = in_file.map(lambda x: x.split(","))
    in_file = in_file.map(lambda x: (x[2], int(x[4]))) # (user_id, condition)

    itemsets_base = in_file.groupByKey().mapValues(list).map(lambda x: list(set(x[1]))) # (conditions)
        
    support_1 = (itemsets_base
                    .flatMap(lambda x: [(x[i], 1) for i in range(len(x))])
                    .reduceByKey(lambda x, y: x + y)
                    .filter(lambda x: x[1] >= args.support)
    ) # ( condition, count )
    frequent_items = support_1.map(lambda x: x[0]) # ( condition )
    print(f"Frequent items count: {frequent_items.count()}")
    top_items = support_1.takeOrdered(args.topN, key=lambda x: -x[1])
    with open(args.output_file_1, "w") as f:
        f.write(f"Top {args.topN} frequent items K=1:\n")
        for item in top_items:
            f.write(f"{item[0]}\t{item[1]}\n")
    
    b_frequent_items = sc.broadcast(set(frequent_items.collect())) # ( condition )
    itemsets_2 = (itemsets_base
                    .map(lambda x: [i for i in x if i in b_frequent_items.value])
                    .filter(lambda x: len(x) > 1)
                    .flatMap(lambda x: [tuple(sorted(pair)) for pair in combinations(x, 2)])
    )  # (conditions) -> [( condition1, condition2) , ( condition2, condition3) , ...] -> ( condition1, condition2) , ( condition2, condition3) 
    support_2 = (itemsets_2
                .map(lambda x: (x, 1))
                .reduceByKey(lambda x, y: x + y)
                .filter(lambda x: x[1] >= args.support)
             )
    frequent_items_2 = support_2.map(lambda x: x[0])
    print(f"Frequent items count: {frequent_items_2.count()}")
    top_items_2 = support_2.takeOrdered(args.topN, key=lambda x: -x[1])
    with open(args.output_file_1, "a") as f:
        f.write(f"\nTop 10 frequent items K=2:\n")
        for item in top_items_2:
            f.write(f"{item[0]}\t{item[1]}\n")
        
    b_frequent_items_2 = sc.broadcast(set(frequent_items_2.collect()))
    itemsets_3 = (itemsets_base
                    .map(lambda x: [tuple(sorted(pair)) for pair in combinations(x, 2)]) # [( condition1, condition2)]
                    .map(lambda x: [i for i in x if i in b_frequent_items_2.value])
                    .filter(lambda x: len(x) > 2)
                    .map(lambda x: list(set([j for i in x for j in i]))) # [conditions]
                    .flatMap(lambda x: [tuple(sorted(pair)) for pair in combinations(x, 3)]) # [( condition1, condition2, condition3)]
    )
    support_3 = (itemsets_3
                .map(lambda x: (x, 1))
                .reduceByKey(lambda x, y: x + y)
                .filter(lambda x: x[1] >= args.support)
             )
    print(f"Frequent items count: {support_3.count()}")
    top_items_3 = support_3.takeOrdered(args.topN, key=lambda x: -x[1])
    with open(args.output_file_1, "a") as f:
        f.write(f"\nTop 10 frequent items K=3:\n")
        for item in top_items_3:
            f.write(f"{item[0]}\t{item[1]}\n")
    
    items_count_dict = in_file.map(lambda x: x[1]).countByValue()
    rules_2 = calculate_association_rules(support_1.map(lambda x: ((x[0],),x[1])), support_2, items_count_dict , 1, args.lift)
    rules_3 = calculate_association_rules(support_2, support_3, items_count_dict , 2, args.lift)
    
    with open(args.output_file_2, "w") as f:
        f.write(f"Association rules for K=2:\n")
        for rule in rules_2:
            f.write(f"{rule}\n")
        
        f.write(f"\nAssociation rules for K=3:\n")
        for rule in rules_3:
            f.write(f"{rule}\n")
    
if __name__ == "__main__":
    conf = (SparkConf()
            .setAppName("Conditions")
            # number of executors cores
            # .set("spark.executor.cores", "12")
    )
    sc = SparkContext(conf=conf)
    sc.setLogLevel("ERROR")
    main()
    sc.stop()