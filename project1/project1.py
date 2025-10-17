import sys

import networkx as nx
import math
import pandas as pd

def write_gph(dag, idx2names, filename):
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(idx2names[edge[0]], idx2names[edge[1]]))


def bayesian_score(graph, data):
    score = 0
    for node_ind in range(len(graph.nodes())):
        for parent_config in graph[node_ind].parents():
            aijo = 0
            mijo = 0
            for value in graph[node_ind].values():
                aijo += value
                mijo += value
            score += math.log(math.gamma(aijo)/math.gamma(aijo + mijo))
            for value in graph[node_ind].values():
                aijk = 0
                mijk = 0
                score += math.log(math.gamma(aijk + mijk)/math.gamma(aijk))
    return score

def compute(infile, outfile):
    df = pd.read_csv(infile)
    D = nx.DiGraph()
    for column_name in df.columns:
        D.add_node(column_name)
    # k2_algo() || local_search_restarts || local_search_simulated_annealing
    idx2names = {i: name for i, name in enumerate(df.columns)}
    write_gph(D, idx2names, outfile)


def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]
    compute(inputfilename, outputfilename)


if __name__ == '__main__':
    main()
