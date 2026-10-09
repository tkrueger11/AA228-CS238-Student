import sys
import numpy as np
import networkx as nx
from typing import Tuple
import scipy.special as sc
import time

# python3 project1.py tiny.csv outfile.gph
def write_gph(dag, idx2names, filename):
    with open(filename, 'w') as f:
        for edge in dag.edges():
            f.write("{}, {}\n".format(idx2names[edge[0]], idx2names[edge[1]]))


def compute(infile: str, outfile: str):
    D = extract_raw_data(infile)
    R = get_value_counts(D)
    DG = rudimentary_get_bayesian_network()
    M, Q = get_graph_stats(DG, D, R)
    bayesian_score = compute_bayesian_score(M, Q, R)
    print(bayesian_score)
    pass

def extract_raw_data(path: str) -> np.ndarray:
    return np.loadtxt(path, delimiter=',', skiprows=1, dtype=np.uint8)

def rudimentary_get_bayesian_network() -> nx.DiGraph:
    DG = nx.DiGraph()
    DG.add_nodes_from([(0), (1), (2), (3)])
    DG.add_edges_from([
        (0, 1),
        (0, 2),
        (3, 2),
    ])
    return DG

# def k2() -> nx.DiGraph:


def compute_bayesian_score(M: np.ndarray, Q: np.ndarray, R: np.ndarray) -> float:
    n = len(Q)
    score = 0
    for i in range(n):
        q_i = Q[i]
        r_i = R[i]
        a_ij0 = q_i * r_i
        sum_m_ij0 = np.sum(M[i])
        m_i_transformed = sc.loggamma(1 + M[i])
        score += sc.loggamma(a_ij0) - sc.loggamma(a_ij0 + sum_m_ij0) + np.sum(m_i_transformed)

    return score

def get_value_counts(D: np.ndarray) -> np.ndarray:
    nodes = D.shape[1]
    r_i = np.ones(nodes)
    for node in range(nodes):
        r_i[node] = np.max(D, axis=0)

    return r_i

def get_graph_stats(DG: nx.DiGraph, 
                    D: np.ndarray, 
                    value_counts: np.ndarray) -> Tuple[list, dict]:
    ordered_nodes = list(nx.topological_sort(DG))
    m_ijk = []
    q_i = {}

    for node in ordered_nodes:
        parents = list(DG.predecessors(node))
        relevant_nodes = [node] + parents
        _, counts = np.unique(D[:, relevant_nodes], axis=0, return_counts=True)

        q_i[node] = np.prod(value_counts[parents])
        m_ijk.append(np.array(counts))

    return (m_ijk, q_i)

def main():
    if len(sys.argv) != 3:
        raise Exception("usage: python project1.py <infile>.csv <outfile>.gph")

    inputfilename = sys.argv[1]
    outputfilename = sys.argv[2]

    compute(inputfilename, outputfilename)


if __name__ == '__main__':
    main()



















# Scatchpad


def compute1(infile: str, outfile: str):
    unique_samples, sample_counts = extract_raw_data1(infile)
    DG = rudimentary_get_bayesian_network()
    print(get_graph_stats1(DG, unique_samples,sample_counts))
    pass

def compute2(infile: str, outfile: str):
    unique_samples, sample_counts = extract_raw_data1(infile)
    DG = rudimentary_get_bayesian_network()
    print(get_graph_stats2(DG, unique_samples,sample_counts))
    pass

def extract_raw_data1(path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, delimiter=',', skiprows=1, dtype=np.uint8)
    unique_samples, sample_counts = np.unique(data, axis=0, return_counts=True)
    return (unique_samples, sample_counts)


def get_graph_stats2(DG: nx.DiGraph, 
                    unique_samples: np.ndarray, 
                    sample_counts: np.ndarray):
    stats = []

    for node in DG.nodes:
        parents = list(DG.predecessors(node))
        relevant_nodes = [node] + parents
        _, inverse_idx = np.unique(unique_samples[:, relevant_nodes], axis=0, return_inverse=True)
        instance_counts = np.bincount(inverse_idx, weights=sample_counts)

        stats.append(np.array(instance_counts))

    return np.array(stats, dtype=object)

def get_graph_stats1(DG: nx.DiGraph, 
                    unique_samples: np.ndarray, 
                    sample_counts: np.ndarray):
    ordered_nodes = list(nx.topological_sort(DG))
    stats = list(range(len(ordered_nodes)))

    for node in ordered_nodes:
        parents = list(DG.predecessors(node))
        node_col = unique_samples[:, node]
        unique_values, inverse_idx = np.unique(node_col, return_inverse=True)
        unique_value_dict = dict((value, index) for index, value in enumerate(unique_values))

        DG.nodes[node]["r"] = unique_values.shape[0]
        DG.nodes[node]["values"] = unique_value_dict

        # We need to calculate j and factor in parental instance
        if parents:
            q = 1
            for parent in parents:
                q *= DG.nodes[parent].get("r", 1)

            i_matrix = np.zeros((q, DG.nodes[node]["r"]))
            for index, sample in enumerate(unique_samples):
                k = DG.nodes[node]["values"][sample[node]]
                j = 0
                for parent in parents:
                    j += DG.nodes[parent]["values"][sample[parent]]

                i_matrix[j][k] += sample_counts[index]

            stats[node] = i_matrix


        # We know j = 1, so we create an array for the rs
        else:
            value_counts = np.bincount(inverse_idx, weights=sample_counts)
            # In order of appearance
            stats[node] = value_counts[:,None]

    return np.array(stats, dtype=object)


def run(infile: str, outfile: str):
    D = extract_raw_data(infile)
    unique_samples, sample_counts = extract_raw_data1(infile)
    DG = rudimentary_get_bayesian_network()

    s = time.time()
    for _ in range(100000):
        get_graph_stats(DG, D)
    print(f'get_graph_stats took {time.time() - s}s')

    s = time.time()
    for _ in range(100000):
        get_graph_stats1(DG, unique_samples, sample_counts)
    print(f'get_graph_stats1 took {time.time() - s}s')

    s = time.time()
    for _ in range(100000):
        get_graph_stats2(DG, unique_samples, sample_counts)
    print(f'get_graph_stats2 took {time.time() - s}s')