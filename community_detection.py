import igraph
import numpy as np
from itertools import combinations
import copy

def find_communities(edge_index_path, edge_masks_path, detection_alg='louvain'):
    """
    Creates communities of nodes in a graph based on edge masks and algorithm
    :param edge_index_path: String which contains path to edge_index file
    :param edge_masks_path: String which contains path to edge_mask file
    :param detection_alg: String which decides on algorithm to be used
    return
    Average edge masks per community and communities
    """

    assert detection_alg in ['louvain', 'opt_modularity']

    edge_index = np.loadtxt(edge_index_path, dtype=int)
    edge_masks = abs(np.loadtxt(edge_masks_path, dtype=float))
    s = list(copy.copy(edge_index[0]))
    t = list(copy.copy(edge_index[1]))

    s.extend(t)
    nodes = list(set(s))

    edges = np.array(edge_index)

    edges = [(row[0], row[1]) for row in edges.T]

    g = igraph.Graph()
    # use max(nodes)+1 for modified datasets for shorter runtime
    # all other cases use len(nodes)
    g.add_vertices(max(nodes)+1)
    #g.add_vertices(len(nodes))
    g.add_edges(edges)
    g.es['weight'] = edge_masks

    if detection_alg == 'louvain':
        partition = g.community_multilevel(weights=g.es['weight'])
    elif detection_alg == 'opt_modularity':
        partition = g.community_optimal_modularity(weights=g.es['weight'])
    combs = []
    for i in range(len(partition)):
        cs = []
        for c in combinations(partition[i], r=2):
            cs.append(c)    
        combs.append(sorted(list(set(edges) & set(cs))))

    avg_edge_masks = []
    for com in combs:
        if(len(com) == 0):
            continue
        avg_mask = 0
        for edge in com:
            avg_mask += g.es[g.get_eid(edge[0], edge[1])]['weight']
        avg_edge_masks.append(avg_mask/len(com))
        
    return avg_edge_masks, partition
