from __future__ import print_function

import numpy as np
import random
import json
import sys
import os

import networkx as nx
from networkx.readwrite import json_graph

# Removed version check for NetworkX

WALK_LEN=5
N_WALKS=50

def load_data(prefix, normalize=True, load_walks=False):
    G_data = json.load(open(prefix + "-G.json"))
    G = json_graph.node_link_graph(G_data)
    
    # In NetworkX 2.0+, G.nodes is a NodeView. 
    # We'll use strings for all map keys to be safe with JSON-loaded data.
    
    if os.path.exists(prefix + "-feats.npy"):
        feats = np.load(prefix + "-feats.npy")
    else:
        print("No features present.. Only identity features will be used.")
        feats = None
        
    id_map = json.load(open(prefix + "-id_map.json"))
    # Ensure id_map keys are strings for consistent lookup
    id_map = {str(k): int(v) for k, v in id_map.items()}
    
    walks = []
    class_map = json.load(open(prefix + "-class_map.json"))
    # Ensure class_map keys are strings
    if isinstance(list(class_map.values())[0], list):
        lab_conversion = lambda n : n
    else:
        lab_conversion = lambda n : int(n)
    class_map = {str(k): lab_conversion(v) for k, v in class_map.items()}

    ## Remove all nodes that do not have val/test annotations
    broken_count = 0
    for node in list(G.nodes()):
        if 'val' not in G.nodes[node] or 'test' not in G.nodes[node]:
            G.remove_node(node)
            broken_count += 1
    print("Removed {:d} nodes that lacked proper annotations".format(broken_count))

    ## Make sure the graph has edge train_removed annotations
    print("Loaded data.. now preprocessing..")
    for edge in list(G.edges()):
        if (G.nodes[edge[0]]['val'] or G.nodes[edge[1]]['val'] or
            G.nodes[edge[0]]['test'] or G.nodes[edge[1]]['test']):
            G[edge[0]][edge[1]]['train_removed'] = True
        else:
            G[edge[0]][edge[1]]['train_removed'] = False

    if normalize and not feats is None:
        from sklearn.preprocessing import StandardScaler
        train_ids = np.array([id_map[str(n)] for n in G.nodes() if not G.nodes[n]['val'] and not G.nodes[n]['test']])
        train_feats = feats[train_ids]
        scaler = StandardScaler()
        scaler.fit(train_feats)
        feats = scaler.transform(feats)
    
    if load_walks:
        with open(prefix + "-walks.txt") as fp:
            for line in fp:
                # Assuming walks file also uses node IDs that match G.nodes
                walks.append([str(n) for n in line.split()])

    return G, feats, id_map, walks, class_map

def run_random_walks(G, nodes, num_walks=N_WALKS):
    pairs = []
    for count, node in enumerate(nodes):
        if G.degree(node) == 0:
            continue
        for i in range(num_walks):
            curr_node = node
            for j in range(WALK_LEN):
                # G.neighbors(curr_node) is an iterator in NX 2.0+
                next_node = random.choice(list(G.neighbors(curr_node)))
                # self co-occurrences are useless
                if curr_node != node:
                    pairs.append((node,curr_node))
                curr_node = next_node
        if count % 1000 == 0:
            print("Done walks for", count, "nodes")
    return pairs

if __name__ == "__main__":
    """ Run random walks """
    graph_file = sys.argv[1]
    out_file = sys.argv[2]
    G_data = json.load(open(graph_file))
    G = json_graph.node_link_graph(G_data)
    nodes = [n for n in list(G.nodes()) if not G.nodes[n]["val"] and not G.nodes[n]["test"]]
    G = G.subgraph(nodes)
    pairs = run_random_walks(G, nodes)
    with open(out_file, "w") as fp:
        fp.write("\n".join([str(p[0]) + "\t" + str(p[1]) for p in pairs]))
