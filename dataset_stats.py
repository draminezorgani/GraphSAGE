import json
import networkx as nx
from networkx.readwrite import json_graph
import numpy as np

prefix = "example_data/toy-ppi"
G_data = json.load(open(prefix + "-G.json"))
G = json_graph.node_link_graph(G_data)
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()

id_map = json.load(open(prefix + "-id_map.json"))
class_map = json.load(open(prefix + "-class_map.json"))

print(f"Nodes: {num_nodes}")
print(f"Edges: {num_edges}")
print(f"Features: {np.load(prefix + '-feats.npy').shape}")
print(f"Classes: {len(set(class_map.values())) if not isinstance(list(class_map.values())[0], list) else len(list(class_map.values())[0])}")
