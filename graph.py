import networkx as nx
import numpy as np
from random import randint
import matplotlib.pyplot as plt

RESERVED = '*'
INF = 10000

def transform(G : nx.Graph(), dv, de, hv, he, beta) :
    """
    Takes a graph G with labels on nodes and edges.
    Labels are respectively denoted by 'v' and 'e'

    Args :
        - G (nx.Graph) : input graph
        - dv : distance function on node labels
        - de : distance function on edge labels
        - hv : histogram on vertex labels
        - he : histogram on edge labels
        - beta : beta coefficient for the histogram : h(u) = beta*hv(u) ; h(e) = (1-beta)*he(e)

    Output :
        - G2 (nx.Graph) : transformed graph
        - d : new distance (function)
        - h : histogram on the new nodes of the transformed graph

        CAUTION : The character '*' cannot be a label
    """

    G2 = nx.Graph()

    def d(a, b):
        if a[0] is RESERVED and b[0] is RESERVED:
            return de(a[1], b[1])
        if a[1] is RESERVED and b[1] is RESERVED:
            return dv(a[0], b[0])
        return INF
 

    nodes = []
    edges = []
    node_index = {}
    edge_index = {}
    for i, u in enumerate(G.nodes):
        nodes.append((u, {'x': (G.nodes[u]['x'], RESERVED)}))
        node_index[u] = i
    for i, (u, v) in enumerate(G.edges):
        nodes.append(((u, v), {'x': (RESERVED, G[u][v]['edge_attr'])}))
        edges.append(((u, v), u, {'edge_attr':0}))
        edges.append((v, (u, v), {'edge_attr':0}))
        edge_index[(u, v)] = i
    
    G2.add_nodes_from(nodes)
    G2.add_edges_from(edges)

    h = []
    for u in G2.nodes:
        if G2.nodes[u]['x'][0] is RESERVED:
            h.append((1-beta)*he[edge_index[u]])
        else:
            h.append(beta*hv[node_index[u]])
    
    G2.graph = G2.graph

    return G2, d, np.array(h)

def all_to_all(G : nx.Graph()):
    """
    Compute all_to_all distance matrix

    Args :
        - G (nx.Graph())
    
    Output :
        - C (np.array) : distance array
    """
    d = dict(nx.shortest_path_length(G))
    n = G.number_of_nodes()
    C = np.zeros((n, n), dtype=np.float64)

    for i, u in enumerate(G.nodes):
        for j, v in enumerate(G.nodes):
            try:
                C[i, j] = d[u][v]
            except KeyError:
                C[i, j] = INF

    return C

def generate(n, nlv):
    """
    Generate a random connected graph.

    Parameters :
    - n : number of nodes used to generate a random graph before taking the largest connected component. The graph is a random graph with the law G(n, min(1, 3/n)). This law is arbitrary.
    - nlv : number of labels on nodes
    - nle : number of labels on edges
    """
    G = nx.fast_gnp_random_graph(n, min(1, 3/n))
    largest_cc = max(nx.connected_components(G), key=len)
    G = G.subgraph(largest_cc).copy()

    relabel = {}
    for i, u in enumerate(G.nodes):
        relabel[u] = i
    G = nx.relabel_nodes(G, relabel, copy=False)

    for u in G.nodes:
        G.nodes[u]['x'] = randint(0, nlv-1)
    for u, v in G.edges:
        G[u][v]['edge_attr'] = 0
    return G

def plot(G : nx.Graph(), map_v, title, n, m, j):
    plt.subplot(n, m, j)
    plt.title(title)
    color_map = []
    for u in G.nodes:
        color_map.append(map_v[G.nodes[u]['x']])
    nx.draw_kamada_kawai(G, node_color=color_map, with_labels=True, width=2, node_size=500)