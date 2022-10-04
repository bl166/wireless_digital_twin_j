import networkx as nx
import os
import pdb
import numpy as np
import networkx.algorithms.isomorphism as iso
import matplotlib.pyplot as plt

def compute_distance(x,y):
    return np.sqrt(np.sum((x-y)**2))

def getTopology(g):
    g1=nx.Graph()
    for uu, vv, keys, weight in g.edges(data="bandwidth", keys=True):
        g1.add_edge(uu,vv)
    return g1

def getDirectedTopology(g):
    g1=nx.DiGraph()
    for uu, vv, keys, weight in g.edges(data="bandwidth", keys=True):
        g1.add_edge(uu,vv)
    return g1

def extract_topological_features(paths, input_file):
    multiGraph = nx.read_gml(input_file, destringizer=int)
    graph_topology = getDirectedTopology(multiGraph)
    
    links1 = list(np.array(graph_topology.edges()))
    links = []
    for elem in links1:
      links.append(list(elem))
    
    nodes = graph_topology.nodes()
    nodes = list(nodes)
    nodes.sort()

    # Path-Link
    sequences_paths_links = []
    paths_to_links = []
    links_to_paths = []
    count = 0
    for elem in paths:
      for i in range(len(list(elem))):
        if i != len(list(elem))-1:
          paths_to_links.append(count)
          sequences_paths_links.append(i)
          b = [elem[i], elem[i+1]]
          a = links.index(b)
          links_to_paths.append(a)
      count += 1

    # Node-Link
    nodes_to_links = []
    sequences_links_nodes = []
    count_link=0
    for elem in nodes:
      for i in range(len(links)):
        if links[i][0] == elem:
          nodes_to_links.append(i)
          sequences_links_nodes.append(count_link)
      count_link += 1

    links_to_nodes = []
    for i in range(len(links)):
      links_to_nodes.append(links[i][0])

    # Path-Node
    path_to_nodes = []
    for elem in paths:
      for i in elem:
        path_to_nodes.append(i)

    nodes_to_paths = []
    paths_to_nodes = []
    sequences_nodes_paths = []
    countP = 0
    for elem in paths:
      count0 = -1
      for i in range(len(elem)):
        count0 += 1
        if i < (len(elem)-1):
          nodes_to_paths.append(elem[i])
          paths_to_nodes.append(countP)
          sequences_nodes_paths.append(count0)
      countP += 1

    topological_features={}

    topological_features['n_paths'] = len(paths)
    topological_features['n_links'] = len(links)
    topological_features['n_nodes'] = len(nodes)
    topological_features['n_total'] = len(paths_to_links)
    
    topological_features['paths_to_links'] = paths_to_links
    topological_features['links_to_paths'] = links_to_paths
    topological_features['sequences_paths_links'] = sequences_paths_links

    topological_features['links_to_nodes'] = links_to_nodes
    topological_features['nodes_to_links'] = nodes_to_links
    topological_features['sequences_links_nodes'] = sequences_links_nodes

    topological_features['paths_to_nodes'] = paths_to_nodes
    topological_features['nodes_to_paths']=nodes_to_paths
    topological_features['sequences_nodes_paths'] = sequences_nodes_paths

    return topological_features 


if __name__ == "__main__":
    paths = np.array([[0,1,7], [1,7,10,11], [2,5,13], [3,0], [6,4,5], [9,12], [10,9,8], [11,10], [12,9], [13,5,4,3]])

    topological_features = extract_topological_features(paths, 'nsfnet.txt')
    pdb.set_trace()

