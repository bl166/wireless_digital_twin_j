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

def distance_based_thresholding(dist_matrix,thresh):
    dist_matrix[dist_matrix==0] = 100 # Exclude self edges
    dist_matrix[dist_matrix<=thresh] = False # 
    dist_matrix[dist_matrix>thresh] = True

    dist_matrix = np.logical_not(dist_matrix)
    return dist_matrix

def compute_pairwise_distance(pos):
    dist_matrix = np.zeros((len(pos),len(pos)))
    for i in range(len(pos)):
        for j in range(len(pos)):
            dist = compute_distance(pos[i],pos[j])
            dist_matrix[i,j] = dist
    return dist_matrix

def pre_process_graph(fileName,thresh,seed):
    multiGraph = nx.read_gml(file, destringizer=int)
    graph_trans = getTopology(multiGraph)
    pos = nx.spring_layout(graph_trans, seed = seed)
    dist_matrix = compute_pairwise_distance(pos)
    dist_matrix_new = distance_based_thresholding(dist_matrix,thresh)
    graph_inter = nx.from_numpy_matrix(dist_matrix_new)

    return graph_trans.edges(), graph_inter.edges()

def get_src_dst_pairs(fileName,n_paths):
    multiGraph = nx.read_gml(file, destringizer=int)
    graph_trans = getDirectedTopology(multiGraph)
    
    # Generate 50% more data and then select <n_paths> 
    # unique pairs
    n_samples = int(1.5*n_paths)
    srcDstPairs = []
    count = 0
    np.random.seed(10)
    while(count<n_samples):
        src = np.random.randint(0,graph_trans.number_of_nodes())
        dst = np.random.randint(0,graph_trans.number_of_nodes())
        
        if src!=dst:
            srcDstPairs.append([src,dst])
            count+=1
    
    srcDstPairs = [list(x) for x in set(tuple(x) for x in srcDstPairs)]
    srcDstPairs = srcDstPairs[:n_paths]

    return srcDstPairs

def convert_tuple_to_set(data):
    return [set(x) for x in data] 

def plot_graphs(graph_trans,graph_inter):
    # title = 'NSFNet'
    # fileName = title+".png"
    # degree_factor = 50
    # degrees = [degree_factor*graph_trans.degree(n) for n in graph_trans.nodes()]
    # nx.draw(graph_trans,pos,node_size=degrees,with_labels=True)
    # nx.draw(graph_inter,pos,node_size=degrees,with_labels=True,style='--')
    # plt.savefig("Undirected_graph.png")
    pass


def extract_topological_features(paths, input_file):
  thresh = 0.45 
  seed = 2
  multiGraph = nx.read_gml(file, destringizer=int)
  graph_topology = getDirectedTopology(multiGraph)
    
  pos = nx.spring_layout(graph_topology, seed = seed)
  links1 = list(np.array(graph_topology.edges()))
  links = []
  for elem in links1:
    links.append(list(elem))
  nodes = graph_topology.nodes()

  seq = []
  paths_ar = []
  count = 0
  links_arr = []
  for elem in paths:
    for i in range(len(list(elem))):
      if i != len(list(elem))-1:
        paths_ar.append(count)
        seq.append(i)
        b = [elem[i], elem[i+1]]
        a = links.index(b)
        links_arr.append(a)
    count += 1

  node_to_link = []
  link_seq = []
  path_seq = []
  nodes = list(nodes)
  nodes.sort()


  count_link=0
  for elem in nodes:
    for i in range(len(links)):
      if links[i][0] == elem:
        node_to_link.append(i)
        link_seq.append(count_link)
    count_link += 1

  count_path = 0
  for elem in nodes:
    for i in range(len(paths)):
      if elem in paths[i]:
        path_seq.append(count_path)
    count_path += 1

  node_to_path = []
  node_to_link = []
  link_seq = []
  path_seq = []
  nodes = list(nodes)
  nodes.sort()


  count_link=0
  for elem in nodes:
    for i in range(len(links)):
      if links[i][0] == elem:
        node_to_link.append(i)
        link_seq.append(count_link)
    count_link += 1
  count_path = 0
  for elem in nodes:
    for i in range(len(paths)):
      if elem in paths[i]:
        node_to_path.append(i)
        path_seq.append(count_path)
    count_path += 1

  links_to_nodes = []
  for i in range(len(links)):
    links_to_nodes.append(links[i][0])

  path_to_nodes = []
  for elem in paths:
    for i in elem:
      path_to_nodes.append(i)
  node_to_paths = []
  paths_to_nodes = []
  seq_nodes_paths = []
  countP = 0
  for elem in paths:
    count0 = -1
    for i in range(len(elem)):
      count0 += 1
      if i < (len(elem)-1):
        node_to_paths.append(elem[i])
        paths_to_nodes.append(countP)
        seq_nodes_paths.append(count0)
    countP += 1
  names = ['n_paths', 'n_links', 'n_total', 'paths_to_links', 'links_to_paths', 'sequences_paths_links', 'link_to_node', "node_to_link", "link_seq",
         'path_to_node', 'node_to_path', 'path_seq']

  arrays = [len(paths), len(links), len(paths_ar), np.array(paths_ar), np.array(links_arr), np.array(seq), np.array(links_to_nodes), np.array(node_to_link),
          np.array(link_seq), np.array(paths_to_nodes), np.array(node_to_paths), np.array(seq_nodes_paths)]



  return arrays 


if __name__ == "__main__":
    paths = np.array([[0,1,7], [1,7,10,11], [2,5,13], [3,0], [6,4,5], [9,12], [10,9,8], [11,10], [12,9], [13,5,4,3]])

    print(extract_topological_features(paths, 'nsfnet.txt'))