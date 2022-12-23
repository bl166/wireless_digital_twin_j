
from asyncio import constants
from unicodedata import name
import tensorflow as tf
import numpy as np
import pandas as pd
import spektral as spk
import networkx as nx
import glob
import os

def get_paths_from_routing(routing):
    if (routing=="paths1"):
        # NSFNet Routing-1
        paths = np.array([
            [0,1,7], [1,7,10,11], [2,5,13], [3,0], [6,4,5], 
            [9,12], [10,9,8], [11,10], [12,9], [13,5,4,3]
        ], dtype=object)

    elif (routing=="paths2"):
        # NSFNet Routing-2
        paths = np.array([
            [9, 10, 7, 1],[3, 4, 5, 13],[12, 5, 2, 0],[10, 9, 8, 3], [13, 10, 7],
            [1, 0, 3, 8],[2, 5, 12, 9],[11, 12, 5, 2],[6, 4, 5, 12],[5, 12, 11],
            [0, 1, 7, 10],[7, 6, 4]
        ], dtype=object)

    elif (routing=="paths3"):
        # GBN Routing-1
        paths = np.array([
            [0, 2, 4, 9, 12, 14, 15],[2, 0, 8, 5, 6],[8, 4, 10, 11, 13],[14, 12, 9, 3, 1], [4, 9, 12],
            [16, 12, 9, 4, 8, 5],[15, 14, 12, 10],[9, 4, 2],[11, 10, 4],[13, 11, 10, 4, 2, 0]
        ], dtype=object)
    return paths

def getTopology(g):
    """
      Input: 
        -- Parsed graph g.
      Output: 
        -- Undirected graph, nx Graph object.
    """
    #Initiliazes empty graph, later adds edges(no additional params for edges)
    g1=nx.Graph()
    for uu, vv, keys, weight in g.edges(data="bandwidth", keys=True):
        g1.add_edge(uu,vv)
    return g1

def getDirectedTopology(g):
    """
      Input: 
        -- Parsed graph g.
      Output: 
        -- Directed graph, nx DiGraph object.
    """
    #Initiliazes empty graph, later adds edges(direction is the only param for an edge)
    g1=nx.DiGraph()
    for uu, vv, keys, weight in g.edges(data="bandwidth", keys=True):
        g1.add_edge(uu,vv)
    return g1


def get_data_gens_sets(config):
    """
    Inputs:
      -- dataPath is a string, a path to the directory which stores all the data.
         Preferrably absolute path.
      -- paths is the option {paths1, paths2, paths3} for extracting the numpy array 
         of all paths that are used for the given topology.
      -- config is result of reading config.ini and parsing it.
    Outputs:
      -- Three Datasets of PlanDataGen objects: for training, validation, and testing.
    """
    ## get output signatures
    def get_output_format(dgen):
        output_types, output_shapes = [], []
        for d in dgen.__getitem__(0):
            out_types, out_shapes = {}, {}
            for k, x in d.items():
                out_types[k]  = x.dtype
                out_shapes[k] = tf.TensorShape([None] * len(x.shape))
            output_types.append(out_types)
            output_shapes.append(out_shapes)
        return tuple(output_types), tuple(output_shapes) # must be tuples!

    # To split into train/validate/test 
    files_kips, files_traf, datagens = {}, {}, {}
    for p in ['train', 'validate', 'test']:
        datagens[p] = []
        
        # To include all Key Performance Indicators files 
        files_kips[p] = sum([glob.glob(os.path.join(dp, p, '*kpis.txt')) for dp in config['Paths']['data']], [])
        files_kips[p].sort() ## You should make sure the file order is correct 
        
        # To include all traffic files 
        files_traf[p] = sum([glob.glob(os.path.join(dp, p, '*traffic.txt')) for dp in config['Paths']['data']], [])
        files_traf[p].sort()
        
        input_argws_gen = {'graphFile': config['Paths']['graph'], 'routing': config['Paths']['routing']}
        datagens[p] = PlanDataGensMulti( filenames=(files_kips[p], files_traf[p]), **input_argws_gen)
        
    #DO *NOT* WRITE THE FOLLOWING IN A LOOP!! 
    #i haven't figured out why but it will cause difference in validation.
    #might have something to do with whatever internal manipulation by tf.. weirdest thing i've ever seen @A@
    input_argws_set = dict(zip(['output_types', 'output_shapes'], get_output_format(datagens['train'])))
    datasets = {}
    datasets['train'] = tf.data.Dataset.from_generator(
        lambda: PlanDataGensMulti(filenames=(files_kips['train'], files_traf['train']), **input_argws_gen), 
        **input_argws_set
    )
    datasets['validate'] = tf.data.Dataset.from_generator(
        lambda: PlanDataGensMulti(filenames=(files_kips['validate'], files_traf['validate']), **input_argws_gen), 
        **input_argws_set
    )
    datasets['test'] = tf.data.Dataset.from_generator(
        lambda: PlanDataGensMulti(filenames=(files_kips['test'], files_traf['test']), **input_argws_gen), 
        **input_argws_set
    )

    return datagens, datasets


class combinedDataGens(tf.keras.utils.Sequence):
    """
     This class takes multiple UnoDataGen objects and returns their common [features, labels].
    """
    def __init__(self, dataGenList):
        """
            Initialize each UnoDataGen object dataset, get total number of data points.
        """
        self.dataGenList = dataGenList
        self.nList = [dG.__len__() for dG in self.dataGenList]
        self.nCumu = np.concatenate([[i]*n for i,n in enumerate(self.nList)])
        self.n = sum(self.nList)
        
    def __getitem__(self,index=None):
        """
            Get item method for the combined dataset.
        """
        return self.dataGenList[self.nCumu[index]].__getitem__(index-sum(self.nCumu[:index]))
        
    def __len__(self):
        return self.n 

    
class PlanDataGen(tf.keras.utils.Sequence):
    def __init__(self, filenames, graphFile, routing):
        """
          Inputs: 
            -- filanames is tuple with filenames for kpi and traffic.
               Each of elems in tuple is a srting of the filename.
            -- graphFile is pathname to read graph from. .gml filename.
            -- routing is numpy array with all paths.
        """
        super(PlanDataGen, self).__init__()

        # Split key performance indicators and traffic.
        kpis, tris = filenames
        kpis = kpis[0] if isinstance(kpis, (list, tuple)) else kpis
        tris = tris[0] if isinstance(tris, (list, tuple)) else tris
        graphFile = graphFile[0] if isinstance(graphFile, (list, tuple)) else graphFile
        routing = routing[0] if isinstance(routing, (list, tuple)) else routing
        
        # Initialiaze how kpi as pandas dataframe looks like.
        kpiframe = pd.read_csv(kpis, header=None)
        # Add columns for indexing. 
        self.kpiframe = kpiframe.reset_index(drop=True).values.astype(np.float32)

        # Number of datapoints 
        self.n = kpiframe.shape[0]
        triframe = pd.read_csv(tris, header=None)
        self.triframe = triframe.reset_index(drop=True).values.astype(np.float32)
    
        # Create a graph(undir and dir version) and store routing 
        multiGraph = nx.read_gml(graphFile, destringizer=int)
        self.graph_topology_undirected = getTopology(multiGraph)
        self.graph_topology_directed = getDirectedTopology(multiGraph)
        self.paths = get_paths_from_routing(routing)

    def on_epoch_end(self):
        # Shuffle data at the end of every epoch
        idx = np.random.permutation(self.n)
        self.kpiframe = self.kpiframe[idx]
        self.triframe = self.triframe[idx]
       
    def __getitem__(self, index=None):
        """ 
          Returns features and corresponding expected labels by reading topological features. 
        """
        # Get topological features as dictionary.
        features=self.get_topological_features()

        # Get values for specific data point 
        a = self.triframe[index]
        b = self.kpiframe[index]
        
        n_links = len(self.graph_topology_directed.edges())
        f_capacities = [1.0]*n_links

        traffic_in = a[0::2]#[float(a[i]) for i in range(0, len(a), 2)]
        traffic_ot = a[1::2]#[float(a[i]) for i in range(1, len(a), 2)]
        f_traffic = np.array([traffic_in, traffic_ot])

        features["path_init"] = tf.Variable(tf.constant(f_traffic), name="path_init")
        features["link_init"] = tf.Variable(tf.constant(f_capacities), name="link_init")

        l_delay = b[2::3]#[float(b[i]) for i in range(2, len(b), 3)]
        l_drops = b[0::3] - b[1::3]

        labels = {
            "delay": tf.Variable(tf.constant(l_delay),name="delay"), #0
            "drops": tf.Variable(tf.constant(l_drops),name="drops"), #1
        }
        return features, labels
    
    
    def get_topological_features(self):
        """ 
          This method extracts all topological features.
        """
        # Gets all links and nodes as lists in the correct format. 
        links, nodes = self.get_links_nodes_arrays()

        # Get Path-Link related features. 
        paths_to_links, links_to_paths, sequences_paths_links = self.get_path_link_features(links)

        # Get Node-Link related features 
        nodes_to_links, sequences_links_nodes, links_to_nodes = self.get_links_nodes(links, nodes)

        # Path-Node
        nodes_to_paths, paths_to_nodes, sequences_nodes_paths = self.get_paths_nodes()

        # Degrees and Lalpacian Matrix.
        degrees = [degree for _,degree in self.graph_topology_directed.out_degree()]

        # Get sparse Laplacian Matrix.
        laplacian = nx.laplacian_matrix(self.graph_topology_undirected) 
        adjacency = nx.adjacency_matrix(self.graph_topology_undirected) 

        # Convert it to dense format.
        laplacian = laplacian.todense()
        adjacency = adjacency.todense()

        topological_features={}

        # Features that are expressed as a single number.
        topological_features["n_paths"] = tf.Variable(tf.constant(len(self.paths)), name="n_paths")
        topological_features["n_links"] = tf.Variable(tf.constant(len(links)), name="n_links")
        topological_features["n_nodes"] = tf.Variable(tf.constant(len(nodes)), name="n_nodes")
        topological_features["n_total"] = tf.Variable(tf.constant(len(paths_to_links)), name="n_total")
        
        # Path-link related features. 
        topological_features["paths_to_links"] = tf.Variable(tf.constant(paths_to_links), name="paths_to_links")
        topological_features["links_to_paths"] = tf.Variable(tf.constant(links_to_paths), name="links_to_paths")
        topological_features["sequences_paths_links"] = tf.Variable(tf.constant(sequences_paths_links), name="sequences_paths_links")

        # Link-node related features 
        topological_features["links_to_nodes"] = tf.Variable(tf.constant(links_to_nodes), name="links_to_nodes")
        topological_features["nodes_to_links"] = tf.Variable(tf.constant(nodes_to_links), name="nodes_to_links")
        topological_features["sequences_links_nodes"] = tf.Variable(tf.constant(sequences_links_nodes), name="sequences_links_nodes")

        # Path-node related features. 
        topological_features["paths_to_nodes"] = tf.Variable(tf.constant(paths_to_nodes), name="paths_to_nodes")
        topological_features["nodes_to_paths"] = tf.Variable(tf.constant(nodes_to_paths), name="nodes_to_paths")
        topological_features["sequences_nodes_paths"] = tf.Variable(tf.constant(sequences_nodes_paths), name="sequences_nodes_paths")
        
        # General topological features, Laplacian matrix and degrees of all nodes.
        topological_features["laplacian_matrix"] = tf.Variable(tf.constant(laplacian, dtype=np.float32), name="laplacian_matrix")
        topological_features["adjacency_matrix"] = tf.Variable(tf.constant(adjacency, dtype=np.float32), name="adjacency_matrix")
        topological_features["node_init"] = tf.Variable(tf.constant(degrees,dtype=np.float32), name="node_init")

        return topological_features
    

    def get_links_nodes_arrays(self):
        """
          Extracts links to nodes features.
        """
        links_from_numpy_to_list = list(np.array(self.graph_topology_directed.edges()))
        # Links stores all links in the graph(edges)
        links = []
        for elem in links_from_numpy_to_list:
            links.append(list(elem))
        
        # Nodes as list in the sorted order.
        nodes = self.graph_topology_directed.nodes()
        nodes = list(nodes)
        nodes.sort()

        return links, nodes

    def get_path_link_features(self, links):
        """
          Extracts paths to links features.
        """
        sequences_paths_links = []
        paths_to_links = []
        links_to_paths = []
        count = 0
        for elem in self.paths:
            for i in range(len(list(elem))):
                if i != len(list(elem))-1:
                    paths_to_links.append(count)
                    sequences_paths_links.append(i)
                    b = [elem[i], elem[i+1]]
                    a = links.index(b)
                    links_to_paths.append(a)
            count += 1
        return paths_to_links, links_to_paths, sequences_paths_links

    def get_links_nodes(self, links, nodes):
        """
          Extracta links to nodes features.
        """
        nodes_to_links = []
        sequences_links_nodes = []
        count_link = 0
        for elem in nodes:
            for i in range(len(links)):
                if links[i][0] == elem:
                    nodes_to_links.append(i)
                    sequences_links_nodes.append(count_link)
            count_link += 1

        links_to_nodes = []
        for i in range(len(links)):
            links_to_nodes.append(links[i][0])
        return nodes_to_links, sequences_links_nodes, links_to_nodes

    def get_paths_nodes(self):
        """
          Extracts paths to nodes features. 
        """
        nodes_to_paths = []
        paths_to_nodes = []
        sequences_nodes_paths = []
        countP = 0
        for elem in self.paths:
            count0 = -1
            for i in range(len(elem)):
                count0 += 1
                if i < (len(elem)-1):
                    nodes_to_paths.append(elem[i])
                    paths_to_nodes.append(countP)
                    sequences_nodes_paths.append(count0)
            countP += 1
        return nodes_to_paths, paths_to_nodes, sequences_nodes_paths

    def __len__(self):
        return self.n 
    
    
class PlanDataGensMulti(PlanDataGen):
    """
     This class takes multiple input files and return a single combined data generator.
    """
    def __init__(self, filenames, graphFile, routing):
        super(PlanDataGensMulti, self).__init__(filenames, graphFile, routing)

        # Split key performance indicators and traffic.
        kpis, tris = filenames
        assert len(tris) == len(kpis) == len(routing)
        
        # Initialiaze how kpi as pandas dataframe looks like.
        self.kpiframeList = [pd.read_csv(k).reset_index(drop=True).values.astype(np.float32) for k in kpis]        
        self.triframeList = [pd.read_csv(t).reset_index(drop=True).values.astype(np.float32) for t in tris]
        self.nList = [kf.shape[0] for kf in self.kpiframeList]
        
        # Number of datapoints 
        self.n = sum(self.nList)
        self.nCumu = np.concatenate([[i]*n for i,n in enumerate(self.nList)])
        
        # Create a graph (undir and dir version)
        assert len(graphFile) in [1, len(self.nList)]
        #convert index to which topology
        self.proj_topo_index = (lambda i: 0) if len(graphFile) == 1 else (lambda i: self.nCumu[i])
        #read all graphs
        self.graphTopoUndList, self.graphTopoDirList = [], []
        for g in graphFile:
            multiGraph = nx.read_gml(g, destringizer=int)
            self.graphTopoUndList.append(getTopology(multiGraph))
            self.graphTopoDirList.append(getDirectedTopology(multiGraph))
            
        # Store routing 
        self.pathsList = [get_paths_from_routing(r) for r in routing]
        
    def on_epoch_end(self):
        # Do not shuffle data at the end of every epoch
        pass

    def __getitem__(self, index=None):
        """ 
          Returns features and corresponding expected labels by reading topological features. 
        """
        # Determine which frame and topo to use
        ti = self.proj_topo_index(index)
        fi = self.nCumu[index]
        index -= sum(self.nCumu[:index])
        
        self.graph_topology_undirected = self.graphTopoUndList[ti]
        self.graph_topology_directed   = self.graphTopoDirList[ti]
        self.triframe = self.triframeList[fi]
        self.kpiframe = self.kpiframeList[fi]
        
        # Get topological features as dictionary.
        features=self.get_topological_features()

        # Get values for specific data point 
        a = self.triframe[index]
        b = self.kpiframe[index]
        
        n_links = len(self.graph_topology_directed.edges())
        f_capacities = [1.0]*n_links

        traffic_in = a[0::2]
        traffic_ot = a[1::2]
        f_traffic = np.array([traffic_in, traffic_ot])

        features["path_init"] = tf.Variable(tf.constant(f_traffic), name="path_init")
        features["link_init"] = tf.Variable(tf.constant(f_capacities), name="link_init")

        l_delay = b[2::3]
        l_drops = b[0::3] - b[1::3]

        labels = {
            "delay": tf.Variable(tf.constant(l_delay),name="delay"), #0
            "drops": tf.Variable(tf.constant(l_drops),name="drops"), #1
        }
        return features, labels
        
    def __len__(self):
        return self.n 

        
    
