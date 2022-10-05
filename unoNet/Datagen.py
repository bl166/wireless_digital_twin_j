from asyncio import constants
from unicodedata import name
import tensorflow as tf
import numpy as np
import pandas as pd
import spektral as spk
import networkx as nx
import glob

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

def get_data_generators(dataPath, paths,config):
      """
        Inputs:
          -- dataPath is a string, a path to the directory which stores all the data.
             Preferrably absolute path.
          -- paths is numpy array of all paths that used for the given topology.
          -- config is result of reading config.ini and parsing it.
        Outputs:
          -- Three UnoDataGen objects: for training, validation, and testing.

      """


      # To include all Key Performance Indicators files 
      kpi_ = '/*kpis.txt'

      # To include all traffic files 
      tfc_ = '/*traffic.txt'

      # To split into train/validate/test 
      traincon = '/train'
      valcon = '/validate'
      testcon = '/test'

      # All files necessary for training
      trainkpis = glob.glob(dataPath+traincon+kpi_)
      trainkpis.sort()
      traintraffics = glob.glob(dataPath+traincon+tfc_)
      traintraffics.sort()

      # All files necessary for validation
      valkpis = glob.glob(dataPath+valcon+kpi_)
      valkpis.sort()
      valtraffics = glob.glob(dataPath+valcon+tfc_)
      valtraffics.sort()

      # All files necessary for testing 
      testkpis = glob.glob(dataPath+testcon+kpi_)
      testkpis.sort()
      testtraffics = glob.glob(dataPath+testcon+tfc_)
      testtraffics.sort()

      traingen = UnoDataGen((trainkpis, traintraffics),config['Paths']['graph'],paths)
      valgen = UnoDataGen((valkpis, valtraffics),config['Paths']['graph'],paths)
      testgen = UnoDataGen((testkpis, testtraffics),config['Paths']['graph'],paths)

      return traingen, valgen, testgen

class combinedDataGens(tf.keras.utils.Sequence):
    """
     This class takes two UnoDataGen objects and returns their common [features, labels].
    """
    def __init__(self, dataGen1,dataGen2):
        """
            Initialize each UnoDataGen object dataset, get total number of data points.
        """
        self.dataGen1 = dataGen1
        self.dataGen2 = dataGen2
        
        self.n1 = dataGen1.__len__()
        self.n2 = dataGen2.__len__()
        self.n = self.n1+self.n2
    
    def __getitem__(self,index=None):
        """
            Get item method for the combined dataset.
        """
        if index<self.n1:
            return self.dataGen1.__getitem__(index)
        else:
            return self.dataGen2.__getitem__(index-self.n1)


    def __len__(self):
        return self.n 

class UnoDataGen(tf.keras.utils.Sequence):
    def __init__(self, filenames,graphFile,routing):
        """
          Inputs: 
            -- filanames is tuple with all filenames for kpi and traffic.
               Each of elems in tuple is a list with all filenames.
            -- graphFile is pathname to read graph from. .gml filename.
            -- routing is numpy array with all paths.
        """

        # Split key performance indicators and traffic.
        kpis, tris = filenames
        
        # Initialiaze how kpi as pandas dataframe looks like.
        kpiframe = pd.read_csv(kpis[0], header=None)
        # Add columns for indexing. 
        self.kpiframe = kpiframe.reset_index(drop=True)

        # Number of datapoints 
        self.n = kpiframe.shape[0]
        triframe = pd.read_csv(tris[0], header=None)
        self.triframe = triframe.reset_index(drop=True)
    
        # Create a graph(undir and dir version) and store routing 
        multiGraph = nx.read_gml(graphFile, destringizer=int)
        self.graph_topology_undirected = getTopology(multiGraph)
        self.graph_topology_directed = getDirectedTopology(multiGraph)
        self.paths = routing

    def on_epoch_end(self):
        pass
        
    def __getitem__(self, index=None):
        """ 
          Returns features and corresponding expected labels by reading topological features. 
        """

        # Get topological features as dictionary.
        features=self.get_topological_features()

        # Get values for specific data point 
        a = self.triframe.loc[index]
        b = self.kpiframe.loc[index]
        
        n_links = len(self.graph_topology_directed.edges())
        f_capacities = [1.0]*n_links

        traffic_in = [float(a[i]) for i in range(0, len(a), 2)]
        traffic_ot = [float(a[i]) for i in range(1, len(a), 2)]
        f_traffic = [traffic_in, traffic_ot]

        features["traffic"] =   tf.Variable(tf.constant(f_traffic))
        features["capacities"] =  tf.Variable(tf.constant(f_capacities))

        l_delay = [float(b[i]) for i in range(2, len(b), 3)]

        labels = [
            tf.Variable(tf.constant(l_delay),name="delay"), #0
        ]
        return [features, labels]
    
    def get_topological_features(self):
        """ 
          This method extracts all topological features.
          
        """

        links1 = list(np.array(self.graph_topology_directed.edges()))
        links = []
        for elem in links1:
            links.append(list(elem))
        
        nodes = self.graph_topology_directed.nodes()
        nodes = list(nodes)
        nodes.sort()

        # Path-Link
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
        for elem in self.paths:
            for i in elem:
                path_to_nodes.append(i)

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


        # Degrees and laplacian
        degrees = [degree for _,degree in self.graph_topology_directed.out_degree()]
        laplacian = nx.laplacian_matrix(self.graph_topology_undirected) # Returns sparse laplacian
        laplacian = laplacian.todense()

        topological_features={}

        topological_features['n_paths'] = tf.Variable(tf.constant(len(self.paths)))
        topological_features['n_links'] = tf.Variable(tf.constant(len(links)))
        topological_features['n_nodes'] = tf.Variable(tf.constant(len(nodes)))
        topological_features['n_total'] = tf.Variable(tf.constant(len(paths_to_links)))
        
        topological_features['paths_to_links'] = tf.Variable(tf.constant(paths_to_links))
        topological_features['links_to_paths'] = tf.Variable(tf.constant(links_to_paths))
        topological_features['sequences_paths_links'] = tf.Variable(tf.constant(sequences_paths_links))

        topological_features['links_to_nodes'] = tf.Variable(tf.constant(links_to_nodes))
        topological_features['nodes_to_links'] = tf.Variable(tf.constant(nodes_to_links))
        topological_features['sequences_links_nodes'] = tf.Variable(tf.constant(sequences_links_nodes))

        topological_features['paths_to_nodes'] = tf.Variable(tf.constant(paths_to_nodes))
        topological_features['nodes_to_paths']=tf.Variable(tf.constant(nodes_to_paths))
        topological_features['sequences_nodes_paths'] = tf.Variable(tf.constant(sequences_nodes_paths))
        
        topological_features["laplacian_matrix"] = tf.Variable(tf.constant(laplacian, dtype=np.float32))
        topological_features["degrees"]    =  tf.Variable(tf.constant(degrees,dtype=np.float32))

        return topological_features

    def __len__(self):
        return self.n 