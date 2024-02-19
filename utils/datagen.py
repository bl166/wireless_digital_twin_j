import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from asyncio import constants
from unicodedata import name
import tensorflow as tf
import numpy as np
import pandas as pd
import spektral as spk
import networkx as nx
import glob
import os
import ast

def get_paths_from_routing(routing):
    if (routing=="paths1"):
        # NSFNet Routing-1
        paths = np.array([
            [0, 1, 7], 
            [1, 7, 10, 11], 
            [2, 5, 13], 
            [3, 0], 
            [6, 4, 5], 
            [9, 12], 
            [10, 11, 8],#[10, 9, 8], 
            [11, 10], 
            [12, 9], 
            [13, 5, 4, 3]
        ], dtype=object)
    elif (routing=="paths2"):
        # NSFNet Routing-2
        paths = np.array([
            [9, 10, 7, 1],
            [3, 4, 5, 13],
            [12, 5, 2, 0],
            [10, 9, 8, 3], 
            [13, 10, 7],
            [1, 0, 3, 8],
            [2, 5, 12, 9],
            [11, 12, 5, 2],
            [6, 4, 5, 12],
            [5, 12, 11],
            [0, 1, 7, 10],
            [7, 6, 4]
        ], dtype=object)
    elif (routing=="paths3"):
        # GBN Routing-1
        paths = np.array([
            [0, 2, 4, 9, 12, 14, 15],
            [2, 0, 8, 5, 6],
            [8, 4, 10, 11, 13],
            [14, 12, 9, 3, 1],
            [4, 9, 12],
            [16, 12, 9, 4, 8, 5],
            [15, 14, 12, 10],
            [9, 4, 2],
            [11, 10, 4],
            [13, 11, 10, 4, 2, 0]
        ], dtype=object)
    else:
        raise
    return paths

def get_paths_from_routing_dynamic(filename):
    paths = []
    with open(filename, 'r') as f:
        for i, p in enumerate(f.readlines()):
            try:
                paths.append(ast.literal_eval(p)) 
            except:
                print(i, filename, p)
                paths.append(ast.literal_eval(p)) 
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

def getTopologyWeighted(g):
    """
      Input: 
        -- Parsed graph g.
      Output: 
        -- Undirected graph, nx Graph object.
    """
    #Initiliazes empty graph, later adds edges(no additional params for edges)
    g1=nx.Graph()
    for uu, vv, weight in g.edges(data="weight"):
        g1.add_weighted_edges_from([(uu, vv, weight)])
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

def getDirectedTopologyWeighted(g):
    """
      Input: 
        -- Parsed graph g.
      Output: 
        -- Undirected graph, nx Graph object.
    """
    #Initiliazes empty graph, later adds edges(no additional params for edges)
    g1=nx.DiGraph()
    for uu, vv, weight in g.edges(data="weight"):
        g1.add_weighted_edges_from([(uu, vv, weight)])
    return g1


def dataset_batchify(ds, bs, padding=True, batching=True):
    if not batching:
        return ds
    if padding:
        return ds.padded_batch(
            bs, padding_values=({
                k:tf.constant(-1,dtype=v.dtype) for k,v in ds.element_spec[0].items()
            },None)
        )
    else:
        return ds.batch(bs)


def get_data_gens_sets(config, fold=0):
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
    files_kips, files_traf, files_path, files_grap = {}, {}, {}, {}
    input_argws_gen, datagens = {}, {}
    for p in ['train', 'validate', 'test']:     
        if p != 'test':
            pf = p if fold==0 else f'cv{fold}_{p}'
        else:
            pf = p 
        
        # To include all Key Performance Indicators files 
        files_kips[p] = sum([glob.glob(os.path.join(dp, pf, '*kpis.txt')) for dp in config['Paths']['data']], [])
        files_kips[p].sort() ## You should make sure the file order is correct 
        
        # To include all traffic files 
        files_traf[p] = sum([glob.glob(os.path.join(dp, pf, '*traffic.txt')) for dp in config['Paths']['data']], [])
        files_traf[p].sort()
        
        # To include all paths files
        files_path[p] = sum([glob.glob(os.path.join(dp, pf, '*paths.txt')) for dp in config['Paths']['data']], [])
        files_path[p].sort()  
        
        # To include all graph files
        files_grap[p] = config['Paths']['graph']
        if not files_grap[p][0].endswith('gml'):
            files_grap[p] = sum([glob.glob(os.path.join(dp, pf, '*gfiles.txt')) for dp in config['Paths']['data']], [])
            files_grap[p].sort()
        
        input_argws_gen[p] = {
            'graphFile': config['Paths']['graph'] if not files_grap[p] else files_grap[p],
            'routing'  : config['Paths']['routing'] if not files_path[p] else files_path[p],
        }                
        # whether to add noise
        if 'add_noise' in config['LearningParams']:
            input_argws_gen[p]['add_noise'] = config['LearningParams']['add_noise'] if p == 'train' else False        
        
        datagens[p] = PlanDataGensMulti( 
            filenames = (files_kips[p], files_traf[p]), 
            **input_argws_gen[p],
            label_scale = False,
        )
        
        # whether to scale labels
        scale_dict = None
        if 'label_scale' in config['LearningParams'] and config['LearningParams']['label_scale']:
            scale_dict = datagens[p].scale_labels(None if p == 'train' else datagens['train'].scale_dict)
                
    #DO *NOT* WRITE THE FOLLOWING IN A LOOP!! 
    #i haven't figured out why but it will cause difference in validation ... weird @A@
    input_argws_set = dict(zip(['output_types', 'output_shapes'], get_output_format(datagens['train'])))
    #print('input_argws_set:', input_argws_set)

    datasets = {}
    datasets['train'] = tf.data.Dataset.from_generator(
        lambda: PlanDataGensMulti(
            filenames=(files_kips['train'], files_traf['train']), 
            label_scale = datagens['train'].scale_dict, 
            **input_argws_gen['train'],
        ), 
        **input_argws_set
    )
    datasets['validate'] = tf.data.Dataset.from_generator(
        lambda: PlanDataGensMulti(
            filenames=(files_kips['validate'], files_traf['validate']), 
            label_scale = datagens['train'].scale_dict, 
            **input_argws_gen['validate']
        ), 
        **input_argws_set
    )
    datasets['test'] = tf.data.Dataset.from_generator(
        lambda: PlanDataGensMulti(
            filenames=(files_kips['test'], files_traf['test']), 
            label_scale = datagens['train'].scale_dict, 
            **input_argws_gen['test']
        ), 
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
    def __init__(self, filenames, graphFile, routing, label_scale=False, add_noise=False):
        """
          Inputs: 
            -- filanames is tuple with filenames for kpi and traffic.
               Each of elems in tuple is a srting of the filename.
            -- graphFile is pathname to read graph from. .gml filename.
            -- routing is numpy array with all paths (or routing tables filename).
            
          Labels: delay, jitter, throughput, drops, drops-rate, drops-bi
        """
        super().__init__()
        self.l2c = {'packet_tx': 0, 'packet_rx': 1, 'delay': 2, 'jitter': 3, 'throughput': 4}
        self.scale_dict = {'delay': 1, 'jitter': 1, 'throughput': 1, 'drops': 1}
        self.n_kpis = 5
        self.add_noise = add_noise

        # Split key performance indicators and traffic.
        kpis, tris = filenames
        kpis = kpis[0] if isinstance(kpis, (list, tuple)) else kpis
        tris = tris[0] if isinstance(tris, (list, tuple)) else tris
        graphFile = graphFile[0] if isinstance(graphFile, (list, tuple)) else graphFile
        routing = routing[0] if isinstance(routing, (list, tuple)) else routing
        #print("PlanDataGen:", routing)
        
        # Initialiaze how kpi as pandas dataframe looks like.
        kpiframe = pd.read_csv(kpis, header=None)
        # Add columns for indexing. 
        self.kpiframe = kpiframe.reset_index(drop=True).values.astype(np.float32)

        # Number of datapoints 
        self.n = kpiframe.shape[0]
        triframe = pd.read_csv(tris, header=None)
        self.triframe = triframe.reset_index(drop=True).values.astype(np.float32)
    
        # Create a graph(undir and dir version) and store routing 
        self.get_graphs(graphFile)
        
        # Paths
        self.get_paths(routing)
        
        if label_scale is not False: # None or dict
            self.scale_labels(label_scale)
        
    def scale_labels(self, scale_dict=None):
        if scale_dict is None:
            # training data, calculate scales
            scale_dict = self.scale_dict
            for lab in scale_dict.keys():
                if lab != 'drops':
                    kpi_curr = self.kpiframe[:, self.l2c[lab]::self.n_kpis].reshape(-1)
                else:
                    tx = self.kpiframe[:, self.l2c['packet_tx']::self.n_kpis].reshape(-1)
                    rx = self.kpiframe[:, self.l2c['packet_rx']::self.n_kpis].reshape(-1)
                    kpi_curr = tx - rx
                scale_dict[lab] = np.quantile(kpi_curr, 0.75) - np.quantile(kpi_curr, 0.25)
        # update to training scales
        self.scale_dict = scale_dict
        return scale_dict
        
    def read_graph_gml(self, graphFile):
        # read single graph
        multiGraph = nx.read_gml(graphFile, destringizer=int)
        if 'wifi' in graphFile.lower():
            self.graph_topology_undirected = getTopologyWeighted(multiGraph)
            self.graph_topology_directed = getDirectedTopologyWeighted(multiGraph)
        else:
            self.graph_topology_undirected = getTopology(multiGraph)
            self.graph_topology_directed = getDirectedTopology(multiGraph)
                
    def get_graphs(self, graphFile):
        self.graph_files = []
        if graphFile.endswith('.gml'): 
            # read single graph
            self.read_graph_gml(graphFile)
        else:
            # to read a list of graph
            with open(graphFile, 'r') as f:
                self.graph_files = f.read().splitlines()
                self.graph_topology_undirected = None
                self.graph_topology_directed = None
        
    def get_paths(self, routing):
        if not routing.endswith('.txt'):
            self.paths_all = None
            self.paths = get_paths_from_routing(routing)
            return self.paths
        else:
            self.paths_all = get_paths_from_routing_dynamic(routing)
            self.paths = self.paths_all[0]
            return self.paths_all

    def on_epoch_end(self):
        # Shuffle data at the end of every epoch
        idx = np.random.permutation(self.n)
        self.kpiframe = self.kpiframe[idx]
        self.triframe = self.triframe[idx]
        if self.paths_all is not None:
            self.paths_all = [self.paths_all[i] for i in idx]
        if len(self.graph_files):
            self.graph_files = [self.graph_files[i] for i in idx]
    
    def get_prepare(self, index):
        # Determine which frame and topo to use
        if self.paths_all is not None:
            self.paths = self.paths_all[index]
        if len(self.graph_files):
            self.read_graph_gml(self.graph_files[index])
        return index
    
    def get_kpis_convert(self, raw):
        labels = {}
        for lab in ['delay', 'jitter', 'throughput']:
            labels[lab] = tf.Variable(
                tf.constant(
                    raw[..., self.l2c[lab]::self.n_kpis] / self.scale_dict[lab]
                ),
                name=lab
            )
        l_drops = raw[..., self.l2c['packet_tx']::self.n_kpis] - raw[..., self.l2c['packet_rx']::self.n_kpis] 
        l_drops_rate = l_drops/raw[..., self.l2c['packet_tx']::self.n_kpis]
        labels.update({
            "drops":  tf.Variable(tf.constant(l_drops / self.scale_dict["drops"]),name="drops"), #1
            "drop-rate":  tf.Variable(tf.constant(l_drops_rate),name="drops-rate"), #1
            "drop-bi":  tf.Variable(tf.constant(tf.cast(l_drops_rate > .1, tf.float32)),name="drops-bi") #1
        })    
        return labels
    
    def __getitem__(self, index=None):
        """ 
          Returns features and corresponding expected labels by reading topological features. 
        """
        index = self.get_prepare(index) # here the topo instance is indexed
        
        # Get topological features as dictionary.
        features = self.get_topological_features()

        # Get values for specific data point 
        a = self.triframe[index]
        b = self.kpiframe[index]
        if self.add_noise:
            a += np.random.normal(scale=(a/10)**.5)
                  
        n_links = len(self.graph_topology_directed.edges())
        try:
            f_capacities = []
            for e in self.graph_topology_directed.edges():
                f_capacities.append(self.graph_topology_directed[e[0]][e[1]]['weight'])
        except:
            f_capacities = [1.0]*n_links ##
            
        traffic_in = a[0::2]#[float(a[i]) for i in range(0, len(a), 2)]
        traffic_ot = a[1::2]#[float(a[i]) for i in range(1, len(a), 2)]
        f_traffic = np.array([traffic_in, traffic_ot])

        features["path_init"] = tf.Variable(tf.constant(f_traffic), name="path_init")
        features["link_init"] = tf.Variable(tf.constant(f_capacities), name="link_init")
        
        n_path = len(traffic_in)
        n_kpis = len(b)//n_path
        assert self.n_kpis == n_kpis
                
        # label <--> column index
        labels = self.get_kpis_convert(b)
        
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
        for elem in self.paths:  # for each path
            for i in range(len(list(elem))): # for each node in the path
                if i != len(list(elem))-1:       # if not the last node
                    paths_to_links.append(count)     # append path index
                    sequences_paths_links.append(i)  # append the index of node within the path [??]
                    b = [elem[i], elem[i+1]]         # the link from src node to dst node
                    a = links.index(b)
                    links_to_paths.append(a)         # append link index
            count += 1
        return paths_to_links, links_to_paths, sequences_paths_links  #n_total

    def get_links_nodes(self, links, nodes):
        """
          Extracta links to nodes features.
        """
        nodes_to_links = []  # **list of link indices that start with each node**
        sequences_links_nodes = []  # **chunks nodes_to_links by which nodes**
        count_link = 0
        for elem in nodes:        # for each node
            for i in range(len(links)):   # for each link
                if links[i][0] == elem:          # if this node is the source node of the link 
                    nodes_to_links.append(i)                 # append link index
                    sequences_links_nodes.append(count_link) # append node count (same to node index)
            count_link += 1

        links_to_nodes = []  # **list of src node index of all links (order = link index order)**
        for i in range(len(links)):      # for each link
            links_to_nodes.append(links[i][0])  # append the source node index
        return nodes_to_links, sequences_links_nodes, links_to_nodes  #n_link

    def get_paths_nodes(self):
        """
          Extracts paths to nodes features. 
        """
        nodes_to_paths = []
        paths_to_nodes = []
        sequences_nodes_paths = []
        countP = 0
        for elem in self.paths:  # for each path
            count0 = -1
            for i in range(len(elem)):  # for each node in the path
                count0 += 1
                if i < (len(elem)-1):          # if not the last node
                    nodes_to_paths.append(elem[i])       # append node index
                    paths_to_nodes.append(countP)        # append path index 
                    sequences_nodes_paths.append(count0) # append the index of node in the path 
            countP += 1
        return nodes_to_paths, paths_to_nodes, sequences_nodes_paths  # len = number of links in all paths added up

    def __len__(self):
        return self.n 
    
        
class PlanDataGensMulti(PlanDataGen):
    """
     This class takes multiple input files and return a single combined data generator.
    """
    def __init__(self, filenames, graphFile, routing, label_scale=False, add_noise=False):
        #super().__init__(filenames, graphFile, routing)
        self.l2c = {'packet_tx': 0, 'packet_rx': 1, 'delay': 2, 'jitter': 3, 'throughput': 4}
        self.scale_dict = {'delay': 1, 'jitter': 1, 'throughput': 1, 'drops': 1}
        self.n_kpis = 5
        self.add_noise=add_noise

        # Split key performance indicators and traffic.
        kpis, tris = filenames
        assert len(tris) == len(kpis) == len(routing)
        
        # Initialiaze how kpi as pandas dataframe looks like.
        self.triframeList = [pd.read_csv(t,header=None).reset_index(drop=True).values.astype(np.float32) for t in tris]
        self.kpiframeList = [pd.read_csv(k,header=None).reset_index(drop=True).values.astype(np.float32) for k in kpis]        
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
            is_g, res = self.get_graphs_multi(g)
            if is_g:
                # returned graph instances
                self.graphTopoUndList.append(res[0])
                self.graphTopoDirList.append(res[1])
            else:
                # returned graph file names
                self.graphTopoUndList.append(res)
                self.graphTopoDirList.append(res)
                                
        # Store routing 
        self.pathsList = [self.get_paths(r) for r in routing]
        
        if label_scale is not False: # None or dict
            self.scale_labels(label_scale)        
        
    def scale_labels(self, scale_dict=None):            
        if scale_dict is None:
            # training data, calculate scales
            scale_dict = self.scale_dict
            for lab in scale_dict.keys():
                kpi_curr = []
                if lab != 'drops':
                    for df in self.kpiframeList:
                        kpi_curr.append(df[:, self.l2c[lab]::self.n_kpis].reshape(-1)) # N x 10
                    kpi_curr = np.hstack(kpi_curr)    
                else:
                    kpi_curr = []
                    for df in self.kpiframeList:
                        tx = df[:, self.l2c['packet_tx']::self.n_kpis].reshape(-1)
                        rx = df[:, self.l2c['packet_rx']::self.n_kpis].reshape(-1)
                        kpi_curr.append(tx - rx)
                kpi_curr = np.hstack(kpi_curr)    
                scale_dict[lab] = np.quantile(kpi_curr, 0.75) - np.quantile(kpi_curr, 0.25)
        # update to training scales
        self.scale_dict = scale_dict
        return scale_dict       
        
    def get_graphs_multi(self, g):
        if g.endswith('.gml'): 
            multiGraph = nx.read_gml(g, destringizer=int)
            if 'wifi' in g.lower():
                return (True, (getTopologyWeighted(multiGraph), getDirectedTopologyWeighted(multiGraph)))
            else:
                return (True, (getTopology(multiGraph), getDirectedTopology(multiGraph)))
        else: #txt
            with open(g, 'r') as f:
                graph_files = f.read().splitlines()
            return (False, graph_files)
                
    def on_epoch_end(self):
        # Do not shuffle data at the end of every epoch
        pass
    
    def get_prepare(self, index):
        # Determine which frame and topo to use
        ti = self.proj_topo_index(index)
        fi = self.nCumu[index]
        index -= sum(self.nCumu[:index])
        
        self.triframe = self.triframeList[fi]
        self.kpiframe = self.kpiframeList[fi]
        
        # Graphs
        self.graph_topology_undirected = self.graphTopoUndList[ti]
        if isinstance(self.graph_topology_undirected, (list, tuple)):
            self.read_graph_gml(self.graph_topology_undirected[index])
            
        self.graph_topology_directed = self.graphTopoDirList[ti]
        if isinstance(self.graph_topology_directed, (list, tuple)):
            self.read_graph_gml(self.graph_topology_directed[index])
        
        # Get topological features as dictionary.
        if self.paths_all is not None:
            self.paths = self.paths_all[index]
        return index
        
    
