import itertools
import pandas as pd
import networkx as nx
import numpy as np
from pprint import pprint

def graph_from_routing(frout, max_neighb):
    with open(frout) as f:
        ftext = f.readlines()

    df_tab, headers = None, None
    ftext_nodes = [list(g) for m, g in itertools.groupby(ftext, key=lambda x: x!='HNA Routing Table: empty\n') if m]
    
    for n, fn in enumerate(ftext_nodes):
        fn = fn[2:-1]
        ni = int(fn[0].split(',')[0].split(':')[-1])

        # headers
        if headers is None:
            headers = fn[1].replace('\n', '').split()
            del headers[2]
            headers.insert(0, 'CurrNode')
            headers = np.array(headers)
        if df_tab is None:
            df_tab = pd.DataFrame(columns=headers)         

        # routes
        routes = [x.replace('\n', '').split() for x in fn[2:]]
        routes_clean = []
        for ri in routes:
            try:
                routes_clean.append([
                    n,                           # curr
                    int(ri[0].split('.')[-1])-1, # destination
                    int(ri[1].split('.')[-1])-1, # next
                    int(ri[3])                   # distance
                ])
            except:
                print(ri)
        if routes:
            df_tab = pd.concat((df_tab, pd.DataFrame(np.array(routes_clean), columns=headers)))

    nodes = set(df_tab[df_tab.Distance <= max_neighb][headers[[0,1]]].values.reshape(-1))
    for n in range(len(ftext_nodes)):
        if n not in nodes:
            df_tab = pd.concat((df_tab, pd.DataFrame(np.array([[n,n,n,0]]), columns=headers)))
        
    df_edges_1 = df_tab[df_tab.Distance <= max_neighb].reset_index(drop=True)
    #df_edges_1.columns = ['source', 'target', 'weight']
        
    graph = nx.from_pandas_edgelist(
        df_edges_1, source='CurrNode', target='Destination', edge_attr='Distance',
        create_using=nx.DiGraph()
    )
    return graph, df_edges_1


def compare_olsr_routings(file1, file2):
    # reading files
    f1 = open(file1, "r") 
    f2 = open(file2, "r") 

    f1_data = f1.readlines()
    f2_data = f2.readlines()
        
    if len(f1_data) != len(f2_data):
        return False
   
    same_flag = True
    for i, (line1, line2) in enumerate(zip(f1_data,f2_data)):
        i += 1
        # matching line1 from both files
        if line1 == line2 or line1.startswith('Node') and line2.startswith('Node'):
            continue
            # print IDENTICAL if similar
            print("Line ", i, ": IDENTICAL")      
        else:
            print("Line ", i, ":")
            # else print that line from both files
            print("\tFile 1:", line1, end='')
            print("\tFile 2:", line2, end='')
            same_flag = False

    # closing files
    f1.close()                                      
    f2.close()        
    return same_flag


#=========== PARSE OLSR ROUTING TABLES NS3 ===========

def create_IP_to_node_map(filename):
    file1 = open(filename, 'r')
    Lines = file1.readlines()
    file1.close()
   
    IP_to_node = {}
    for line in Lines:
        tmp=line.strip().split(':')
        node_id = tmp[0]
        IP_addresses = tmp[1].split('\t')
        #print(IP_addresses)
        for ip in IP_addresses:
            IP_to_node[ip] = int(node_id)

    return IP_to_node


def compute_path_taken(ip_map_fn, rout_tab_fn, source_idx, destination_idx):
    IP_to_node_idx = create_IP_to_node_map(ip_map_fn)
    #print(IP_to_node_idx)

    #process the OLSR routing file
    file1 = open(rout_tab_fn, 'r')
    Lines = file1.readlines()
    file1.close()

    path = find_path_taken_by_packet(Lines, IP_to_node_idx, source_idx, destination_idx) + [destination_idx]
    return path

def find_path_taken_by_packet(olsr_table, IP_to_node_idx, source_idx, destination_idx):   
    if (source_idx == destination_idx):
        return []
   
    #find the source IP in the routing table
    for idx, line in enumerate(olsr_table):
        node_identifier = "Node: " + str(source_idx) +"," in line
        prot_identifier = "OLSR" in line
        if node_identifier and prot_identifier:
            loc = idx+2
            break
           
    #find the destination IP in the sub-table corresponding to the source node
    for idx2, line in enumerate(olsr_table[loc:]):
        result = line.split()[0]
        if (IP_to_node_idx[result] == destination_idx):
            next_hop_IP = line.split()[1]
            next_hop_idx = IP_to_node_idx[next_hop_IP]
            break

    return [source_idx] + find_path_taken_by_packet(olsr_table, IP_to_node_idx, next_hop_idx, destination_idx)

#=========== PARSE OLSR ROUTING TABLES NS3 (END) ===========
