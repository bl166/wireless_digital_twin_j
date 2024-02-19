import itertools
import re
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


#================= DATA PROCESS FUNCTIONS ==================
def plot_histogram(x, axis, qtiles, **argw):
    q_lo, q_hi = np.percentile(x, qtiles)
    bin_width = 2 * (q_hi - q_lo) * len(x) ** (-1/3)
    bins = round((x.max() - x.min()) / bin_width)
    if 'bins' not in argw:
        argw['bins'] = bins
    axis.hist(x, **argw)
    return bins

def clean_flows(orig_files, clean_files, npath=10, intv=5, dmax=None, filter_inplace=True):
    """
    Clean up missing flows
    ----
    Inputs
        - orig_files: (name of traffic file, name of kpis file)
        - clean_files: name of *cleaned* files if to save
    """
    Files = {
        'fi_kpis': open(orig_files[1],  "r"),
        'fo_kpis': open(clean_files[1], "w") if clean_files is not None and clean_files[0] is not None else None,
        'fi_traf': open(orig_files[0],  "r"),
        'fo_traf': open(clean_files[0], "w") if clean_files is not None and clean_files[1] is not None else None,
    }
    idx_del = []
    stats = {'outliers':[], 'unfinished': [], 'missing': []}
    for i, (l_kpis, l_traf) in enumerate(zip(Files['fi_kpis'], Files['fi_traf'])):
        if i in idx_del:
            continue        
        kpis = [float(k) for k in l_kpis.replace('\n', '').split(',')]
        wflag = True
        if len(kpis) == intv*npath:
            if dmax is not None and max(kpis[2::intv]) > dmax: # invalid
                wflag = False
                idx_del.append(i)
                #print(f'Outlier flow(s) in: #{i}, violation:', max(kpis[2::intv]))
                stats['outliers'].append(i)
        else:
            wflag = False
            idx_del.append(i)
            if l_kpis.startswith('0,'):
                #print(f'Unfinished flow(s) in: #{i}')
                stats['unfinished'].append(i)
            else:
                #print(f'Missing flow(s) in: #{i}')
                stats['missing'].append(i)
        # write to new file        
        if wflag or not filter_inplace:
            if Files['fo_kpis'] is not None:
                Files['fo_kpis'].write(l_kpis)
            if Files['fo_traf'] is not None:
                Files['fo_traf'].write(l_traf)
                
    {f.close() for f in Files.values() if f is not None}
    
    # summary
    print('\n'.join([f'{len(ii)} {ki} samples: {str(ii[:5])}...' for ki, ii in stats.items()]))
                
    return idx_del 

def get_kpis_distribution(fname_kpis, intv=5):
    # kpis: tx_1, rx_1, delay_1, jitter_1, ..., tx_10, rx_10, delay_10, jitter_10 
    kpis = {'drops': [], 'delay':[], 'jitter':[], 'throughput':[]}
    with open(fname_kpis) as f:
        for fl in f.readlines():
            tx = [float(i) for i in fl.replace('\n','').split(',')][0::intv]
            rx = [float(i) for i in fl.replace('\n','').split(',')][1::intv]
            p = [(t-r)/t for t,r in zip(tx, rx)]
            l = [float(i) for i in fl.replace('\n','').split(',')][2::intv]
            kpis['drops'].append(p)
            kpis['delay'].append(l)
            
            if intv > 3:
                j = [float(i) for i in fl.replace('\n','').split(',')][3::intv]
                kpis['jitter'].append(j)
            if intv > 4:
                t = [float(i) for i in fl.replace('\n','').split(',')][4::intv]
                kpis['throughput'].append(t)                
            
    for k in kpis.keys():
        kpis[k] = np.array(sum(kpis[k],[])).flatten()
        
    return kpis


def find_real_idx(idx_done_del, idx_to_del, n):
    idx_zeros = np.zeros(n)
    idx_zeros[idx_to_del] = 1
    idx_zeros = np.delete(idx_zeros, idx_done_del)
    return np.where(idx_zeros)[0].tolist()


def merge_data_parts(nkpis, fns_parts, fn_save, force=False):
    lines = []
    num_samples = 0
    for i, fn in enumerate(fns_parts):
        num_this = [int(i[1]) - int(i[0]) for i in re.findall('_IDX\+(\d+)-(\d+).txt', fn)][0]
        placehold = ','.join(['0']*nkpis)+'\n'
        with open(fn, 'r') as f:
            lines_this = f.readlines()
            lines += lines_this+[placehold for _ in range(num_this-len(lines_this))]
        num_samples += num_this
    if not force:
        if os.path.exists(fn_save):
            print('double check!!', fn_save)
            raise
    with open(fn_save, 'w') as f:
        f.write(''.join(lines))
    return num_samples


def edit_lines_in_file(fname, i_list, val_list=None, op=None):
    with open(fname, 'r') as f:
        curr_lines = f.readlines()
    for ii in sorted(enumerate(i_list), key=lambda x:x[1], reverse=True): 
        if op.lower()[0] == 'd': #delete
            if ii[1] < len(curr_lines):
                del curr_lines[ii[1]]
        elif op.lower()[0] == 'a': #add 
            curr_lines.insert(ii[1], val_list[ii[0]])
        elif op.lower()[0] == 'r': #replace
            curr_lines[ii[1]] = val_list[ii[0]]
        else:
            raise ValueError
    with open(fname, 'w') as f:
        f.write(''.join(curr_lines))

        
def align_kpi_values_across(path_format, idx_del, del_inplace=False):
    idx_del_fin = []
    for mi in sorted(np.unique(sum(list(idx_del.values()),[])), reverse=True):
        f_no_miss, f_miss = [], []
        for k,v in idx_del.items():
            if mi in v:
                f_miss.append(k)
            else:
                f_no_miss.append(k)
        if 'kpis' in f_miss or len(f_no_miss)<=1: # missing in main file or all val files
            idx_del_fin.append(mi)
            for m in f_no_miss: # remove from all kpi files
                if del_inplace:
                    edit_lines_in_file(path_format(m), [mi], op='del')
        else:
            # replace invalid values with valid ones
            with open(path_format(f_no_miss[0]), 'r') as f:
                v_kpis = f.readlines()[mi]
            for m in f_miss:
                edit_lines_in_file( path_format(m), [mi], [v_kpis], op='replace')   
    return idx_del_fin

#=============== DATA PROCESS FUNCTIONS (end) ===============

