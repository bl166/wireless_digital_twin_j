import re
import os
import numpy as np

# some predefined paths
SD_STAR_PAR = [[1,4],[2,5],[3,6]]
SD_NSF_1 = [[0, 7],[1, 11],[2, 13],[3, 0],[6, 5],[9, 12],[10, 8],[11, 10],[12, 9], [13, 3]]
SD_NSF_2 = [[9,1],[3, 13],[12, 0],[10, 3],[13, 7], [1, 8],[2, 9],[11, 2],[6, 12],[5, 11],[0, 10],[7, 4]]
SD_GBN_1 = [[0,15],[2,6], [8,13],[14,1],[4,12], [16,5],[15,10], [9,2],[11,4],[13,0]]


def get_topo_meta(topology):
    if topology in ['star', 'star-inter']:
        return 7
    elif topology in ['parallel', 'parallel-inter']:
        return 9
    elif topology in ['NSFNet', 'wifi']:
        return 14;
    elif topology == 'GBN':
        return 17
    elif topology.startswith('wifi-grid'):
        if sz := re.findall("-(\d+)x(\d+)", topology):
            s1, s2 = [int(s) for s in sz[0]]
        else:
            s1, s2 = 4, 4 #default: 4x4
        return s1*s2
    else:
        raise ValueError(f'No such topology: {topology}')
        

def generate_sdpairs(sd_path, paths_flows, num_flows, topology, num_samples, random_seed, fix_source=False):
    np.random.seed(random_seed)
    num_devices = get_topo_meta(topology)
    paths = int(paths_flows)
    flows = round(paths_flows - paths, 2)
    path_mode = f'{topology}-{paths}'
    
    print(flows, paths)
    
    if not os.path.exists(sd_path):
        si = list(range(num_devices))
        sd_pairs = None
        for ns in range(num_samples):
            # fixed paths
            if 'star' in topology or 'parallel' in topology:
                sd_pairs = SD_STAR_PAR  
            elif paths == 0:
                # random but fixed
                if sd_pairs is None:
                    sd_pairs = [np.random.choice(num_devices,2, replace=False).tolist()]
                    while len(sd_pairs) < num_flows:
                        snode = np.random.choice(np.setdiff1d(si, [sd[0] for sd in sd_pairs]))
                        dnode = np.random.choice(np.setdiff1d(si, [sd[1] for sd in sd_pairs]+[snode]))
                        sd_pairs.append([snode, dnode])                           
            elif paths == 1:
                if path_mode == 'GBN-1':
                    sd_pairs = SD_GBN_1
                else:
                    sd_pairs = SD_NSF_1
            elif path_mode == 'NSFNet-2':
                sd_pairs = SD_NSF_2
            else:
                raise ValueError(f'No such paths: {path_mode}')
                
            # random flows (partially)
            if flows > 0: 
                n = int(round(num_flows*flows))
                sd_pairs = []
                while len(sd_pairs) < num_flows:
                    ni = len(sd_pairs)
                    if ni < num_flows-n:
                        snode = SD_NSF_1[ni][0]
                        dnode = SD_NSF_1[ni][1]
                    else:
                        if fix_source: # random dst nodes only, fix source
                            snode = SD_NSF_1[ni][0]
                        else:
                            snode = np.random.choice(np.setdiff1d(si, [sd[0] for sd in sd_pairs]))
                        dnode = np.random.choice(np.setdiff1d(si, [sd[1] for sd in sd_pairs]+[snode]))
                    sd_pairs.append([snode, dnode])                      
            
            with open(sd_path, 'a') as f:
                f.write(';'.join([','.join([str(s) for s in ss]) for ss in sd_pairs])+';\n')
    return sd_path
            
    
def load_tensor_file(path, dtype=int): #std::vector<std::vector<std::vector<uint32_t>>>
    tensor = []
    with open(path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = re.findall('([\d\.]+),([\d\.]+);', line)
        tensor.append([[dtype(p) for p in pp] for pp in parts])
    return np.array(tensor, dtype)
   