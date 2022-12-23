import configparser

def str2num(x):
    if x.lower()[0] in ['t', 'f']:
        return eval(x)
    return float(x) if '.' in x else int(x)


class PathConfigParser(configparser.ConfigParser):
    def __init__(self, cpath):
        super().__init__()
        self.optionxform = str
        self.read(cpath)
        
    def as_dict(self):
        d = dict(self._sections)
        for k in d:
            d[k] = dict(self._defaults, **d[k])
            d[k].pop('__name__', None)
            if k.lower() == 'paths':
                for kk, v in d[k].items():
                    d[k][kk] = v.strip().split()
            else:
                for kk, v in d[k].items():
                    v = v.strip()
                    if v[0] in ['[', '('] and v[-1] in [']', ')']:
                        d[k][kk] = [str2num(ss) for ss in v[1:-1].split(',')]
                    else:
                        d[k][kk] = str2num(v)             
        return d

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    

