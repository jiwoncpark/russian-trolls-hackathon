# Part of Deep Lab package
# Author: Hatef Monajemi (monajemi@stanford.edu)

import json
import hashlib
import pandas as pd

class Dump():
    
    def __init__(self, path):
        self.path=path
        
        try:
            with open(self.path, 'r') as fp:
                self.results = json.load(fp)
        except:
                self.results = {}

    def count(self):
        return(len(self.results))

    def append(self, x):
        
        json_x = json.dumps(x)
        hash = hashlib.sha1(json_x.encode("UTF-8")).hexdigest()
        hash = hash[:10];   # take only the first 10. it s enough here
        tmp  = {hash:x}
        self.results.update(**tmp)
    
    def save(self):
        with open(self.path, 'w') as fp:
            json.dump(self.results, fp)

    def to_csv(self):
        df = pd.DataFrame.from_dict(self.results)
        df = df.transpose()
        filename = self.path
        filename = filename.split('.')
        if len(filename)>1:
            filename[-1] = 'csv'

        filename = '.'.join(filename)
        df.to_csv(filename)
