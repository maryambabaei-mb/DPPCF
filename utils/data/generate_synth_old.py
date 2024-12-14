from reprosyn.methods import MST
import json 
import numpy as np
import pandas as pd
from pmlb import fetch_data
from sklearn.preprocessing import KBinsDiscretizer


DOMAIN_PTH= 'domain_adult.json'

CONTINUOUS_COL=['age', 'capital-gain', 'capital-loss']


def discretisize_continuous_var(df, num_col=CONTINUOUS_COL):
    kbin= KBinsDiscretizer(n_bins=100, encode='ordinal', strategy='uniform')
    kbin.fit(df[num_col])
    df= pd.concat([df.drop(columns=num_col),
                    pd.DataFrame(kbin.transform(df[num_col]), columns= num_col)], axis=1)


def generate_synth(df, domain_pth= DOMAIN_PTH):
    # train reprosyn
    with open(domain_pth, 'r') as js_file:#reposyn read json with json.loads instead of json.load. this  workaround fix the metadata loading issues
            domaine= json.load(js_file)

    gen = MST(dataset=df.copy(), size=df.shape[0] *2, epsilon = 1000, metadata= domaine['columns'])
    #gen= 
    gen.run()
    synth= gen.output
    synth= synth[df.columns]
    synth['target']= synth['target'].astype('int64')
    return synth


if __name__=='__main__':
     adult = fetch_data('adult')
     adult = adult.drop(columns=['education-num','fnlwgt','native-country'])
     synth=generate_synth(adult)
     synth.to_csv('synth.csv', index=False)
