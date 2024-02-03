#!/usr/bin/env python3
#
import sys, os, pdb
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from openai_api import openai_api_chat



class PreProcessData(object):
    """docstring for PreProcessData"""
    def __init__(self):
        super(PreProcessData, self).__init__()



    def _load_json(self, path=None):
        if path is None or not os.path.exists(path):
            raise IOError('File does not exist: %s' % path)
            # return None
        with open(path) as df:
            data = json.loads(df.read())
        return data

    
    def _load_txt(self, path=None, split_tok="\n"):
        if path is None or not os.path.exists(path):
            raise IOError('File does not exist: %s' % path)
        with open(path) as df:
            data = df.read().strip().split(split_tok)
        return data


    def _load_csv(self, path=None, sep=","):
        if path is None or not os.path.exists(path):
            raise IOError('File does not exist: %s' % path)
        with open(path) as df:
            data = pd.read_csv(df, sep=sep)
        return data

    def _save_json(self, data, file_path):
        dir_path = "/".join(file_path.split("/")[:-1])
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)
        # print(f"Saving file {file_path} ...")
        with open(file_path, "w") as tf:
            json.dump(data, tf, indent=2)

    def analysis(self):
        # sample_path = "./Product_QA_Samples.csv"
        sample_path = "./data/examples/category_questions/tv_questions/tv_questions.csv"
        data = self._load_csv(sample_path)
        data_sample = data[:1000]
        data_in_list = data_sample.question.tolist()
        pdb.set_trace()
    
    
def main():
    proc = PreProcessData()
    # proc.analysis()
    proc.process_attributes()


if __name__ == "__main__":
    main()
