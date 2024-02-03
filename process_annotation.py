#!/usr/bin/env python3
#
import sys, os, pdb
import json, math
import random, argparse, time
import numpy as np
import pandas as pd
from tqdm import tqdm


random.seed(42)

class AnnoProcessor(object):
    def __init__(self) -> None:

        self.category="vacuum"
        self.save_path = f"./annotations/processed/r2_{self.category}.json"
        self.source_path_ann = f"./annotations/raw/export-result_batch3.jsonl"
        self.source_path_data = f"./data/result_gen/gen_pair_r2_{self.category}.json"


    def _load_json(self, path):
        if path is None or not os.path.exists(path):
            raise IOError(f"File doe snot exists: {path}")
        print(f"Loading data from {path} ...")
        with open(path) as df:
            data = json.loads(df.read())
        return data
    
    def _load_txt(self, path=None, split_tok="\n", readlines=True):
        if path is None or not os.path.exists(path):
            raise IOError('File does not exist: %s' % path)
        with open(path) as df:
            if readlines:
                data = df.read().strip().split(split_tok)
            else:
                data = df.read()
        return data


    def _load_jsonl(self, path):
        if path is None or not os.path.exists(path):
            raise IOError(f"File doe snot exists: {path}")
        print(f"Loading data from {path} ...")
        data = []
        with open(path) as df:
            for row in df.readlines():
                data.append(json.loads(row))
        return data


    def _save_json(self, data, file_path):
        dir_path = "/".join(file_path.split("/")[:-1])
        if dir_path and not os.path.exists(dir_path):
            os.makedirs(dir_path)
        print(f"Saving file {file_path} ...")
        with open(file_path, "w") as tf:
            json.dump(data, tf, indent=2)

    def process(self):
        data_ann = self._load_jsonl(self.source_path_ann)
        data_src = self._load_json(self.source_path_data)
        data_proc = []
        for idx, item in enumerate(data_src.items()):
            data_proc.append({
                "index": idx,
                "attribute": item[0],
                "pair": {
                    "Question A (real)": item[-1]["real_user_question"],
                    "Question B (synt)": item[-1]["synthesized_question"],
                },
                "annotation": list(data_ann[idx]['projects'].values())[0]['labels'][0]['annotations']['classifications'][0]['radio_answer']['name']
            })


        self._save_json(data_proc, self.save_path)

def main():
    proc = AnnoProcessor()
    proc.process()
    

if __name__ == "__main__":
    main()