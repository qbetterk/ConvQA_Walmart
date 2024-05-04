#!/usr/bin/env python3
#
import sys, os, pdb
import json, math
import random, argparse, time
import numpy as np
import pandas as pd
from tqdm import tqdm
from openai_api import openai_api_chat
from constants import *
from base import BaseClass
from parse_args import parse_args

SPLIT="; "

class GPTGeneratorBase(BaseClass):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.args = args
        self.random_seed = args.seed
        self.category = args.category
        self.sample_num = args.sample_num
  
    def sample_product(self, attrs=None):
        """
        sample a product in self.category
        all databases locate in data/from_walmart/products/{category}/[1-20].json
        raw data is stored in form List[Dict{"CATLG_ITEM_ID": int(), "PROD_ATTR_NM_VAL_LST_TXT": List[Dict{"key": str(), "value": str()}]}]
        we randomly select one and return in terms of a long string 'str(): str()\nstr():str()\n...'"""
        default_attrs = ["brand", "model"]
        if attrs: default_attrs += attrs
        file_idx = random.randint(1, 20) # 0 for test
        file_path = f"./data/from_walmart/products/{self.category}/{file_idx}.json"
        data = self._load_json(file_path)
        item = random.choice(data) # data[0] #
        attr_list = []
        for attr in item["PROD_ATTR_NM_VAL_LST_TXT"]:
            if attrs and attr["key"] not in default_attrs: continue
            if len(str(attr["value"])) > 1000: continue
            if attr["value"] == "N": attr["value"] = "No"
            if attr["value"] == "Y": attr["value"] = "Yes"
            attr_list.append(attr["key"]+": "+str(attr["value"]))
        return "\n".join(attr_list)
        # return "\n ".join([attr["key"]+": "+str(attr["value"]) for attr in item["PROD_ATTR_NM_VAL_LST_TXT"]])


class GPTGeneratorQ(GPTGeneratorBase):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.save_dir = args.save_dir if args.save_dir else "./data/gen_questions/"
        self.annotate_num = 10000
        self.repeat_num = 3 # number of gen times if attr_post does not match attr_prior
        self.version_q = args.version_q
        self.gen_q_with_item = args.gen_q_with_item
        self.prompt_sys_path = args.prompt_path if args.prompt_path else f"./prompt/gen_q_attr_sys_v{self.version_q}.txt"

    def extract_attributes(self):
        # # # load questions
        csv_path = f"./data/from_walmart/examples/category_questions/{self.category}_questions/{self.category}_questions_wi_attr.csv"
        if os.path.exists(csv_path):
            question_list = self._load_csv(csv_path)
        else:
            question_list = self._load_csv(f"./data/from_walmart/examples/category_questions/{self.category}_questions/{self.category}_questions.csv")
        # # # setup progress bar
        self.progress_bar = tqdm(range(min(self.annotate_num, len(question_list))))
        # # # annotation and add to an extra column
        question_list['attribute'] = question_list.apply(self.extract_attr_row, axis=1)
        # Save the modified DataFrame to the original CSV file
        question_list.to_csv(csv_path, index=False)

    def extract_attr_row(self, row):
        """input is a row, just to fit function extract_attrs"""
        self.progress_bar.update(1)
        if "attribute" in row and row["attribute"] != "none": return row["attribute"]
        if row.name >= self.annotate_num: return "none"
        return self.extract_attr_q(row['question'])

    def extract_attr_q(self, question):
        # # # extract attribute from sampled questions
        prompt = (
            "You are a helpful assistant. Here is a list of ATTRIBUTES related to {category}:\n\n{attribute_list}\n\n"
            "You will be given a question about {category}. "
            "Your task is to determine which ATTRIBUTE the question is referring to. "
            "Please directly give the ATTRIBUTE. All ATTRIBUTES should be directly copied from the above list."
            "If you find no ATTRIBUTE from the above list match, please reply: no_match"
            "If a question applies to multiple attributes, list all that apply, connected with \"{split}\""
        )
        question = question.strip('"')
        category = self.category
        attribute_list = open(f"data/from_walmart/attributes/{self.category}_product_attributes.txt").read()
        SYSTEM_PROMPT = prompt.format(category=category, attribute_list=attribute_list, split=SPLIT)
        attr = openai_api_chat(self.args, input_seq=question, system_prompt=SYSTEM_PROMPT, temperature=0).strip('"')
        return attr

    def verify_annotation(self):
        annotation_path = f"./data/from_walmart/examples/category_questions/{self.category}_questions/{self.category}_questions_wi_attr.csv"
        data = self._load_csv(annotation_path)
        attribute_list = open(f"data/from_walmart/attributes/{self.category}_product_attributes.txt").read().split("\n")
        count_error, count_total = 0, 0
        for index, row in data.iterrows():
            if row.name >= self.annotate_num: continue
            count_total += 1
            if row["attribute"] == "no_match": continue
            if row["attribute"] in attribute_list: continue
            if ", " in row["attribute"]:
                for attr in row["attribute"].split(", "):
                    if attr not in attribute_list:
                        count_error += 1
                        print(index, attr, row["attribute"], "-"*10,  row["question"])
            else:
                count_error += 1
                print(index, row["attribute"], "-"*10,  row["question"])
        print(count_error, count_total)

    def generate_q_pair(self):
        num_for_attr = 1 # how many data point to sample for each attribute
        attribute_path = f"data/from_walmart/attributes/{self.category}_product_attributes.txt"
        attr_list = self._load_txt(attribute_path)
        csv_path = f"./data/from_walmart/examples/category_questions/{self.category}_questions/{self.category}_questions_wi_attr.csv"
        data_ori = self._load_csv(csv_path)
        new_pair_dict = {}
        # database = self.sample_product() if self.gen_q_with_item else ""
        for _, row in tqdm(data_ori.iterrows()):
            # row["attribute"] can contain multiple attribute
            if not self.validate_attrs(row["attribute"], attr_list): continue

            for i in range(num_for_attr):
                new_attr_name = str(i) + ": " + row["attribute"]
                if new_attr_name in new_pair_dict: continue
                if self.gen_q_with_item:
                    database = self.sample_product(attrs=row["attribute"].split("; "))
                    self.prompt_usr_path = f"./prompt/gen_q_attr_usr_item_{self.category}.txt"
                    new_synthesized_question = self.generate_q_with_attr(attr=row["attribute"], database=database)
                else:
                    self.prompt_usr_path = f"./prompt/gen_q_attr_usr_{self.category}.txt"
                    new_synthesized_question = self.generate_q_with_attr(attr=row["attribute"])
                # pdb.set_trace()
                if not new_synthesized_question: break

                new_pair_dict[new_attr_name] = {
                    "real_user_question": row["question"],
                    "synthesized_question": new_synthesized_question,
                    "database": database,
                }
                break
            if len(new_pair_dict) >= self.sample_num: break
        new_pair_dict = dict(sorted(new_pair_dict.items()))
        # adding index and transform into list
        new_pair_list = [{
            "index" : idx,
            "attr"  : key_.split(": ")[-1],
            "real_user_question": value_["real_user_question"],
            "synthesized_question": value_["synthesized_question"],
            "database": value_["database"],
        } for idx, (key_, value_) in enumerate(new_pair_dict.items())]

        save_file_name = f"gen_pair_v{self.version_q}_{self.category}_{self.model}_{self.sample_num}.json" if not self.args.save_filename else self.args.save_filename
        self._save_json(new_pair_list, os.path.join(self.save_dir, save_file_name))

    def generate_q_with_attrs(self):
        attribute_path = f"./data/from_walmart/attributes/{self.category}_product_attributes.txt"
        attrs = self._load_txt(attribute_path)
        # attr = random.sample(attrs)
        questions = {}
        for attr in tqdm(attrs):
            questions[attr] = self.generate_q_with_attr(attr=attr)
        self._save_json(questions, f"{self.save_dir}/gen_attr_{self.category}_{self.model}.json")

    def generate_q_with_attr(self, attr=None, database=None):
        """
        based on a sampled attribute
        or 
        based on a sampled attribute and a sampled product info (attr-value pair), generate a question"""
        SYS_PROMPT = self._load_txt(self.prompt_sys_path, readlines=False)
        USER_PROMPT = self._load_txt(self.prompt_usr_path, readlines=False)
        if self.gen_q_with_item:
            USER_PROMPT = USER_PROMPT.format(feature=attr, database=database)
        else:
            USER_PROMPT = USER_PROMPT.format(feature=attr)
        for i in range(self.repeat_num):
            output = openai_api_chat(
                                self.args, 
                                input_seq=USER_PROMPT, 
                                system_prompt=SYS_PROMPT.format(category=self.category),
                                temperature=1.0
                                )
            output = output.strip('"')
            # verify if generated question match the attr
            attr_post = self.extract_attr_q(output)
            # if attr_post == attr or attr in attr_post:
            if self.validate_attr_overlap(attr, attr_post):
                return output
            else:
                print("")
                print(f"attr: {attr}")
                print("attr_post", attr_post)
                print(output)
        return ""

    def validate_attr_overlap(self, attr, attr_post):
        # validate if attributes for generation and after generation match
        # here we consider they match if they have at least one same attribute
        intersaction = set(attr.split(SPLIT)).intersection(set(attr_post.split(SPLIT)))
        return len(intersaction) != 0

    def validate_attrs(self, attrs, attr_list):
        # validate if extracted attributes are valid (align with attribute list)
        for attr in attrs.split(SPLIT):
            if attr not in attr_list: return False
        return True


class GPTGeneratorA(GPTGeneratorBase):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.save_dir = "./data/gen_answers"
        self.questions_dir = "./data/gen_questions"
        self.version_q = args.version_q
        self.version_a = args.version_a

    def generate_a(self, product_info=None, question=None):
        """
        based on sampled item and generated questions, generate an answer
        information is created for specific product"""
        SYS_PROMPT = self._load_txt(f"./prompt/gen_a_sys_v{self.version_a}.txt", readlines=False)
        USER_PROMPT = self._load_txt(f"./prompt/gen_a_usr.txt", readlines=False)
        # pdb.set_trace()
        output = openai_api_chat(
                            self.args, 
                            input_seq=USER_PROMPT.format(product_info=product_info, question=question), 
                            system_prompt=SYS_PROMPT.format(category=self.category),
                            temperature=0.1
                            )
        output = output.strip('"')
        return output

    def generate_a_wi_item(self):
        """
        generate answers for created questions (ignore real questions),
        based on information of a selected item
        """
        questions = self._load_json(os.path.join(self.questions_dir, 
                            f"gen_pair_v{self.version_q}_{self.category}_{self.model}_{self.sample_num}.json"))
        answers = []
        for question_pair in tqdm(questions[:self.sample_num]):
            # real_q = question_pair["real_user_question"]
            synt_q = question_pair["synthesized_question"]
            # product_info = {attr.split(": ")[0]:attr.split(": ")[1] for attr in question_pair["database"].split("\n")}
            product_info = question_pair["database"]
            # pdb.set_trace()
            for attr in question_pair["attr"].split(": ")[-1].split("; "):
                if f"{attr}: " not in product_info:
                    # pdb.set_trace()
                    product_info += f"\n{attr}: No"
                    print(f"\n{attr}: No")

            answers.append({
                "index": question_pair["index"],
                "attr": question_pair["attr"].split(": ")[-1],
                # "real_user":{
                #     "question": real_q,
                #     "answer": self.generate_a(product_info=product_info, question=real_q),
                # },
                "synthesized": {
                    "question": synt_q,
                    "answer": self.generate_a(product_info=product_info, question=synt_q),
                },
                "database": product_info,
            })
        self._save_json(answers, f"{self.save_dir}/gen_vq{self.version_q}a{self.version_a}_{self.category}_{self.model}_{self.sample_num}.json")


def main():
    args = parse_args()
    if args.target == "question":
        gen = GPTGeneratorQ(args)
    elif args.target == "answer":
        gen = GPTGeneratorA(args)
    else:
        raise ValueError("Choose to generate questions or answers")


    function = getattr(gen, args.task, None)
    if callable(function):
        function()
    else:
        raise ValueError("Choose a pre-defined task to execute ... ")
    

if __name__ == "__main__":
    main()