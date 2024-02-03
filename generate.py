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

random.seed(42)
SPLIT="; "

class GPTGenerator(object):
    def __init__(self, args) -> None:
        self.args = args
        self.save_dir = "./data/gen_questions"
        self.prompt_filename = "prompt"
        self.prompt_filename = "prompt_casual_2"
        self.prompt_path = f"./prompt/{self.prompt_filename}.txt"
        self.save_filename = f"gen_attr_vacuum_gpt4.json"
        self.save_path = os.path.join(self.save_dir, f"{self.save_filename}.json")
        self.random_seed = args.seed
        self.category=args.category
        self.version=args.version
        self.annotate_num = 10000
        self.repeat_num = 3 # number of gen times if attr_post does not match attr_prior
        self._decide_model(args)
        self._set_seed()


    def _set_seed(self):
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

    def _decide_model(self, args):
        #"gpt-4-1106-preview", # "gpt-3.5-turbo", # "gpt-4", "gpt-3.5-turbo-1106"
        if args.model_name_or_path == "gpt-4":
            self.model="gpt4"
        elif args.model_name_or_path == "gpt-4-1106-preview":
            self.model="gpt4t"
        elif args.model_name_or_path == "gpt-3.5-turbo":
            self.model="gpt35t"
        elif args.model_name_or_path == "gpt-3.5-turbo-1106":
            self.model="gpt35tnew"

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
        print(f"Saving file {file_path} ...")
        with open(file_path, "w") as tf:
            json.dump(data, tf, indent=2)


    def sample(self):
        """
        Sample an item from the database. 
        Here we sample from a simulated database, constants.py"""
        def get_category():
            return random.choice(list(FEATURE_PER_CATEGORY.keys()))

        def get_brand():
            sample_item["brand"] = random.choice(BRAND[category])

        def get_price():
            sample_item["price"] = random.randint(PRICE[category][0], PRICE[category][1])

        def get_image():
            # not available for now
            sample_item["image"] = random.choice(IMAGE[category])

        def get_size():
            feature, unit, minimum, maximum = SIZE[category]
            size = random.randint(minimum, maximum)
            sample_item[feature] = f"{size} {unit}"

        def get_color():
            weight_options, color_options = [], []
            for color, weight in COLOR[category]:
                weight_options.append(weight)
                color_options.append(color)
            sample_item["color"] = random.choices(color_options, weights=weight_options, k=1)[0]

        def get_rating():
            rating = random.gauss(3.5, 1.5) % 5
            rating = max(rating, 1)
            sample_item["rating"] = round(rating, 1)

        def get_shipping():
            sample_item["shipping"] = random.choice(OTHER["shipping"])

        def get_condition():
            sample_item["condition"] = random.choice(OTHER["condition"][category])

        def get_specific_feature():
            for binary_feature in FEATURE[category]["binary"]:
                sample_item[binary_feature] = random.choice(["Yes", "No"])
            for special_feature, value_list in FEATURE[category]["special"].items():
                # outlier case
                if category == "vacuum" and sample_item["Cordless"] == "Yes" and special_feature == "Cord Length":
                    continue
                sample_item[special_feature] = random.choice(value_list)

        sample_item = {}
        category = get_category()
        sample_item["category"] = category
        for feature in FEATURE_PER_CATEGORY[sample_item["category"]]:
            func_name = f"get_{feature}"
            locals()[func_name]()
        return sample_item

    def parse_output(self, output):
        question_list = output.split("\n")
        return question_list

    def generate_q_with_item(self):
        """
        based on sampled item, along with its attributes, generate a question"""
        # question_prompt = "You are trying to generate variations of diverse customer questions about products within the {category} category. \nPlease come up with {question_num} questions that customers might possibly ask, even those that are specific and technical. Feel free to be creative in your questions.\nThe Questions should be concerning with this product:\n{information}\nQuestions:\n1."
        question_prompt = open(self.prompt_path).read() + "\n"
        sample_num = 10 # number of sampled items
        question_num = 8 # relevant question for each sampled item
        data = []
        sample_item = self.sample()
        information = "\n".join([f"{feature}: {value}" for feature, value in sample_item.items()])
        input_seq = question_prompt.format(category=sample_item["category"], 
                                            question_num=question_num, 
                                            information=information)
        output = openai_api_chat(self.args,  input_seq=input_seq)
        self._save_json(self.parse_output(output), self.save_path)

    def sum_attributes(self):
        attribute_path = "./data/from_walmart/attributes/vacuum_product_attributes.txt"
        data = self._load_txt(attribute_path)
        SYS_PROMPT = (
            'You are a helpful assistant. You have been given a list of attributes (#Attribute 1#, #Attribute 2#, ....) about vacuums. '
            'Your task is to categorize these attributes and provide a concise yet detailed name for each category: #Category#. '
            'Each category may contain only one attribute. Aim to create as many categories as possible. '
            'Avoid using generic terms like "feature," "information," or "attributes" for #Category#. '
            'Please present the results in the following format: \n#Category#: #Attribute 1#, #Attribute 2#, ...\n'
            'Ensure that all #Attribute# are directly copied from the given list.'
        )

        input_seq="\n".join(data)
        categorized_attributes = openai_api_chat(self.args, input_seq=input_seq, system_prompt=SYS_PROMPT)
        pdb.set_trace()
        self._save_json(categorized_attributes, f"{self.save_dir}/attr_vaccum.json")

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
        # self.prompt_path = "prompt/extract_attr.txt"
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

    def generate_q_with_attrs(self):

        attribute_path = f"./data/from_walmart/attributes/{self.category}_product_attributes.txt"
        attrs = self._load_txt(attribute_path)
        # attr = random.sample(attrs)
        questions = {}
        for attr in tqdm(attrs):
            questions[attr] = self.generate_q_with_attr(attr=attr)
        self._save_json(questions, f"{self.save_dir}/gen_attr_{self.category}_{self.model}.json")

    def validate_attr_overlap(self, attr, attr_post):
        # validate if attributes for generation and after generation match
        # here we consider they match if they have at least one same attribute
        intersaction = set(attr.split(SPLIT)).intersection(set(attr_post.split(SPLIT)))
        return len(intersaction) != 0

    def generate_q_with_attr(self, attr):
        """
        based on a sampled attribute, generate a question"""
        SYS_PROMPT = self._load_txt(f"./prompt/gen_q_attr_sys_v{self.version}.txt", readlines=False)
        USER_PROMPT = self._load_txt(f"./prompt/gen_q_attr_usr_{self.category}.txt", readlines=False)
        for i in range(self.repeat_num):
            output = openai_api_chat(
                                self.args, 
                                input_seq=USER_PROMPT.format(feature=attr), 
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
                print(self.validate_attr_overlap(attr, attr_post))
                print(f"attr: {attr}")
                print("attr_post", attr_post)
                print(output)

        return ""

    def validate_attrs(self, attrs, attr_list):
        # validate if extracted attributes are valid (align with attribute list)
        for attr in attrs.split(SPLIT):
            if attr not in attr_list: return False
        return True

    def generate_q_pair(self):
        num_for_attr = 1 # how many data point to sample for each attribute
        attribute_path = f"data/from_walmart/attributes/{self.category}_product_attributes.txt"
        attr_list = self._load_txt(attribute_path)
        csv_path = f"./data/from_walmart/examples/category_questions/{self.category}_questions/{self.category}_questions_wi_attr.csv"
        data_ori = self._load_csv(csv_path)
        new_attr_dict = {}
        for _, row in tqdm(data_ori.iterrows()):
            # row["attribute"] can contain multiple attribute
            if not self.validate_attrs(row["attribute"], attr_list): continue

            for i in range(num_for_attr):
                new_attr_name = str(i) + ": " + row["attribute"]
                if new_attr_name in new_attr_dict: continue
                new_synthesized_question = self.generate_q_with_attr(attr=row["attribute"])
                # pdb.set_trace()
                if not new_synthesized_question: break
                new_attr_dict[new_attr_name] = {
                    "real_user_question": row["question"],
                    "synthesized_question": new_synthesized_question,
                }
                break
            if len(new_attr_dict) >= 100: break
        new_attr_dict = dict(sorted(new_attr_dict.items()))
        # adding index and transform into list
        new_attr_list = [{
            "index" : idx,
            "attr"  : key_,
            "real_user_question": value_["real_user_question"],
            "synthesized_question": value_["synthesized_question"],
        } for idx, (key_, value_) in enumerate(new_attr_dict.items())]
        self._save_json(new_attr_list, f"{self.save_dir}/gen_pair_v{self.version}_{self.category}_{self.model}_100.json")

    def generate_a(self, information=None, question=None):
        """
        based on sampled item and generated questions, generate an answer
        information is created for specific product"""
        SYS_PROMPT = self._load_txt("./prompt/gen_a_sys.txt", readlines=False)
        USER_PROMPT = self._load_txt(f"./prompt/gen_a_usr_{self.category}.txt", readlines=False)
        output = openai_api_chat(
                            self.args, 
                            input_seq=USER_PROMPT.format(question=question), 
                            system_prompt=SYS_PROMPT.format(category=self.category),
                            temperature=1.0
                            )
        output = output.strip('"')
        return output

    def generate_a_pair(self):
        """
        generate answers for created question pairs"""
        questions = self._load_json(os.path.join(self.save_dir, "gen_pair_r2_vacuum.json"))
        answers = []
        for idx, question_pair in tqdm(enumerate(questions.items())):
            if idx > 50 : break
            real_q = question_pair[-1]["real_user_question"]
            synt_q = question_pair[-1]["synthesized_question"]
            answers.append({
                "index": idx,
                "real_user":{
                    "question": real_q,
                    "answer": self.generate_a(question=real_q),
                },
                "synthesized": {
                    "question": synt_q,
                    "answer": self.generate_a(question=synt_q),
                },
            })
        self._save_json(answers, "./data/gen_answers/gen_pair_r2_vacuum.json")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="gpt-4", #"gpt-4-1106-preview", # "gpt-3.5-turbo", # "gpt-4", "gpt-3.5-turbo-1106"
        help="The name of the model for generation",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="extract_attributes",
        help="Choose which task/function to conduct",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="vacuum",
        choices=["vacuum", "tv", "diapers", "sofa", "smartphone"],
        help="Choose which category to process",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=2,
        help="Choose which version of prompt we use for generation",
    )
    # parameters for lm api
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for decoding",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=256,
        help="Maximum decoding length",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Maximum decoding length",
    )
    parser.add_argument(
        "--frequency_penalty",
        type=float,
        default=0.0,
        help="Penalty for token frequency",
    )
    parser.add_argument(
        "--presence_penalty",
        type=float,
        default=0.0,
        help="Penalty for token presence",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()
    return args


def main():

    args = parse_args()
    gen = GPTGenerator(args)

    function = getattr(gen, args.task, None)
    if callable(function):
        function()
    else:
        raise ValueError("Choose a pre-defined task to execute ... ")
        
    # gen.generate_q()
    # gen.generate_q_with_attrs()
    # # gen.extract_attributes()
    # # gen.verify_annotation()
    # # gen.generate_q_pair()
    # # gen.generate_a_pair()
    

if __name__ == "__main__":
    main()