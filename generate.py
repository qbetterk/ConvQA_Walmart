#!/usr/bin/env python3
#
import sys, os, pdb
import json
import random, argparse, time
import numpy as np
from tqdm import tqdm
from openai_api import openai_api_chat, openai_api_ppl, construct_input
from constants import *

random.seed(42)

class GPTGenerator(object):
    def __init__(self, args) -> None:
        self.args = args
        self.save_dir = "./result"
        # self.save_filename = f"gen_q{args.q_model_version}_a{args.a_model_version}_item{args.sample_num}_qnum{args.question_num}.json"
        self.save_path = "./result/gen_q_test_10.json"
        self.random_seed = args.seed
        self._set_seed()


    def _set_seed(self):
        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

    def _load_json(self, path):
        if path is None or not os.path.exists(path):
            raise IOError(f"File doe snot exists: {path}")
        print(f"Loading data from {path} ...")
        with open(path) as df:
            data = json.loads(df.read())
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
        for idx in range(len(question_list)):
            if question_list[idx][1:].startswith(". "):
                question_list[idx] = question_list[idx][3:]
        return question_list

    def generate_q(self):
        """
        based on sampled item, along with its attributes, generate a question"""
        question_prompt = "You are trying to generate variations of diverse customer questions about products within the {category} category. \nPlease come up with {question_num} questions that customers might possibly ask, even those that are specific and technical. Feel free to be creative in your questions.\nThe Questions should be concerning with this product:\n{information}\nQuestions:\n1."
        sample_num = 10 # number of sampled items
        question_num = 8 # relevant question for each sampled item
        data = []
        for _ in range(1000):
            sample_item = self.sample()
            information = "\n".join([f"{feature}: {value}" for feature, value in sample_item.items()])
            input_seq = question_prompt.format(category=sample_item["category"], question_num=question_num, information=information)
            output = openai_api_chat(self.args,  input_seq=input_seq)
            for question in self.parse_output(output):
                answer = self.generate_a(category=sample_item["category"], information=information, question=question)
                data.append({
                    "sample_item": sample_item,
                    # "information": information,
                    "question": question,
                    "answer": answer,
                })
            if len(data) >= sample_num:
                break
        self._save_json(data, self.save_path)


    def generate_a(self, category, information, question):
        """
        based on sampled item and generated questions, generate an answer"""
        answer_prompt = "You are a helpful EcommerceBot that answers user's questions about {category}. Please provide a detailed answer based on production information and user's question.\nProduct Information:\n{information}\nQuestion:\n{question}\nAnswer:\n"
        input_seq = answer_prompt.format(category=category, information=information, question=question)
        output = openai_api_chat(self.args,  input_seq=input_seq)
        return output



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file_path",
        type=str,
        default="",
        help="The path to the input data for topic extraction / summarization / ppl computation, usually not used together with --file_name",
    )
    parser.add_argument(
        "--save_file_name",
        type=str,
        default="alpaca_data_sample_topic.json",
        help="The path to save the data with extracted topic",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="gpt-3.5-turbo", # "gpt-3.5-turbo", # "gpt-4"
        help="The name of the model for generation",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="get_topic",
        choices=["get_topic", "get_ppl", "get_topic_summarized", "get_rewriting"],
        help="Choose which task to conduct",
    )
    parser.add_argument(
        "--sample_num",
        type=int,
        default=500,
        help="The number of sampled datapoint",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for decoding",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=1024,
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
    parser.add_argument(
        "--prompt_style",
        type=str,
        default="instruction",
        choices=["instruction", "instruction_input", "instruction_input_output"],
        help="Choose what is contained within the prompt",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    gen = GPTGenerator(args)
    gen.generate_q()
    

if __name__ == "__main__":
    main()