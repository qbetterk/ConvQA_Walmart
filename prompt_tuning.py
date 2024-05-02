#!/usr/bin/env python3
#
import sys, os, pdb
import random, json
from tqdm import tqdm
from openai_api import openai_api_chat
from base import BaseClass
from parse_args import parse_args

SPLIT="; "

class GPTPromptEditorBase(BaseClass):
    def __init__(self, args) -> None:
        self.args = args
        self.category = args.category
        self.sample_num = args.sample_num
        self.gen_q_dir = "./data/gen_questions/refine"
        self.ann_q_dir = "./annotations/processed"
        

class GPTPromptEditorQ(GPTPromptEditorBase):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.version_q = args.version_q
        self.sample_num = 20
        self.edit_prompt_sys_path = "./prompt/edit_q_from_q_sys_v1.txt"
        self.edit_prompt_usr_path = "./prompt/edit_q_from_q_usr_v1.txt"
        self.gen_q_prompt_sys_path = f"./prompt/edit_q_from_q_v1/refine_gen_q_sys_v{self.version_q}.txt"
        self.gen_q_prompt_usr_path = "./prompt/gen_q_attr_usr_item_vacuum.txt"
        self.gen_q_pair_path = os.path.join(self.gen_q_dir, "gen_pair_v1_vacuum_gpt4t_20.json")
        self.preference_path = os.path.join(self.ann_q_dir, "r1_vacuum.json")


    def uniform_q_pair(self, data):
        if type(data) == list:
            return data
        elif type(data) == dict:
            return [{
                "index": idx,
                "attr": key_,
                "real_user_question": data[key_]["real_user_question"],
                "synthesized_question": data[key_]["synthesized_question"],
            } for idx, key_ in enumerate(data)]
        else:
            raise ValueError("Unknown data structure, need to implement")

        
    def edit_from_q(self):
        """
        This function edits question generation prompt by comparing real_user_question and synthesized_question directly
        and does not consider preference.
        """
        # original_prompt = self._load_txt(f"./prompt/gen_q_attr_sys_v1.txt")
        q_pairs = self._load_json(self.gen_q_pair_path)
        q_pairs = self.uniform_q_pair(q_pairs)
        q_pairs_sample = random.sample(q_pairs, k=self.sample_num)
        gen_q_pair = ""
        for pair in q_pairs_sample:
            gen_q_pair += f"PRODUCT FEATURE DATABASE:\n{pair['database']}\n"
            gen_q_pair += f"FEATURE: {pair['attr']}\n"
            gen_q_pair += f"Real User Question: {pair['real_user_question']}\n"
            gen_q_pair += f"Generated Question: {pair['synthesized_question']}\n\n"
            # print("real user questions:", pair["real_user_question"])
            # print("generated questions:", pair["synthesized_question"])
            # print()

        gen_q_prompt_sys = self._load_txt(self.gen_q_prompt_sys_path, readlines=False)
        gen_q_prompt_usr = self._load_txt(self.gen_q_prompt_usr_path, readlines=False)
        SYS_PROMPT = self._load_txt(self.edit_prompt_sys_path, readlines=False)
        USER_PROMPT = self._load_txt(self.edit_prompt_usr_path, readlines=False)
        USER_PROMPT = USER_PROMPT.format(gen_q_prompt_sys=gen_q_prompt_sys, gen_q_prompt_usr=gen_q_prompt_usr, gen_q_pair=gen_q_pair)
        print(SYS_PROMPT, "\n")
        print(USER_PROMPT)
        output = openai_api_chat(self.args, input_seq=USER_PROMPT, system_prompt=SYS_PROMPT, temperature=0)
        with open(f"./prompt/edit_q_from_q_v1/refine_gen_q_sys_v{self.version_q + 1}.txt", "w") as tf:
            tf.write(output)
            print(f"Saving revised prompt file: ./prompt/edit_q_from_q_v1/refine_gen_q_attr_sys_v{self.version_q + 1}.txt ...")


    def edit_from_p(self):
        """
        This function edits question generation prompt based on human/machine's preference between
        real_user_question and synthesized_question"""
        
        q_pairs = self._load_json(self.preference_path)
        q_pairs_needimprove = [pair for pair in q_pairs if pair["annotation"] == "Question A"] # filter only pairs where annotators think Qestion A (real user) is better
        if len(q_pairs_needimprove) > self.sample_num:
            print(f"There are {len(q_pairs_needimprove)} pairs, we randomly sample {self.sample_num} to modify the prompt. ")
            q_pairs_needimprove = random.sample(q_pairs, k=self.sample_num)
        for pair in q_pairs_needimprove:
            print("real user questions:", pair["pair"]["Question A (real)"])
            print("generated questions:", pair["pair"]["Question B (synt)"].split("Real User Scenario: ")[-1].strip('"'))
            print()



def main():
    args = parse_args()
    if args.target == "question":
        gen = GPTPromptEditorQ(args)
    elif args.target == "answer":
        pass # TODO implement GPTEvalA()
    else:
        raise ValueError("Choose to generate questions or answers")

    function = getattr(gen, args.task, None)
    if callable(function):
        function()
    else:
        raise ValueError("Choose a pre-defined task to execute ... ")
    

if __name__ == "__main__":
    main()