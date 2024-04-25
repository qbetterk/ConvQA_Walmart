#!/usr/bin/env python3
#
import sys, os, pdb
from tqdm import tqdm
from openai_api import openai_api_chat
from base import BaseClass
from parse_args import parse_args

SPLIT="; "

class GPTEvalBase(BaseClass):
    def __init__(self, args) -> None:
        self.args = args
        self.category = args.category
        self.sample_num = args.sample_num
        self.gen_q_dir = "./data/gen_questions"
        self.eval_q_dir = "./data/eval_questions"

    def eval(self, str1, str2):
        data_path = os.path.join(self.gen_q_path, "gen_pair_r1_vacuum.json")

    def openai_api_call(self, sys_prompt, user_prompt, args):
        return openai_api_chat(args, input_seq=user_prompt, system_prompt=sys_prompt, temperature=0.1)
        

class GPTEvalQ(GPTEvalBase):
    def __init__(self, args) -> None:
        super().__init__(args)
        self.eval_q_path = os.path.join(self.gen_q_dir, "gen_pair_r1_vacuum.json")
        self.version_q = 1


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

        
    def evaluate_q(self):
        """
        compare generated question and real user question pair and give preference
        """
        SYS_PROMPT = self._load_txt(f"./prompt/eval_q_sys_v{self.version_q}.txt", readlines=False)
        USER_PROMPT = self._load_txt(f"./prompt/eval_q_usr_{self.category}.txt", readlines=False)
        data_eval = self._load_json(self.eval_q_path)
        data_eval = self.uniform_q_pair(data_eval)
        
        for pair in data_eval:
            real_user_question = pair["real_user_question"]
            synthesized_question = pair["synthesized_question"]
        SYS_PROMPT = SYS_PROMPT.format(category=self.category)
        USER_PROMPT = USER_PROMPT.format(str1=real_user_question, str2=synthesized_question)
        output = openai_api_chat(self.args, input_seq=USER_PROMPT, system_prompt=SYS_PROMPT, temperature=0.1)
        return output


def main():
    args = parse_args()
    if args.target == "question":
        gen = GPTEvalQ(args)
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