#!/usr/bin/env python3
#
import sys, os, pdb
import pandas as pd
import re

from tqdm import tqdm
from openai_api import openai_api_chat
from base import BaseClass
from parse_args import parse_args
from collections import defaultdict

SPLIT="; "

class GPTEvalBase(BaseClass):
    def __init__(self, args) -> None:
        self.args = args
        self.category = args.category
        self.sample_num = args.sample_num
        self.gen_q_dir = "./data/gen_questions/"
        self.eval_q_dir = args.save_dir if args.save_dir else "./annotations/auto/"

    def openai_api_call(self, sys_prompt, user_prompt, args):
        return openai_api_chat(args, input_seq=user_prompt, system_prompt=sys_prompt, temperature=0.1)
        

class GPTEvalQ(GPTEvalBase):
    def __init__(self, args) -> None:
        super().__init__(args)
        # self.eval_q_path = os.path.join(self.gen_q_dir, "gen_pair_r1_vacuum.json")
        self.eval_q_path = os.path.join("./annotations/processed/r1_vacuum.json")
        self.version_q = args.version_q
        self.prompt_sys_path = args.prompt_path if args.prompt_path else f"./prompt/eval_q_sys_v{self.version_q}.txt"
        self.prompt_usr_path = f"./prompt/eval_q_usr_{self.category}.txt"


    def uniform_q_pair(self, data):
        if type(data) == list:
            if "pair" in data[0]:
                # human annotated files
                data_new = []
                for pair in data:
                    pair["attr"] = pair["attribute"]
                    pair["real_user_question"] = pair["pair"]["Question A (real)"]
                    pair["synthesized_question"] = pair["pair"]["Question B (synt)"]
                    data_new.append(pair)
                return data_new
            else:
                # normal generated pairs
                return data
        elif type(data) == dict:
            # old generated pairs
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
        SYS_PROMPT = self._load_txt(self.prompt_sys_path, readlines=False)
        SYS_PROMPT = SYS_PROMPT.format(category=self.category)
        USER_PROMPT = self._load_txt(self.prompt_usr_path, readlines=False)
        data_eval = self._load_json(self.eval_q_path)
        data_eval = self.uniform_q_pair(data_eval)
        count = 0
        # print(SYS_PROMPT)
        for pair in tqdm(data_eval):
            real_user_question = pair["real_user_question"]
            synthesized_question = pair["synthesized_question"]
            feature = pair["attr"]
            USER_PROMPT_ = USER_PROMPT.format(real_user_question=real_user_question, synthesized_question=synthesized_question, feature=feature)
            output = openai_api_chat(self.args, input_seq=USER_PROMPT_, system_prompt=SYS_PROMPT, temperature=0.1)
            pair["AutoEval"] = output.split("\"\n\n")[0].replace("Preference: ", "").strip('"')
            # print(USER_PROMPT_)
            # pdb.set_trace()
            if pair["annotation"].startswith(pair["AutoEval"]):
                count += 1

            print(pair["AutoEval"], pair["annotation"], count)

        print(count, len(data_eval), count/len(data_eval))
        self._save_json(data_eval, os.path.join(self.eval_q_dir, f"autoeval_v{self.version_q}.json"))


    def evaluate_wi_walmart_metrics(self, model_name="gpt_4"):
        prompt_list = self._load_json(f"prompt/eval/evaluator_judge_prompts_{model_name}.json")
        gen_prompt = self._load_txt(f"prompt/edit_q_from_q_v1/refine_gen_q_sys_v{self.version_q}.txt")
        if os.path.exists(os.path.join(self.eval_q_dir, f"autoeval_qa_walmart_{model_name}_q{self.version_q}a1.json")):
            data = self._load_json(os.path.join(self.eval_q_dir, f"autoeval_qa_walmart_{model_name}_q{self.version_q}a1.json"))
        else:
            data = self._load_json(f"data/gen_answers/gen_vq{self.version_q}a1_vacuum_gpt4o_100.json")
        count = {}
        # print(SYS_PROMPT)
        for pair in tqdm(data):
            question = pair["synthesized"]["question"]
            answer = pair["synthesized"]["answer"]
            database = pair["database"]
            for metric in prompt_list:
                if metric == "DA_instruction": continue
                if metric not in count:
                    count[metric] = defaultdict(int)
                
                if metric in pair:
                    output = pair[metric]
                else:
                    SYS_PROMPT = prompt_list[metric]["sys"]
                    USER_PROMPT = prompt_list[metric]["q"].format(
                                                    question=question, 
                                                    answer=answer, 
                                                    context=database,
                                                    instructions=gen_prompt)
                    output = openai_api_chat(self.args, input_seq=USER_PROMPT, system_prompt=SYS_PROMPT, temperature=0.1)
                    output = self.normalize(output)
                    pair[metric] = output
                count[metric][output] += 1

        print(count)
        self._save_json(count, os.path.join(self.eval_q_dir, f"result_autoeval_qa_walmart_{model_name}_q{self.version_q}a1.json"))
        self._save_json(data, os.path.join(self.eval_q_dir, f"autoeval_qa_walmart_{model_name}_q{self.version_q}a1.json"))


    def normalize(self, output):
        output = output.strip("()")
        if output[0] in "ABCDE":
            output = output[0]
        return output

    def compute_score(self, count):
        metric_bi = ["Friendliness (A)", "Quality (A)", "Friendliness (Q)", "Quality (Q)", ]
        metric_tri = ["Entailment (A)", "Verbosity (A)", "Customer Safety (A)", "Brand Safety (A)", "Brand Preference (A)", 
                        "Question Relevance (A)", "Context Relevance PDP (A)", "Brand Safety (Q)", "Brand Preference (Q)", 
                        "Customer Safety (Q)", "Verbosity (Q)", "Prompt Leakage (A)", "Coherence (A)"]
        metric_qua = ["Context Relevance PLP (Q)"]
        metric_five = ["Non_Answerability Compliance (A)", "Truthfulness (A)"]
        scores = {}
        for metric in count:
            score = 0
            total_num = sum(count[metric].values())
            if metric != "Non_Answerability Compliance (A)": continue

            if metric in metric_bi: choices = 2
            elif metric in metric_tri: choices = 3
            elif metric in metric_qua: choices = 4
            elif metric in metric_five: choices = 5
            else:
                print(metric)
                raise ValueError("Unknown metric detected ... ")

            for choice in count[metric]:
                choice_norm = self.normalize(choice)
                if choice_norm in "ABCDE":
                    score += count[metric][choice] * (choices - "ABCDE".index(choice_norm)) / choices 
                if choice_norm == "NA":
                    # total_num -= count[metric][choice]
                    score += count[metric][choice] / choices 
            # import pdb
            # pdb.set_trace()
            scores[metric] = score / total_num
        return scores

    def compute_score_file(self, model_name="gpt_4"):
        count = self._load_json(os.path.join(self.eval_q_dir, f"result_autoeval_qa_walmart_{model_name}_q{self.version_q}a1.json"))
        scores = self.compute_score(count)
        print(scores["Non_Answerability Compliance (A)"])

        # Path to your existing CSV file
        file_path = os.path.join(self.eval_q_dir, f"results.csv")

        # Check if the CSV exists
        if os.path.exists(file_path):
            # Read the existing CSV into a DataFrame
            existing_df = pd.read_csv(file_path, index_col=0)
        else:
            # Create an empty DataFrame with the appropriate index if CSV does not exist
            existing_df = pd.DataFrame(index=scores.keys())  # Assuming all dicts have the same keys


        # Convert the dictionary to a DataFrame, transpose it, and give it a new column name
        new_column = pd.DataFrame([scores]).T
        new_column.columns = [f"{self.version_q}"]  # You might want to dynamically set this based on your data

        # Concatenate the new column to the existing DataFrame
        updated_df = pd.concat([existing_df, new_column], axis=1)

        # Write the updated DataFrame back to the CSV file
        updated_df.to_csv(file_path)


class GPTEvalDial(GPTEvalBase):
    def __init__(self, args) -> None:
        super().__init__(args)
        # self.eval_path = "data/gen_dial_topdown/gen_vq3a1_vacuum_gpt4o_20.json" 
        self.eval_path = "data/gen_dial/gen_vq3a1_vacuum_gpt4o_20.json" 

    def evaluate(self):
        # Initialize score accumulators
        total_scores = {
            "Coherence": 0,
            "Informativeness": 0,
            "Truthfulness": 0,
            "Naturalness": 0,
            "Completeness": 0,
            "Overall Quality": 0
        }
        data = self._load_json(self.eval_path)
        prompt = self._load_txt("prompt/eval/dial_eval.txt", readlines=False)
        
        # Evaluate each dialogue and accumulate scores
        for row in tqdm(data):
            dialogue = "\n".join(row["turns"])
            scores = openai_api_chat(
                            self.args,
                            input_seq="",
                            system_prompt=prompt.format(dialogue=dialogue),
                            temperature=0.1
                            )
            # pdb.set_trace()
            scores = self.parse_scores(scores)
            row["scores"] = scores
            for key in total_scores:
                total_scores[key] += scores[key]
        
        # Compute average scores
        num_dialogues = len(data)
        average_scores = {key: total_scores[key] / num_dialogues for key in total_scores}
        
        # Print average scores
        print("Average Scores:")
        for aspect, score in average_scores.items():
            print(f"{aspect}: {score:.2f}")
        self._save_json(data, self.eval_path)

    def parse_scores(self, raw_text):

        # Define a dictionary to store the scores
        scores = {}

        # Regular expression patterns to find scores and their justifications
        patterns = {
            "Coherence": r"Coherence: Score: (\d)",
            "Informativeness": r"Informativeness: Score: (\d)",
            "Truthfulness": r"Truthfulness: Score: (\d)",
            "Naturalness": r"Naturalness: Score: (\d)",
            "Completeness": r"Completeness: Score: (\d)",
            "Overall Quality": r"Overall Quality: Score: (\d)"
        }

        # Iterate over the patterns and extract scores
        for key, pattern in patterns.items():
            match = re.search(pattern, raw_text)
            if match:
                scores[key] = int(match.group(1))
            else:
                scores[key] = None  # No score found, assign None

        return scores




def main():
    args = parse_args()
    if args.target == "question":
        gen = GPTEvalQ(args)
    elif args.target == "answer":
        pass # TODO implement GPTEvalA()
    elif args.target == "dial":
        gen = GPTEvalDial(args)
    else:
        raise ValueError("Choose to generate questions or answers")

    function = getattr(gen, args.task, None)
    if callable(function):
        function()
    else:
        raise ValueError("Choose a pre-defined task to execute ... ")
    

if __name__ == "__main__":
    main()