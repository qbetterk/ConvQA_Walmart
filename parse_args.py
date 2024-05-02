import random, argparse, time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="gpt-4", #"gpt-4-1106-preview", # "gpt-3.5-turbo", # "gpt-4", "gpt-3.5-turbo-1106"
        help="The name of the model for generation",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="question",
        choices=["question", "answer"],
        help="Choose to generate questions or answers",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="extract_attributes",
        help="Choose which task/function to conduct",
    )
    parser.add_argument(
        "--gen_q_with_item",
        type=bool,
        default=True,
        help="Choose whether to use an item or just an attribute to generate a question",
    )
    parser.add_argument(
        "--category",
        type=str,
        default="vacuum",
        choices=["vacuum", "tv", "diapers", "sofa", "smartphone"],
        help="Choose which category to process",
    )
    parser.add_argument(
        "--version_a",
        type=int,
        default=1,
        help="Choose which version of prompt we use for answer generation",
    )
    parser.add_argument(
        "--version_q",
        type=int,
        default=2,
        help="Choose which version of prompt we used for question generation",
    )
    parser.add_argument(
        "--sample_num",
        type=int,
        default=20,
        help="Choose how many samples to generate for both questions and answers",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./data/gen_questions",
        help="Claim dir for saving files",
    )
    parser.add_argument(
        "--save_filename",
        type=str,
        default="",
        help="Claim filename for saving files",
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        default="",
        help="Claim path to prompt file",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for decoding",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=4096,
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
