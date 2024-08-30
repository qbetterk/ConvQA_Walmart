#!/bin/bash
set -xue



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # pipeline for generating QA pairs and dialogues # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# version_q=6
# version_a=1
# category=vacuum
# sample_num_qa=2000
# sample_num_dial=1000
# for category in vacuum sofa diapers tv; do
#     python generate.py \
#         --target pair \
#         --task gen_pairs \
#         --model_name_or_path gpt-4o \
#         --sample_num ${sample_num_qa} \
#         --category ${category} \
#         --version_a ${version_a} \
#         --version_q ${version_q}

#     python generate.py \
#         --target dial \
#         --task generate_dial \
#         --model_name_or_path gpt-4o \
#         --sample_num ${sample_num_dial} \
#         --category ${category} \
#         --version_a ${version_a} \
#         --version_q ${version_q}
# done  



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # pipeline for iteratively refine generation prompt # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# version=1
# category=sofa
# for category in vacuum sofa diapers tv; do
#     sample_num=20
#     TARGET_DIR="prompt/edit_q_from_q/${category}/"
#     TARGET_FILE="${TARGET_DIR}/refine_gen_q_sys_v1.txt"
#     SOURCE_FILE="prompt/edit_q_from_q/initial_prompt.txt"
#     for version in 1 2 3 4 5 6 7; do
#         # Check if the target directory does not exist
#         if [ ! -d "$TARGET_DIR" ]; then
#             # Create the target directory
#             mkdir -p "$TARGET_DIR"
#             echo "Directory created successfully."
#         fi

#         # Check if the target file does not exist
#         if [ ! -f "$TARGET_FILE" ]; then
#             # Copy the source file to the target location
#             cp "$SOURCE_FILE" "$TARGET_FILE"
#             echo "File copied successfully."
#         fi

#         # # generating questions with items (attribute and value pairs)
#         python generate.py \
#                 --target "question" \
#                 --task "generate_q_pair" \
#                 --model_name_or_path "gpt-4o" \
#                 --sample_num ${sample_num} \
#                 --gen_q_with_item True \
#                 --category ${category} \
#                 --version_q ${version} \
#                 --save_dir "./data/gen_questions/refine/${category}/" \
#                 --prompt_path "./prompt/edit_q_from_q/${category}/refine_gen_q_sys_v${version}.txt"

#         # revise prompt
#         python prompt_tuning.py \
#             --target "question" \
#             --task "edit_from_q" \
#             --model_name_or_path "gpt-4" \
#             --category ${category} \
#             --sample_num ${sample_num} \
#             --version_q ${version} \
#             --frequency_penalty 0.7
#     done
# done

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # pipeline for iteratively refine evaluation prompt # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# category=vacuum
# sample_num=100
# version=1
# for version in 4 5 6; do
#     # gpt 4 evaluation
#     python eval.py \
#             --target "question" \
#             --task "evaluate_q" \
#             --model_name_or_path "gpt-4-turbo" \
#             --sample_num ${sample_num} \
#             --version_q ${version} \
#             --prompt_path "./prompt/edit_eval_prompt/refine_eval_q_sys_v${version}.txt" \
#             --top_p 0.1

#     # revise prompt
#     python prompt_tuning.py \
#         --target "question" \
#         --task "edit_eval_prompt" \
#         --model_name_or_path "gpt-4-turbo" \
#         --category "vacuum" \
#         --version_q ${version} \
#         --top_p 0.1


# done




# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # pipeline for generating questions with attributes # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# version=3
# # # # # # # # # # # # extract attribute # # # # # # # # #
# for category in sofa tv ; do
# # for category in vacuum; do
#     # # # # load sampled questions and save those questions with attributes to ${category}_questions_wi_attr.csv
#     # python generate.py \
#     #         --target question \
#     #         --task extract_attributes \
#     #         --category ${category} \
#     #         --version_q=${version}

#     # # # # check sampled questions and corresponding attributes, take two and create prompt ./prompt/gen_q_attr_usr_{self.category}.txt
#     # # be done manually

#     # # # # option 1: generating questions with only attributes
#     # # # # # sample real user question and generate new question based corresponding attribute, saving as gen_pair_{self.category}_{self.model}_100.json
#     # python generate.py \
#     #         --target question \
#     #         --task generate_q_pair \
#     #         --category ${category} \
#     #         --version_q=${version}

#     # # option 2: generating questions with items (attribute and value pairs)
#     # python generate.py \
#     #         --target question \
#     #         --task generate_q_pair \
#     #         --model_name_or_path gpt-4-0125-preview \
#     #         --sample_num 100 \
#     #         --gen_q_with_item True \
#     #         --category ${category} \
#     #         --version_q=${version}




#     # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     # # # # # # # # # # # # pipeline for generating answers with database # # # # # # # # # # # # # # # #
#     # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
#     version_a=1
#     version_q=3
#     python generate.py \
#             --target answer \
#             --task generate_a_wi_item \
#             --model_name_or_path gpt-4-0125-preview \
#             --sample_num 100 \
#             --category ${category} \
#             --version_a ${version_a} \
#             --version_q ${version_q}


# done




# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # evaluation question and answer with walmart metrics # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# version_a=1
# category=vacuum
# sample_num=100
# for version_q in 1 2 3 4 5 6 7; do
#     # python generate.py \
#     #     --target answer \
#     #     --task generate_a_wi_item \
#     #     --model_name_or_path gpt-4o \
#     #     --sample_num 100 \
#     #     --category ${category} \
#     #     --version_a ${version_a} \
#     #     --version_q ${version_q}
#     # python eval.py \
#     #     --target "question" \
#     #     --task "evaluate_wi_walmart_metrics" \
#     #     --model_name_or_path "gpt-4o" \
#     #     --version_a ${version_a} \
#     #     --version_q ${version_q}
#     python eval.py \
#             --target "question" \
#             --task "compute_score_file"\
#             --model_name_or_path "gpt-4o" \
#             --version_a ${version_a} \
#             --version_q ${version_q}
# done
