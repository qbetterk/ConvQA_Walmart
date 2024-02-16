#!/bin/bash
set -xue

# export HF_HOME=/local/data/shared/huggingface_cache
# export HF_CACHE_HOME=/local/data/shared/huggingface_cache
# export HF_DATASETS_CACHE=/local/data/shared/huggingface_cache
# # columbia
# export OPENAI_API_KEY="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"
# export OPENAI_ORG_ID="org-vBJ9i7PnvVK5CUVrD7emHHC2"
# salesforce
export OPENAI_API_KEY="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"
export OPENAI_ORG_ID="org-Y7zi4Bj4dOxFLW4CpEFnAffn"


# python generate.py
# python preprocess.py
# python process_annotation.py


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # pipeline for generating questions with attributes # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# version=3
# # # # # # # # # # # # extract attribute # # # # # # # # #
# # for category in diapers sofa vacuum tv ; do
# for category in vacuum ; do
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

#     # option 2: generating questions with items (attribute and value pairs)
#     python generate.py \
#             --target question \
#             --task generate_q_pair \
#             --gen_q_with_item True \
#             --category ${category} \
#             --version_q=${version}
# done




# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # pipeline for generating answers with database # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
version_a=1
version_q=3
category=vacuum
python generate.py \
        --target answer \
        --task generate_a_pair \
        --category ${category} \
        --version_a ${version_a} \
        --version_q ${version_q}