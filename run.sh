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


category=diapers
version=1

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
# # # # # # # # # # # # pipeline for generating questions with attributes # # # # # # # # # # # # # # # #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

# # # # # # # # # # # extract attribute # # # # # # # # #
for category in diapers sofa vacuum tv ; do
    # # # load sampled questions and save those questions with attributes to ${category}_questions_wi_attr.csv
    # python generate.py --task extract_attributes --category ${category} --version=${version}

    # # # # check sampled questions and corresponding attributes, take two and create prompt ./prompt/gen_q_attr_usr_{self.category}.txt
    # # be done manually

    # # # # sample real user question and generate new question based corresponding attribute, saving as gen_pair_{self.category}_{self.model}_100.json
    python generate.py --task generate_q_pair --category ${category} --version=${version}
done