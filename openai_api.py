#!/usr/bin/env python3
#
import sys, os, pdb
import random, time
import openai
openai.api_key = os.getenv("OPENAI_API_KEY") # Use the API
openai.organization = os.getenv("OPENAI_ORG_ID")

def construct_input(instruction=None, test_case=None):

    # Instruction = """
    # Consider the following serval sentences, do you know what topic each is about? Some of them may have the same topic. Please reponse in only one word.
    # """
    Instruction = """
    Consider the following instruction, what topic it is about? Ignore the context of the input and output. Please reponse in only one word.
    """ if instruction is None else instruction

    # Instruction = """
    # Try to combine the following words and phrases into serveral clusters. And give a name for each cluster. put the name at the beginning of each cluster.
    # """

    # Test_case = """
    # What are the three primary colors.
    # Give three tips for staying healthy.
    # Describe the structure of an atom.
    # Generate a list of ten items a person might need for a camping trip.
    # What color is it when you mix red and blue.
    # """ if test_case is None else test_case

    Test_case = """
    Give three tips for staying healthy.\nWhat are the three primary colors?\nDescribe the structure of an atom.\nHow can we reduce air pollution?\nDescribe a time when you had to make a difficult decision.\nIdentify the odd one out.\nExplain why the following fraction is equivalent to 1/4\nWrite a short story in third person narration about a protagonist who has to make an important career decision.\nRender a 3D model of a house\nEvaluate this sentence for spelling and grammar mistakes\nHow did Julius Caesar die?\nWhat is the capital of France?\nGenerate a list of ten items a person might need for a camping trip\nDiscuss the causes of the Great Depression\nClassify the following into animals, plants, and minerals\nExplain the use of word embeddings in Natural Language Processing\nDescribe the function of a computer motherboard\nReverse engineer this code to create a new version\nPropose an ethical solution to the problem of data privacy\nGenerate three verbs that mean the same as "to apologize"\nWhat does DNA stand for?\nCompare and contrast the Cuban Missile Crisis and the Vietnam War.\nGenerate a list of random words.\nTranscribe the recording into text.\nWho is the world\'s most famous painter?\nExplain the concept of cogging torque.\nLook up the boiling point of water.\nDescribe the life and reign of King Charles II.\nFind the area of a circle given its radius.\nIdentify the lines of longitude that divides North and South America.\nExplain the process of cellular respiration in plants.\nRewrite the following sentence using active voice.\nGenerate a list of adjectives that describe a person as brave.\nOutline the consequences of deforestation.\nDevelop a plan to reduce electricity usage in a home.\nArrange the words in the given sentence to form a grammatically correct sentence.\nAnalyze the given text for its tone.\nUse the given data to calculate the median.\nRewrite the given paragraph in a shorter, easier to understand form.\nDesign a logo for a website about recycling.\nGenerate a poem that expresses joy.\nConvert the given equation into an algebraic expression.\nBrainstorm possible solutions to reduce water pollution.\nExplain why the given definition is wrong.\nVariable x is defined as “4x + 2y = 10”. Find the value of x.\nWrite a short paragraph about the given topic.\nExplain the concept of artificial intelligence in simple terms.\nDesign an app for a delivery company.\nSummarize the given passage.\nExtract the facts from the paragraph.
    """ if test_case is None else test_case

    Input = f"{Instruction} {Test_case}"
    # print(Input)
    return Input


def openai_api_chat(args, model=None, input_seq=None, system_prompt=None):
    if system_prompt:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": input_seq},
        ]
    else:
        messages = [
            {"role": "user", "content": input_seq},
        ]
    model = model if model is not None else args.model_name_or_path

    for delay_secs in (2**x for x in range(10)):
        try:
            response = openai.ChatCompletion.create(
                model=model,  # assuming the GPT-4 model identifier
                temperature=args.temperature,
                max_tokens=args.max_length,
                top_p=args.top_p,
                n=1,
                frequency_penalty=args.frequency_penalty,
                presence_penalty=args.presence_penalty,
                messages=messages,
            )
            break
        
        except openai.OpenAIError as e:
            randomness_collision_avoidance = random.randint(0, 1000) / 1000.0
            sleep_dur = delay_secs + randomness_collision_avoidance
            print(f"Error: {e}. Retrying in {round(sleep_dur, 2)} seconds.")
            time.sleep(sleep_dur)
            continue
            
    output_seq = response['choices'][0]['message']['content']
    return output_seq



def openai_api_ppl(args, input_seq=None):
    for delay_secs in (2**x for x in range(0, 6)):
        try:
            response = openai.Completion.create(
                prompt=input_seq,
                model="text-davinci-003",
                temperature=args.temperature,
                top_p=args.top_p,
                n=1,
                frequency_penalty=args.frequency_penalty,
                presence_penalty=args.presence_penalty,
                logprobs=0,
                max_tokens=0,
                echo=True,
            )
            break
        
        except openai.OpenAIError as e:
            randomness_collision_avoidance = random.randint(0, 1000) / 1000.0
            sleep_dur = delay_secs + randomness_collision_avoidance
            print(f"Error: {e}. Retrying in {round(sleep_dur, 2)} seconds.")
            time.sleep(sleep_dur)
            continue
    
    assert input_seq == response['choices'][0]['text']
    # Get the token probabilities
    token_probs = response['choices'][0]['logprobs']['token_logprobs'][1:]
    ppl = -1 * sum(token_probs) / len(token_probs)
    return ppl