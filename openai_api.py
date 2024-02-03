#!/usr/bin/env python3
#
import sys, os, pdb
import random, time
import openai
openai.api_key = os.getenv("OPENAI_API_KEY") # Use the API
openai.organization = os.getenv("OPENAI_ORG_ID")


def openai_api_chat(args, model=None, input_seq=None, system_prompt=None, temperature=None):
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
    if temperature is None: temperature = args.temperature

    for delay_secs in (2**x for x in range(10)):
        try:
            response = openai.ChatCompletion.create(
                model=model,  # assuming the GPT-4 model identifier
                temperature=temperature,
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

