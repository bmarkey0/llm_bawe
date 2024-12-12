import argparse
import json

import pandas as pd
import pyarrow.parquet as pq
import os
from openai import OpenAI
import random
def json_for_prompt(doc_id, chunk_1, model = "gpt-4o-mini"):
    number = random.randint(1, 1000000)
    obj = {
        "custom_id": f"{doc_id}_{args.temp}_{number}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                          "text":
                              "You're a teacher grading a paper. Use a five point scale to grade whether the following paper uses correct grammar. Do not summarize or provide feedback.Just give the grade(e.g., level 2, level 4).Use the following five point scale, where level 5 is the highest and level 1 is the lowest: 5: Skillful use of varied sentence structure contributes to fluidity ofideas. Use of standard English grammar, punctuation, capitalization, and spelling demonstrates consistent command of the communication of ideas. 4: Sentence structure is varied and demonstrates language facility.Use of grammar, punctuation, capitalization, and spelling demonstrates appropriate command of standard English conventions. 3: Sentence structure is controlled, though somewhat simplistic.Inconsistent use of correct grammar, punctuation, capitalization, and / or spelling contains a few distracting errors, demonstrating inconsistent command of standard English conventions and the clear communication of ideas inconsistent. 2: Sentence structure is partially controlled, somewhat simplistic, or lacking appropriate language facility. Inconsistent use of correct grammar, punctuation, capitalization, and / or spelling contain multiple distracting errors, demonstrating partial command of standard English conventions. 1: Sentence structure is simplistic or confusing.Use of grammar, punctuation, capitalization, and / or spelling evidences a density and variety of severe errors, demonstrating lack of command of standard English conventions, often obscuring meaning."
                          }
                    ]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": chunk_1
                        }
                    ]
                }
            ],
            "max_completion_tokens": 10,
            "temperature": args.temp,
            "logprobs": True,
            "top_logprobs": 3
        }
    }

    return json.dumps(obj)

def get_prompts(sourcefile):
    prompts = pq.read_table(sourcefile).to_pandas()

    return prompts["filename"].tolist(), prompts["text"].tolist()

def dump_batch(sourcefile, outfile, model, size=None):
    doc_ids, prompts = get_prompts(sourcefile)

    if size is not None:
        doc_ids = doc_ids[:size]
        prompts = prompts[:size]

    with open(outfile, "w") as o:
        for doc_id, prompt in zip(doc_ids, prompts):
            out = json_for_prompt(doc_id, prompt, model)

            o.write(out)
            o.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload JSONL-formatted prompts for OpenAI models.")
    parser.add_argument("chunks", help="Parquet file containing document chunks to complete.")
    parser.add_argument("--model", help="OpenAI model to use")
    parser.add_argument("--size", type=int, help="Maximum number of prompts to generate")
    parser.add_argument("--temp", type=float, help="Prompt temperature")
    parser.add_argument("outcsv", help="directory to csv of batch ids")
    parser.add_argument("--discipline", type=str, help="types of papers being sent to OPENAI")

    args = parser.parse_args()

    out_df = pd.read_csv(args.outcsv)

    outfile = args.model + ".jsonl"

    print("Creating JSONL")
    dump_batch(args.chunks, outfile, args.model, args.size)

    print("Uploading batch to OpenAI")

    client = OpenAI(
        organization="org-cimVbBLWakLMoKplp71wMCo6",
        project="proj_fESDY7nPwkz1bGxzR1pS9Pus"
    )

    batch_input_file = client.files.create(
        file=open(outfile, "rb"),
        purpose="batch"
    )

    batch = client.batches.create(
        input_file_id=batch_input_file.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"
    )

    new_row = {"parquet": args.discipline, "temp": args.temp, "batch_id": batch.id}
    out_df.loc[len(out_df)] = new_row
    out_df.to_csv(args.outcsv)

    print(f"Submitted batch {batch.id}")

