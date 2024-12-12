import argparse

import pandas as pd
from openai import OpenAI

parser = argparse.ArgumentParser(description="Check OpenAI API batch status")
parser.add_argument("indir", help="Directory to csv with batch ids")

args = parser.parse_args()

client = OpenAI()

df = pd.read_csv(args.indir)
for i in df.index:
    batch_id = df.loc[i, "batch_id"]
    batch = client.batches.retrieve(batch_id)

    print(f"Batch {batch.id}:")
    print(f"Status: {batch.status}")
    print(f"Errors: {batch.errors}")

    if batch.status == "in_progress":
        print(f"Completed {batch.request_counts.completed} of {batch.request_counts.total} prompts")

    if batch.output_file_id is not None:
        df.loc[i, "file_id"] = batch.output_file_id
        print(f"Output file ID: {batch.output_file_id}")

df.to_csv(args.indir)

