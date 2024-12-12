import argparse
import json
import os.path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from openai import OpenAI

def extract_text(jsonl):
    """Load a JSONL line and return the docid, chunk tuple."""
    row = json.loads(jsonl)

    return (row["custom_id"],
            row["response"]["body"]["choices"][0]["message"]["content"])

def extract_logprobs(jsonl):
    """Load JSONL line and return logprobs"""
    row = json.loads(jsonl)

    return row["response"]["body"]["choices"][0]["logprobs"]

parser = argparse.ArgumentParser(description="Download OpenAI batch file and write Parquet")
parser.add_argument("indir", help="Directory to CSV with File ids")
parser.add_argument("outdir", help="Directory to save output in. Filename will be the model used.")

args = parser.parse_args()

client = OpenAI()

df = pd.read_csv(args.indir)
for i in df.index:

    if df.loc[i, "parquet_created?"] == "yes":
        continue

    file_id = df.loc[i, "file_id"]
    batch_id = df.loc[i, "batch_id"]

    if type(file_id) != str:
        print(f"No file for {batch_id}")
        continue

    output = client.files.content(file_id)

    out_lines = output.text.split("\n")
    num_out = len(out_lines)

    model = json.loads(out_lines[0])["response"]["body"]["model"]

    docs = [extract_text(line)
            for line in out_lines
            if line != ""]

    logprobs = [extract_logprobs(line)
                for line in out_lines
                if line != ""]

    out_df = pd.DataFrame(
        {
            "doc_id": [doc_id for doc_id, _ in docs],
            model: [content for _, content in docs],
            "logprobs": logprobs
    }
)

    pq.write_table(pa.Table.from_pandas(out_df), os.path.join(args.outdir, file_id + ".parquet"),
                   compression="gzip")

    df.loc[i, "parquet_created?"] = "yes"
    df.to_csv(args.indir)

    print(f"{file_id} to parquet done")