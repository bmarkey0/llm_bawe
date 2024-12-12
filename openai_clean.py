import pandas as pd
import os
import argparse
import re
import numpy as np

def clean_parquet(infile, outfile):
    in_df = pd.read_parquet(infile)
    out_df = pd.read_parquet(outfile)

    discipline = input("what is the discipline?")
    genre = input("what is the genre?")
    rubric = input("what is the rubric")
    condition = input("what is the rubric condition?")

    for i in in_df.index:

        proto_doc_id = in_df.loc[i, "doc_id"]
        index = str.find(proto_doc_id, "_")
        doc_id = proto_doc_id[:index]

        proto_level = re.findall("_\d\.", in_df.loc[i, "doc_id"])
        level = proto_level[0][1]

        # temperature = args.temperature
        temperature_list = re.findall("\d\.\d", in_df.loc[i, "doc_id"])
        temperature = float("".join(temperature_list))

        grade = in_df.loc[i, args.filename]

        first = 0
        first_logprob = 0

        second = 0
        second_logprob = 0

        third = 0
        third_logprob = 0

        for count in range(len(in_df.loc[i,"logprobs"]["content"])):
            array = in_df.loc[i, "logprobs"]["content"][count]["top_logprobs"][0]["bytes"]
            if np.all([49 <= array, array <= 53]):
                first = in_df.loc[i, "logprobs"]["content"][count]["top_logprobs"][0]["token"]
                first_logprob = in_df.loc[i, "logprobs"]["content"][count]["top_logprobs"][0]["logprob"]

                second = in_df.loc[i, "logprobs"]["content"][count]["top_logprobs"][1]["token"]
                second_logprob = in_df.loc[i, "logprobs"]["content"][count]["top_logprobs"][1]["logprob"]

                third = in_df.loc[i, "logprobs"]["content"][count]["top_logprobs"][2]["token"]
                third_logprob = in_df.loc[i, "logprobs"]["content"][count]["top_logprobs"][2]["logprob"]
                break
        row = (doc_id, grade, first, first_logprob, second, second_logprob, third, third_logprob, discipline, genre,
               level, condition, temperature, rubric)

        out_df.loc[len(out_df)] = row

    print(out_df.iloc[-1])
    x = input("write to master parquet?")
    if x == "y":
        out_df.to_parquet(outfile)

    print("done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Organize data from OpenAI and add to parquet")
    parser.add_argument("indir", help="path to parquet directory from retrieve_batch.py")
    parser.add_argument("outfile", help="path to master parquet")
    parser.add_argument("--filename", help="name of parquet file")
    # parser.add_argument("--temperature", type=float, help="temperature of model")



    args = parser.parse_args()
    directory = args.indir

    for file in os.listdir(directory):
        filename = os.path.join(directory, file)
        if filename.endswith(".parquet"):
            print(filename)
            clean_parquet(filename, args.outfile)



















