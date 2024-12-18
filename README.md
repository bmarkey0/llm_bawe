# llm_bawe
Code and parquets for using LLMs to assess BAWE texts. Batch requests take up tp 24 hours. After that, regardless of status, batch requested are killed by OpenAI. 

To batch requests to OpenAI:

1) Run gpt_batch_4o-mini.py through a shell script containing the defined arguments. Define outcsv variable as day-specific filepath. Every submitted batch that day will then have it's ID written to the csv.
2) Once all batches for a day are submitted, check their status by running check_batch_status.py that passes the day-specific csv as an argument. For batches that are completed, their file_ids will be written to the csv.
3) Once all batches are completed and all file_ids are in the csv, run download_batch.py. For each file_id, this script will download a parquet file of OpenAIs responses.
4) Using a folder of parquets, clean_openai.py will iterate through each file, prompting the user in the terminal to specify the discipline, genre, rubric, and prompt condition. User can cross check file name printed in terminal to file_id on csv to obtain responses. Script will then print the final line of the dataframe for user to double check information before it is written to the master parquet. 

To set up and run Llama on a cluster:
1) Create a conda environment on the headnode. Install the required pacakges. Exit the conda environment.
2) On the head node, create six directories: cond1, cond2...cond6.
3) Copy all data, python files, and shell scripts from your local space to the headnode using scp.
4) Run the shell script using sbatch <your_shell_script>.sh
5) Each job produces 54 parquets, which will be spread out across the six "cond" directories on the headnode. 
