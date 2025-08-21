import os, sys, re, time
from argparse import ArgumentParser
from openai import OpenAI

FAILED = False
MAXIMUM_REQUEST = 50000
TEMPERATURE=0.0
os.environ['OPENAI_API_KEY'] = 'your gpt api key'

def check_status(batch):
    return batch

def main():
    
    parser = ArgumentParser()
    parser.add_argument('--jsonl_dir', type=str, default='gpt_inputs')
    parser.add_argument('--split', type=str, default=None)
    parser.add_argument('--process_numbers', type=int, nargs='+', default=[], help='List of numbers to process')
    
    args = parser.parse_args()
    
    client = OpenAI()
    
    files = [
        f for f in os.listdir(os.path.join(args.jsonl_dir, args.split))
        if re.search(r"part_(\d+)\.jsonl", f) and int(re.search(r"part_(\d+)\.jsonl", f).group(1)) in args.process_numbers
    ]
    files = sorted(files)
    
    print('Collected files:')
    for fname in files:
        print(fname)
    
    for fname in files:
        
        batch_input_file = client.files.create(
            file=open(os.path.join(args.jsonl_dir, args.split, fname), "rb"),
            purpose="batch"
        )

        batch_input_file_id = batch_input_file.id
        batch = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )

        print(f'Metadata of the created batch:\n{batch}')
        
        time.sleep(30)
        
        while True:
            current_status = client.batches.retrieve(batch.id).status
            print(f'Batch ID: {batch.id}, Current Status: {current_status}')
            
            if current_status == "failed":
                print(f"Error: Batch ID {batch.id} failed. Exiting the program.")
                sys.exit(1)
            
            elif current_status == "expired":
                print(f"Batch ID {batch.id} has expired. Exiting the program.")
                sys.exit(1) 
            
            elif current_status == "completed":
                print(f"Batch ID {batch.id} completed successfully.")
                break 
            
            elif current_status == "in_progress":
                print(f"Batch ID {batch.id} is in progress.")
            
            time.sleep(600)
    
if __name__=='__main__':
    main()