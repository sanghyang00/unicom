import os, json, itertools
import pandas as pd
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser

VOXPOPULI_LANGS = ['Bulgarian',
                   'Czech',
                   'Croatian',
                   'Danish',
                   'Dutch',
                   'English',
                   'Estonian',
                   'Finnish',
                   'French',
                   'German',
                   'Greek',
                   'Hungarian',
                   'Italian',
                   'Latvian',
                   'Lithuanian',
                   'Maltese',
                   'Polish',
                   'Portuguese',
                   'Romanian',
                   'Slovak',
                   'Slovenian',
                   'Spanish',
                   'Swedish'
                   ]

LINES = []

def chunk_list(lst, chunk_size):
    full_list = []
    for i in range(0, len(lst), chunk_size):
        sub_list = lst[i:i + chunk_size]
        full_list.append(sub_list)
        
    return full_list

def fit_to_instruction(lang1, lang2):
    template = \
        f"You are a language expert specializing in {lang1} and {lang2}."

    return template

def fit_to_restriction():
    architecture = \
        """
        matches:
            noun:
                -
            verb:
                -
            adverb:
                -
            adjective:
                -
            interjection:
                -    
        """

    template =\
        f"The final outputs must be returned in YAML format, and each component in part of speech is a tuple of words with the same meaning. The YAML file structure must strictly adhere to the following format: {architecture}"
    
    return template
    
def fit_to_template(lang1, trans1, lang2, trans2):
    
    template = \
        f"Find pairs of words with the same meaning and sort it with the part of speech information in the given two sentences from different languages.\n{lang1} sentence: {trans1}\n{lang2} sentence: {trans2}"
    
    return template

def generate_prompt(rid, lang1, trans1, lang2, trans2):
    
    line = {"custom_id": f"{rid}", 
            "method": "POST", 
            "url": "/v1/chat/completions", 
            "body": {"model": "gpt-4o-mini", 
                     "messages": [
                         {"role": "system",
                          "content": fit_to_instruction(lang1, lang2)
                          },
                         {"role": "system",
                          "content": fit_to_restriction()
                          },
                         {"role": "user", 
                          "content": fit_to_template(lang1, trans1, lang2, trans2)}
                         ],
                     "temperature": 0.0
                     }
            }
    
    return line

def generate_pairs(data, langs, num_sample=None):
    lang1 = langs[0]
    lang2 = langs[1]
    
    data_l1 = data[data['language']==lang1].reset_index(drop=True)
    data_l2 = data[data['language']==lang2].reset_index(drop=True)
    
    unique_ids_l1 = data_l1['id'].unique()
    unique_ids_l2 = data_l2['id'].unique()
    
    common_ids = np.intersect1d(unique_ids_l1, unique_ids_l2)
    
    interm_lines = []
    for id_ in tqdm(common_ids, leave=False):
        indices1 = data_l1[data_l1['id']==id_].index.to_numpy()
        indices2 = data_l2[data_l2['id']==id_].index.to_numpy()
        
        trans1 = str(data_l1.iloc[indices1[0]]['transcription'])
        trans2 = str(data_l2.iloc[indices2[0]]['transcription'])
        
        rid = f'{id_}_{lang1}_{lang2}'
        line = generate_prompt(rid, lang1, trans1, lang2, trans2)
        interm_lines.append(line)
        
    return interm_lines
        
def main():
    
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='path/to/preprocessed/csvs')
    parser.add_argument('--split', type=str, default=None)
    parser.add_argument('--threshold', type=int, default=85)
    parser.add_argument('--voxpopuli', type=bool, default=True)
    parser.add_argument('--save_dir', type=str, default='gpt_inputs')
    parser.add_argument('--samples_per_batch', type=int, default=10000)
    
    args = parser.parse_args()
    
    data_path = os.path.join(args.data_dir, f'{args.split}.csv')
    data = pd.read_csv(data_path)
    if args.voxpopuli:
        langs = np.intersect1d(VOXPOPULI_LANGS, data['language'].unique())
    else:
        langs = data['language'].unique()
    
    combinations = list(itertools.combinations(langs, 2)) # Total 253 combinations
    grouped_ids = data.groupby('id')['language'].unique()
    samples_per_id = grouped_ids.apply(len)
    selected_ids = samples_per_id[samples_per_id > 85].index.to_numpy()
    
    data = data[data['language'].isin(langs)].reset_index(drop=True)
    data = data[data['id'].isin(selected_ids)].reset_index(drop=True)
    
    for comb in tqdm(combinations):
        
        interm_lines = generate_pairs(data, comb)
        LINES.extend(interm_lines)
    
    save_dir = os.path.join(args.save_dir, args.split)
    os.makedirs(save_dir, exist_ok=True)
    print(f'Line count: {len(LINES)}')
    if len(LINES) > args.samples_per_batch:
        print(f'Line counts of the jsonl file exceeds maximum batch size (Actual count: {len(LINES)})')
        
        chunked_lines = list(chunk_list(LINES, args.samples_per_batch))
        
        for idx, sublist in enumerate(chunked_lines):
            save_path = os.path.join(save_dir, f'intra_sentential_cs_part_{str(idx + 1).zfill(2)}.jsonl')
            with open(save_path, "w", encoding="utf-8") as f:
                for l in sublist:
                    f.write(json.dumps(l) + "\n")
            print(f"Saved chunk {idx + 1} to {save_path}")
    
    else:
        save_path = os.path.join(save_dir, 'intra_sentential_cs.jsonl')
        with open(save_path, "w", encoding="utf-8") as f:
            for l in LINES:
                f.write(json.dumps(l) + "\n")
        print(f"Saved file to {save_path}")
    
if __name__=='__main__':
    main()
        
        