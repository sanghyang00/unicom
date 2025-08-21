import os, re, json, yaml
from argparse import ArgumentParser
from tqdm import tqdm
from collections import OrderedDict

def remove_before_first_brace(text):
    first_brace_idx = text.index('{')
    return text[first_brace_idx:]

def remove_comments(text):
    result = re.sub(r"//.*?\n", "\n", text)
    return result

def validate_dict(data, strict=False):
    if not isinstance(data, dict):
        return False
    
    if 'matches' not in data:
        return False
    
    if not isinstance(data['matches'], dict):
        return False
    
    allowed_keys = {'noun', 'verb', 'adverb', 'adjective', 'interjection'}
    
    keys = data['matches'].keys()
    if not strict:
        for key in keys:
            if key not in allowed_keys:
                return False
    else:
        if keys != allowed_keys:
            return False
    
    return True

def initialize_missing_keys(data):
    required_keys = {'noun', 'verb', 'adverb', 'adjective', 'interjection'}
    
    assert 'matches' in data
    for key in required_keys:
        if key not in data['matches']:
            data['matches'][key] = []

    sorted_matches = {}
    for key in required_keys:
        sorted_matches[key] = data['matches'].pop(key)

    data['matches'] = sorted_matches
    
    return data

def dict_to_list(data):
    data = [item for pair in data.items() for item in pair]
    
    return data

def string_to_list(data):
    data = data.strip('()')
    data = data.split(',')
    data = [item.strip() for item in data]

    return data

def cast_elements(data):
    for key, value in data['matches'].items():
        if isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, dict):
                    data['matches'][key][i] = dict_to_list(item)
                elif isinstance(item, str):
                    data['matches'][key][i] = string_to_list(item)
                    
    return data

def prune_invalid_pair(data):
    for key, value in data['matches'].items():
        if isinstance(value, list):
            data['matches'][key] = [item for item in value \
                if isinstance(item, list) and len(item) == 2 and all(isinstance(element, str) for element in item)]  

    return data

def check_empty_wordmap(data):
    for key, value in data['matches'].items():
        if not (isinstance(value, list) and len(value) == 0):
            return False
        
    return True
    
def save_dict_to_yaml(data, filename):
    with open(filename, 'w') as file:
        yaml.dump(data, file, default_flow_style=False, allow_unicode=True)

def main():
    data_cnt, err_cnt, load_err, initialize_err, cast_err, prune_err, empty_err, save_err, match_err = \
        0, 0, 0, 0, 0, 0, 0, 0, 0
    parser = ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='gpt_outputs')
    parser.add_argument('--split', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='wordmap')
    
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    pattern = r'```(.*?)```'
    fnames = sorted(os.listdir(os.path.join(args.output_dir, args.split)))
    for f in tqdm(fnames):
        fpath = os.path.join(args.output_dir, args.split, f)
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                data_cnt += 1
                gpt_response = json.loads(line.strip())
                id_ = gpt_response['custom_id']
                content = gpt_response['response']['body']['choices'][0]['message']['content']
                
                if content:
                    match = re.search(pattern, content, re.DOTALL)
                    
                    if match:
                        data = match.group(1).strip()
                        data = remove_comments(data)
                        
                        try:
                            data_dict = yaml.safe_load(data[4:])
                            assert validate_dict(data_dict, strict=False)
                            
                        except:
                            err_cnt += 1
                            load_err += 1
                            continue
                            
                        try:    
                            data_dict = initialize_missing_keys(data_dict)
                            assert validate_dict(data_dict, strict=True)
                        
                        except:
                            err_cnt += 1
                            initialize_err += 1
                            continue
                        
                        try:
                            data_dict = cast_elements(data_dict)
                        
                        except:
                            err_cnt += 1
                            cast_err += 1
                            continue
                        
                        try:
                            data_dict_final = prune_invalid_pair(data_dict)
                            
                        except:
                            err_cnt += 1
                            prune_err += 1
                            continue
                            
                        if check_empty_wordmap(data_dict_final):
                            err_cnt += 1
                            empty_err += 1
                            continue
                        
                        try:
                            savepath = os.path.join(args.save_dir, f'{id_}.yaml')
                            
                            if os.path.exists(savepath):
                                print(f"Warning: File already exists at {savepath}")
                                
                            save_dict_to_yaml(data_dict_final, savepath)
                            
                        except:
                            err_cnt += 1
                            save_err += 1
                            continue
                            
                    else:
                        err_cnt += 1
                        match_err += 1
                            
    print(f'Total data count: {data_cnt}')
    load_err, initialize_err, cast_err, prune_err, save_err, match_err
    print(f'Total error count: {err_cnt}')        
    print(f'Error log - Load: {load_err}, Init: {initialize_err}, Cast: {cast_err}, Prune: {prune_err}, Empty: {empty_err}, Save: {save_err}, Match: {match_err}')

if __name__=='__main__':
    main()