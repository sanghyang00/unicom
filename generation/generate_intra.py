import os, re, json, yaml, itertools, scipy, librosa, torch, torchaudio, random
import numpy as np
import pandas as pd
import scipy.io.wavfile as wav
import uroman as ur
from typing import List
from torchaudio.pipelines import MMS_FA as bundle
from argparse import ArgumentParser
from tqdm import tqdm
from jiwer import cer
from utils import *

# CNT = 0
SAMPLE_RATE = 16000
VAD_TRIGGER_LEVEL = 2e-3
SEARCH_TIME = 0.5
VAD = torchaudio.transforms.Vad(sample_rate=SAMPLE_RATE, trigger_level=VAD_TRIGGER_LEVEL, search_time=SEARCH_TIME)

METADATA = pd.DataFrame(columns=['file path', 'major language', 'word pair', 'anchor', 'transcription', 'source path 1', 
                                 'source id 1', 'source language 1', 'source transcription 1', 'source roman 1',
                                 'source path 2', 'source id 2', 'source language 2', 'source transcription 2', 'source roman 2'])
METADATA_IDX = 0

VOXPOPULI_LANGS = np.array(['Bulgarian', # WF
                   'Czech', # WF
                   'Croatian', # WF
                   'Danish', # WF
                   'Dutch', # WF
                   'English', # WF
                   'Estonian', # WF
                   'Finnish', # WF
                   'French', # WF
                   'German', # WF
                   'Greek', # WF
                   'Hungarian', # WF
                   'Italian', # WF
                   'Latvian', # WF
                   'Lithuanian', # WF
                   'Maltese', # WF
                   'Polish', # WF
                   'Portuguese', # WF
                   'Romanian', # WF
                   'Slovak', # WF
                   'Slovenian', # WF
                   'Spanish', # WF
                   'Swedish' # WF
                   ])

DEVICE = 'cuda'
MMS_FA = bundle.get_model()
MMS_FA.to(DEVICE)
ROMANIZER = ur.Uroman()
TOKENIZER = bundle.get_tokenizer()
ALIGNER = bundle.get_aligner()
LANG_TO_ISO = load_dictionary('fleurs_to_iso3.json')

def normalize_uroman(text):
    text = text.lower()
    text = text.replace("’", "'")
    text = re.sub("([^a-z' ])", " ", text)
    text = re.sub(' +', ' ', text)
    return text.strip()

def normalize_audio(y):
    return y / torch.max(torch.abs(y))

def perform_bpf(y, sr=SAMPLE_RATE, cutoff_l=80, cutoff_h=7000):
    y = torchaudio.functional.highpass_biquad(y, sample_rate=sr, cutoff_freq=cutoff_l)
    y = torchaudio.functional.lowpass_biquad(y, sample_rate=sr, cutoff_freq=cutoff_h)
    
    return y

def preprocess_audio(y, sr=SAMPLE_RATE, cutoff_l=80, cutoff_h=7000):
    y = normalize_audio(y)
    y = perform_bpf(y, sr=sr, cutoff_l=cutoff_l, cutoff_h=cutoff_h) 
    
    return y

def search_longest_sample_path(df):
    idx = df['duration'].idxmax()
    return df.iloc[idx]['file path']

def apply_trim(y):
    y = VAD(y)
    y = torch.flip(y, (-1,))
    y = VAD(y)
    y = torch.flip(y, (-1,))
    
    return y

def search_closest_word(transcription, query):
    
    window_size = len(query.split())
    tokens = transcription.split()
    
    if window_size > 1:
        candidates = [tokens[i:i + window_size] for i in range(len(tokens) - window_size + 1)]
        candidates = [' '.join(c) for c in candidates]
    
    elif window_size == 1:
        candidates = tokens
    
    else:
        print(f'Invalid query: {query}, {query.split()}')
        raise ValueError()
    
    cers = np.array([cer(c, query) for c in candidates])
    min_idx = np.argmin(cers)
    
    closest_word = candidates[min_idx]
    
    return closest_word

def select_wordpair(wordmap, pairs=1):
    body = wordmap['matches']
    
    # pos_list = ['noun', 'verb', 'adjective', 'adverb', 'interjection']
    pos_list = ['noun', 'verb', 'interjection']
    pos_cnt = len(pos_list)
    sampling_pool = []
    for pos in pos_list:
        sorted_pool = [p for p in body[pos] if not (len(p[0]) < 3 or len(p[1]) < 3)]
        sampling_pool += sorted_pool
        
    pairs = min(pairs, len(sampling_pool))
    result = random.sample(sampling_pool, pairs)
    
    return result

def substitute_word_(wordpairs, trans1, trans2, major_lang_idx):
    if major_lang_idx == 0:
        major_trans = trans1
        for wordpair in wordpairs:
            major_trans = major_trans.replace(wordpair[0], wordpair[1], 1)
    
    else:
        major_trans = trans2    
        for wordpair in wordpairs:
            major_trans = major_trans.replace(wordpair[1], wordpair[0], 1)
    
    return major_trans

def substitute_word(wordpairs, trans1, trans2, major_lang_idx):
    if major_lang_idx == 0:
        major_trans = trans1
        for wordpair in wordpairs:
            pattern = r'\b' + re.escape(wordpair[0]) + r'\b'  
            major_trans = re.sub(pattern, wordpair[1], major_trans)
    
    else:
        major_trans = trans2
        for wordpair in wordpairs:
            pattern = r'\b' + re.escape(wordpair[1]) + r'\b'  
            major_trans = re.sub(pattern, wordpair[0], major_trans)
    
    return major_trans
    
def compute_alignments(y: torch.Tensor, transcript: List[str]):
    with torch.inference_mode():
        emission, _ = MMS_FA(y.to(DEVICE))
        token_spans = ALIGNER(emission[0], TOKENIZER(transcript))
    return emission, token_spans

def _score(spans):
    return sum(s.score * len(s) for s in spans) / sum(len(s) for s in spans)

def search_span(y, spans, num_frames, transcript, sample_rate=bundle.sample_rate):
    ratio = y.size(1) / num_frames
    x0 = int(ratio * spans[0].start)
    x1 = int(ratio * spans[-1].end)
    
    return x0, x1

def search_idx(transcript, ref):
    cers = np.array([cer(gt, ref) for gt in transcript])
    idx = np.argmin(cers)
    
    return idx

def compute_timestamp(y, roman, word, lang):
    switch = [False, False]
    if len(word.split()) > 1:
        subwords = word.split()
        timestamps, scores, switches = zip(*[compute_timestamp(y, roman, w, lang) for w in subwords])
        timestamp = (timestamps[0][0], timestamps[-1][-1])
        score = np.mean(scores)
        switch = [any(switch[i] for switch in switches) for i in range(len(switches[0]))]
    
    else: 
        transcript = roman.strip().split()
        tokens = TOKENIZER(transcript)
        emission, token_spans = compute_alignments(y, transcript)
        num_frames = emission.size(1)
        idx = search_idx(transcript, word)
        if idx == 0:
            switch[0] = True
        if idx == len(token_spans) - 1:
            switch[1] = True
        
        start, end = search_span(y, token_spans[idx], num_frames, transcript[idx])
        score = round(_score(token_spans[idx]), 2)
        timestamp = (start, end)
    
    return timestamp, score, switch

def generate_one_intra_sample(id_, y1, lang1, trans1, roman1, y2, lang2, trans2, roman2, num_subs, modulation):
    
    valid = True
    init_switch = False
    end_switch = False
    
    y1 = apply_trim(y1)
    y2 = apply_trim(y2)
    
    if y1.shape[-1] < SAMPLE_RATE or y2.shape[-1] < SAMPLE_RATE:
        
        print('Zero sample after trim!')
        valid = False
        merged_sample, wordpairs, transcription, major_language, anchors = \
            None, None, None, None, None
            
        return merged_sample, wordpairs, transcription, major_language, anchors, valid
    
    with open(os.path.join('wordmap', f'{id_}_{lang1}_{lang2}.yaml'), "r", encoding="utf-8") as f:
        wordmap = yaml.safe_load(f) 
        
    pairs = random.randint(1, num_subs)
    wordpairs = select_wordpair(wordmap, pairs=pairs)
    
    if len(wordpairs) == 0:
        
        valid = False
        merged_sample, wordpairs, transcription, major_language, anchors = \
            None, None, None, None, None
            
        return merged_sample, wordpairs, transcription, major_language, anchors, valid
    
    assert len(wordpairs) != 0
    
    timestamps = []
    switches1, switches2 = [], []
    for i, wordpair in enumerate(wordpairs):
        newpair = []
        
        word1 = search_closest_word(trans1, wordpair[0])
        newpair.append(word1)
        word1 = ROMANIZER.romanize_string(wordpair[0], lang=LANG_TO_ISO[lang1])
        word1 = normalize_uroman(word1)
        
        timestamp1, score1, switch1 = compute_timestamp(y1, roman1, word1, lang1)
        switches1.append(switch1)
        
        word2 = search_closest_word(trans2, wordpair[1])
        newpair.append(word2)
        word2 = ROMANIZER.romanize_string(wordpair[1], lang=LANG_TO_ISO[lang2])
        word2 = normalize_uroman(word2)
        
        timestamp2, score2, switch2 = compute_timestamp(y2, roman2, word2, lang2) 
        switches2.append(switch2)
        timestamps.append({'pair': (newpair[0], newpair[1]), 'timestamp': (timestamp1, timestamp2)})
    
    if random.random() < 0.5:
        major_language = lang1
        major_lang_idx = 0
        major = y1
        substitution_source = y2
        timestamps = sorted(timestamps, key=lambda x: x['timestamp'][0][0])
        timestamps_major = [t['timestamp'][0] for t in timestamps]
        timestamps_substitution = [t['timestamp'][1] for t in timestamps]
        switch = [any(switch[i] for switch in switches1) for i in range(len(switches1[0]))]
        init_switch = switch[0]
        end_switch = switch[1]
    
    else:
        major_language = lang2
        major_lang_idx = 1
        major = y2
        substitution_source = y1
        timestamps = sorted(timestamps, key=lambda x: x['timestamp'][1][0])
        switch = [any(switch[i] for switch in switches2) for i in range(len(switches2[0]))]
        timestamps_major = [t['timestamp'][1] for t in timestamps]
        timestamps_substitution = [t['timestamp'][0] for t in timestamps]
        init_switch = switch[0]
        end_switch = switch[1]
    
    wordpairs = [t['pair'] for t in timestamps]
    init = 0
    final = major.shape[-1] - 1

    segment_timestamps = [[init, timestamps_major[0][0]]] + \
         list(itertools.chain.from_iterable([[timestamps_substitution[i][0], timestamps_substitution[i][1]], \
              [timestamps_major[i][1], timestamps_major[i+1][0]]] for i in range(len(timestamps_substitution) - 1))) + \
              [[timestamps_substitution[-1][0], timestamps_substitution[-1][1]], [timestamps_major[-1][1], final]]
    
    segments = [substitution_source[:, ts[0]:ts[1]] if i % 2 else major[:, ts[0]:ts[1]] for i, ts in enumerate(segment_timestamps)]
    
    if init_switch:
        segments = segments[1:] # or mute padding
    
    if end_switch:
        segments = segments[:-1] # or mute padding
    
    merged_sample = torch.cat(segments, axis=-1)
    anchors = np.cumsum([seg.shape[-1] for seg in segments])
    
    transcription = substitute_word(wordpairs, trans1, trans2, major_lang_idx)
    
    return merged_sample, wordpairs, transcription, major_language, anchors, valid

def generate_intra(data, langs, knn_vc, num_samples=None, num_subs=None, modulation=False, topk=4, savedir=None):
    
    global METADATA
    global METADATA_IDX
    
    lang1 = langs[0]
    lang2 = langs[1]
    
    data_l1 = data[data['language']==lang1].reset_index(drop=True)
    data_l2 = data[data['language']==lang2].reset_index(drop=True)
    
    unique_ids_l1 = data_l1['id'].unique()
    unique_ids_l2 = data_l2['id'].unique()
    
    common_ids = sorted(np.intersect1d(unique_ids_l1, unique_ids_l2))
    
    if num_samples is not None:
        common_ids = random.sample(common_ids, num_samples)
        
    for id_ in tqdm(common_ids, leave=False):
        row_l1 = data_l1[data_l1['id']==id_].sample(1)
        row_l2 = data_l2[data_l2['id']==id_].sample(1)
            
        fid = f'{id_}_{lang1}_{lang2}'
        fname = f'{fid}.yaml'
        if fname not in os.listdir('wordmap'):
            continue
        
        y1, sr1 = torchaudio.load(row_l1['file path'].values[0])
        y2, sr2 = torchaudio.load(row_l2['file path'].values[0])
        if sr1 != SAMPLE_RATE:
            y1 = torchaudio.functional.resample(y1, orig_freq=sr1, new_freq=SAMPLE_RATE)
            sr1 = SAMPLE_RATE
        if sr2 != SAMPLE_RATE:
            y2 = torchaudio.functional.resample(y2, orig_freq=sr2, new_freq=SAMPLE_RATE)
            sr2 = SAMPLE_RATE
        
        assert sr1 == sr2
        sr = sr1
        
        y1 = normalize_audio(y1)
        y2 = normalize_audio(y2)
        
        fpath1 = row_l1['file path'].values[0]
        fpath2 = row_l2['file path'].values[0]
        
        trans1 = row_l1['transcription'].values[0]
        trans2 = row_l2['transcription'].values[0]
        
        roman1 = row_l1['roman'].values[0]
        roman2 = row_l2['roman'].values[0]
        
        intra_sample_diffstyle, wordpair, merged_transcription, major_language, anchors, valid = \
            generate_one_intra_sample(id_, y1, lang1, trans1, roman1, y2, lang2, trans2, roman2, num_subs=num_subs, modulation=modulation)
        
        if not valid:
            continue
        
        if major_language == lang1:
            longest_path = search_longest_sample_path(data_l1)
        else:
            longest_path = search_longest_sample_path(data_l2)
        
        longest_sample, sr_l = torchaudio.load(longest_path)
        if sr_l != SAMPLE_RATE:
            longest_sample = torchaudio.functional.resample(longest_sample, orig_freq=sr_l, new_freq=SAMPLE_RATE)
        
        try:
            query = knn_vc.get_features(intra_sample_diffstyle, vad_trigger_level=0)
            matching_set = knn_vc.get_matching_set([longest_sample], vad_trigger_level=VAD_TRIGGER_LEVEL) # 이거 짧은 utt 기준 긴거기때문에 7.5 미만임. 고려해보기
            
            sample_samestyle = knn_vc.match(query, matching_set, topk=topk)
            sample_samestyle = normalize_audio(sample_samestyle)
            
            savepath = os.path.join(savedir, f'{fid}_({major_language}).wav')
            wav.write(savepath, 16000, sample_samestyle.cpu().numpy())
            
            row = [savepath, major_language, wordpair, anchors, merged_transcription, fpath1, id_, lang1, trans1, roman1, fpath2, id_, lang2, trans2, roman2]
            METADATA.loc[METADATA_IDX] = row
            METADATA_IDX += 1
        
        except:
            print(f'Error occured during conversion: {fid}')
        
def main():
    
    global METADATA
    global METADATA_IDX
    
    parser = ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='path/to/preprocessed/csvs')
    parser.add_argument('--dataset', type=str, default='fleurs-r')
    parser.add_argument('--save_dir', type=str, default='path/to/save')
    parser.add_argument('--voxpopuli', type=bool, default=True)
    parser.add_argument('--split', type=str, default=None)
    parser.add_argument('--cutoff', type=float, default=20)
    parser.add_argument('--max_id_cnt', type=int, default=None)
    parser.add_argument('--max_sub_cnt', type=int, default=3)
    parser.add_argument('--continue_idx', type=int, default=None)
    parser.add_argument('--end_idx', type=int, default=None)
    
    args = parser.parse_args()
    
    if args.max_sub_cnt > 3:
        raise ValueError('Substitution over than three words may cause the invalid result!')
    
    dataset_dir = os.path.join(args.base_dir, args.dataset)
    data = pd.read_csv(os.path.join(dataset_dir, f'{args.split}.csv'))
    if args.voxpopuli:
        data = data[data['language'].isin(VOXPOPULI_LANGS)].reset_index(drop=True)
    data = data[data['duration'] < args.cutoff].reset_index(drop=True)
    
    lang_to_iso = load_dictionary('fleurs_to_iso3.json')
    langs = data['language'].unique()
    lang_combinations = list(itertools.combinations(langs, 2))
    lang_combinations = [tuple(sorted(comb)) for comb in lang_combinations]
    knn_vc = torch.hub.load('bshall/knn-vc', 'knn_vc', prematched=True, trust_repo=True, pretrained=True, device=DEVICE)

    os.makedirs(os.path.join(args.save_dir, 'intra'), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'intra', args.split), exist_ok=True)
    
    lang_combinations = lang_combinations[args.continue_idx:args.end_idx]
    print(f'Starting generation from {lang_combinations[0]} to {lang_combinations[-1]}')
        
    for comb in tqdm(lang_combinations):
        METADATA = pd.DataFrame(columns=['file path', 'major language', 'word pair', 'anchor', 'transcription', 'source path 1', 
                                 'source id 1', 'source language 1', 'source transcription 1', 'source roman 1',
                                 'source path 2', 'source id 2', 'source language 2', 'source transcription 2', 'source roman 2'])
        METADATA_IDX = 0
        savedir = os.path.join(args.save_dir, 'intra', args.split, f'{lang_to_iso[comb[0]]}_{lang_to_iso[comb[1]]}')
        os.makedirs(savedir, exist_ok=True)
        generate_intra(data, comb, knn_vc, num_samples=args.max_id_cnt, num_subs=args.max_sub_cnt, savedir=savedir)
    
        METADATA.to_csv(os.path.join(savedir, 'metadata.csv'), index=False)
    
if __name__=='__main__':
    main()