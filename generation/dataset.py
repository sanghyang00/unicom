import os, torch, torchaudio
import pandas as pd
from torch.utils.data import Dataset

class CSDataset(Dataset):
    def __init__(self, csv_dir, mode='train'):
        self.data = pd.read_csv(os.path.join(csv_dir, f'{mode}.csv'))
        # data의 language컬럼 값 별 인덱스로 분할
        # 각 인덱스별 모든 조합 수행
        # 해당 조합을 self.combinations에 할당
        self.combinations = None
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        i1, i2 = self.combinations[idx]
        pass