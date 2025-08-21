from datasets import load_dataset

CACHE_DIR = 'path/to/your/cache/dir'

fleurs = load_dataset('google/fleurs', 'all', cache_dir=CACHE_DIR, trust_remote_code=True)