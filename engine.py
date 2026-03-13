import os
import requests
import numpy as np
import py7zr
from gensim.models import KeyedVectors

MODEL_URL = "https://huggingface.co/BlackKakapo/word-embeddings-ro/resolve/main/SG_pruned_PCA_50k.7z"
ARCHIVE_PATH = "model_ro.7z"
MODEL_PATH = "SG_pruned_PCA.model"

def download_model():
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0:
        if not os.path.exists(ARCHIVE_PATH) or os.path.getsize(ARCHIVE_PATH) == 0:
            try:
                response = requests.get(MODEL_URL, stream=True, timeout=60)
                response.raise_for_status()
                with open(ARCHIVE_PATH, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
            except Exception as e:
                if os.path.exists(ARCHIVE_PATH):
                    os.remove(ARCHIVE_PATH)
                raise e
        
        try:
            with py7zr.SevenZipFile(ARCHIVE_PATH, mode='r') as z:
                z.extractall(path=".")
        except Exception as e:
            if os.path.exists(MODEL_PATH):
                os.remove(MODEL_PATH)
            raise e

class ContextoEngine:
    def __init__(self):
        download_model()
        try:
            self.model = KeyedVectors.load(MODEL_PATH)
            if hasattr(self.model, 'wv'):
                self.model = self.model.wv
        except Exception:
            self.model = KeyedVectors.load_word2vec_format(MODEL_PATH, binary=False)

    def get_similarity(self, word1, word2):
        try:
            return self.model.similarity(word1.lower(), word2.lower())
        except KeyError:
            return 0.0

    def get_sorted_vocab_ranks(self, secret):
        if secret not in self.model:
            return []
        similarities = self.model.most_similar(secret, topn=len(self.model))
        return similarities

def get_daily_word():
    import random
    common_words = [
        "casă", "masă", "scaun", "copac", "mare", "soare", "luna", "om", "femeie", "copil", 
        "școală", "prieten", "carte", "muncă", "zi", "noapte", "cer", "pământ", "apă", "foc", 
        "aer", "bucurie", "tristețe", "iubire", "speranță", "vis", "realitate", "oraș", "sat", 
        "drum", "mașină", "tren", "avion", "munte", "deal", "vale", "râu", "lac", "pădure", 
        "animal", "câine", "pisică", "cal", "pasăre", "pește", "floare", "fruct", "legumă", 
        "pâine", "lapte", "brânză", "carne", "vin", "bere", "cafea", "ceai", "zahăr", "sare", 
        "piper", "gând", "cuvânt", "adevăr", "minciună", "cale", "scop", "viață", "moarte", 
        "vreme", "timp", "ceas", "vânt", "ploaie", "zăpadă", "gheață", "umbrelă", "haine", 
        "pantofi", "geantă", "bani", "muzică", "film", "joc", "familie", "mama", "tata", 
        "frate", "soră", "doctor", "polițist", "istorie", "pace", "război", "lume", "univers"
    ]
    return random.choice(common_words)
