import nltk
import pandas as pd
from tqdm import tqdm
import numpy as np
import math

class DataReader():
    def __init__(self, dir_path):
        self.dir_path = dir_path

    def readData(self,  names=['document', 'type'],) -> pd.DataFrame:
        test_df = pd.read_csv(self.dir_path,
                              names=names,  encoding='utf-8')
        test_df.fillna('<None>', inplace=True)
        return test_df.iloc[:, 0]

    def dataTokenize(self, dataset: pd.DataFrame) -> np.ndarray:
        dataset = dataset.to_numpy()
        res = []
        for i, items in enumerate(tqdm(dataset)):
            sents = nltk.sent_tokenize(items)
            for sent in sents:
                token_sent = nltk.word_tokenize(sent)
                res.append(token_sent)
        return np.array(res)

    def __call__(self, *args, **kwargs):
        return self.dataTokenize(self.readData())


datareader = DataReader("../dataset/Movies/train.csv")
dataset_tokenized_np = datareader()
np.save("../dataset/Movies/train.npy", dataset_tokenized_np)
print(dataset_tokenized_np[:10])
