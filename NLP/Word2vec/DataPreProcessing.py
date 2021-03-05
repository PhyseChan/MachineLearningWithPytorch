import nltk
import pandas as pd
from tqdm import tqdm
import numpy as np
import math

class DataReader():
    def __init__(self, dir_path):
        self.dir_path = dir_path

    def readData(self, skiprows=1, index_col=0, names=['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate'],
                 low_memory=False) -> pd.DataFrame:
        test_df = pd.read_csv(self.dir_path, skiprows=skiprows, index_col=index_col,
                              names=names, low_memory=low_memory, encoding='utf-8')
        test_df.fillna('<None>', inplace=True)
        return test_df.iloc[:, [2, 3]]

    def dataTokenize(self, dataset: pd.DataFrame) -> np.ndarray:
        dataset = dataset.to_numpy()
        res = []
        for i, items in enumerate(tqdm(dataset)):
            row_data = []
            for j, item in enumerate(items):
                for sent_token in nltk.sent_tokenize(item):
                    word_token = nltk.word_tokenize(sent_token)
                row_data.append(word_token)
            res.append(row_data)
        return np.array(res)

    def __call__(self, *args, **kwargs):
        return self.dataTokenize(self.readData())


datareader = DataReader("../dataset/quora_pair/train.csv")
dataset_tokenized_np = datareader()
np.save("../dataset/quora_pair/train.npy", dataset_tokenized_np)
print(dataset_tokenized_np[:10])
