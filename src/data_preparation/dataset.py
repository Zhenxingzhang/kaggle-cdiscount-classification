import pandas as pd
import numpy as np
from sklearn import preprocessing
from src.common import paths


def one_hot_label_encoder(csv_path = paths.CATEGORIES):
    train_Y_orig = pd.read_csv(csv_path, dtype={'category_id': np.str})
    lb = preprocessing.LabelBinarizer()
    lb.fit(train_Y_orig['category_id'])

    def encode(labels):
        return np.asarray(lb.transform(labels), dtype=np.float32)

    def decode(one_hots):
        return np.asarray(lb.inverse_transform(one_hots), dtype=np.str)

    return encode, decode


if __name__ == '__main__':
    one_hot_encoder, _ = one_hot_label_encoder("data/category_names.csv")
    label_vector = one_hot_encoder(["1000012712"]).reshape(-1).tolist()
    print(label_vector.index(max(label_vector)))

