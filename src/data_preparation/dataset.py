import pandas as pd
import numpy as np
from sklearn import preprocessing
from src.common import paths


def one_hot_label_encoder(csv_path=paths.CATEGORIES):
    category_labels = pd.read_csv(csv_path, dtype={'category_id': np.str})
    lb = preprocessing.LabelBinarizer()
    lb.fit(category_labels['category_id'])

    def find_max_idx(lb_vec):
        lb_vector = lb_vec.reshape(-1).tolist()
        return lb_vector.index(max(lb_vector))

    def encode(lbs_str):
        lbs_vector = np.asarray(lb.transform(lbs_str), dtype=np.float32)
        return np.apply_along_axis(find_max_idx, 1, lbs_vector)

    def decode(one_hots):
        return np.asarray(lb.inverse_transform(one_hots), dtype=np.str)

    return encode, decode


if __name__ == '__main__':
    one_hot_encoder, _ = one_hot_label_encoder("data/category_names.csv")
    lb_idx = one_hot_encoder(["1000012764"])

    print(lb_idx)

