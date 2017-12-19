import argparse
import bson
from tqdm import tqdm
import random
from bson import BSON
from random import shuffle


def random_keep_n_product(prod_list, r_size):
    if len(prod_list) > r_size:
        return [prod_list[i] for i in sorted(random.sample(xrange(len(prod_list)), r_size))]
    else:
        return prod_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-i', dest="full_bson_filename", type=str, required=True, help='the input file in bson format')
    parser.add_argument('-o', dest="subset_bson_filename", type=str, required=True, help='the output file in bson format')
    parser.add_argument('-n', dest="total_records", type=int, required=True, help='number of records to convert.')
    args = parser.parse_args()

    input_bson_filename = args.full_bson_filename
    output_bson_filename = args.subset_bson_filename
    n = args.total_records

    data = bson.decode_file_iter(open(input_bson_filename, 'rb'))
    categories_idx = dict()

    for c, d in tqdm(enumerate(data), total=n):
        category_id = d['category_id']
        if category_id in categories_idx:
            categories_idx[category_id].append(c)
        else:
            categories_idx[category_id] = list([c])

    sample_idxs = []

    for key, value in categories_idx.iteritems():
        sample_list = random_keep_n_product(value, 100)
        sample_idxs.extend(sample_list)

    # for item in sample_idxs:
    #     print(item)
    sample_idxs.sort()

    samples = []
    r_idx = 0
    # print(sample_idxs[36])
    data1 = bson.decode_file_iter(open(input_bson_filename, 'rb'))
    for c, d in tqdm(enumerate(data1), total=n):
        if c != sample_idxs[r_idx]:
            continue
        else:
            samples.append(d)
            r_idx = r_idx + 1
            if r_idx == len(sample_idxs):
                break

    print("random select {} training samples".format(len(samples)))

    with open(output_bson_filename, 'w') as output:
        shuffle(samples)

        for item in samples:
            output.write(BSON.encode(item))

        print("Finished balance data.")
