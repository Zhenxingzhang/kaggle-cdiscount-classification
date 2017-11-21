import bson
import random
from tqdm import tqdm
import argparse


def convert_bson_2_record(input_bson_filename, output_bson_filename,  n=100, number_random_example=10):

    data = bson.decode_file_iter(open(input_bson_filename, 'rb'))

    print(n, number_random_example)
    random_items = random.sample(range(n), number_random_example)
    random_items.sort()

    idx = 0
    r_idx = 0

    for c, d in tqdm(enumerate(data), total=n):
        print(idx, random_items[r_idx])
        if idx != random_items[r_idx]:
            idx = idx + 1
            continue
        else:
            # for c, d in enumerate(data):
            idx = idx + 1
            r_idx = r_idx + 1

            if r_idx >= number_random_example:
                    break

    print("Finish convert tfrecords with {} records".format(r_idx))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-i', dest="full_bson_filename", type=str, required=True, help='the input file in bson format')
    parser.add_argument('-o', dest="subset_bson_filename", type=str, required=True, help='the output file in bson format')
    parser.add_argument('-n', dest="total_records", type=int, required=True, help='number of records to convert.')
    parser.add_argument('-r', dest="number_of_random_records", type=int, required=True, help='number of random records to convert.')
    args = parser.parse_args()

    convert_bson_2_record(args.full_bson_filename, args.subset_bson_filename,
                          n=args.total_records, number_random_example=args.number_of_random_records)
