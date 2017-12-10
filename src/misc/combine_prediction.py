import pandas as pd
import csv
import argparse
import os.path

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Default argument')
    parser.add_argument('-i', dest="input_csv", type=str, required=True, help='csv file')
    parser.add_argument('-o', dest="output_csv", type=str, required=True, help='csv')
    args = parser.parse_args()

    # df_all_predictions = pd.DataFrame(columns=['_id', 'category_id', 'prob'])
    #
    # prediction_files = ['/data/outputs/LR_predict.csv', '/data/outputs/LR_predict.csv']
    #
    # for csv_file in prediction_files:
    #     df = pd.read_csv(csv_file)
    #
    #     df_all_predictions = df_all_predictions.append(df)
    #
    # idx = df_all_predictions.groupby(['_id'])['prob'].transform(max) == df_all_predictions['prob']
    #
    # df_all_predictions = df_all_predictions[idx]

    # for idx, row in df_all_predictions.iterrows():
    #     print [row._id, row.category_id, row.prob]

    prediction_df = pd.read_csv(args.input_csv, dtype=object)

    idx = prediction_df.groupby(['_id'])['prob'].transform(max) == prediction_df['prob']

    prediction_df = prediction_df[idx]
    prediction_df.drop_duplicates('_id', inplace=True)

    output_file = args.output_csv
    print("write final results to : {}".format(output_file))

    if not os.path.exists(output_file):
        with open(output_file, 'w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(["_id", "category_id"])

    with open(output_file, 'a') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for index, row in prediction_df.iterrows():
            csv_writer.writerow([row._id, row.category_id])
