import pandas as pd
import json
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-input_file")
    parser.add_argument("-output_file")
    parser.add_argument("-shape_name")
    args = parser.parse_args() 

    df = pd.read_csv(args.input_file)
    shape_data = df['shape'].fillna('')
    shape_selection = df.loc[shape_data == args.shape_name]

    with open(args.output_file, 'w', encoding='utf-8') as f:
        data = {}
        csv_attrs = ['datetime', 'duration (seconds)','country','city','shape','comments']
        json_attrs = ['datetime', 'duration','country','city','shape','comments']
        data = {}
        for csv_attr, json_attr in zip(csv_attrs, json_attrs):
            data[json_attr] = shape_selection[csv_attr].tolist()

        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


