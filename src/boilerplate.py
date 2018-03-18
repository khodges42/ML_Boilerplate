import pandas as pd
import argparse, json, os
from algorithms.library.awesome_algorithm import Algorithm_Algorithm

def test(args, df, config):
    pass

def predict(args, df, config):
    pass

def train(args, df, config):
    pass

def load_data(args):
    print("Loading input data...")
    input_name,input_extension = os.path.splitext(args.input)
    with open(args.input, 'rb') as f:
        if input_extension.upper() == ".JSON": df = pd.read_json(args.input)
        if input_extension.upper() == ".CSV": df = pd.read_csv(args.input, engine='python', sep='\t')
        print("Data Loaded.")
    return df

def run(config):
    df = load_data(args)

    if args.train: train(args, df, config)
    if args.test: test(args, df, config)
    if args.predict: predict(args, df, config)

    if args.output:
        output_name,output_extension = os.path.splitext(args.output)
        if output_extension.upper() == ".CSV":
            df.to_csv(args.output, sep='\t')
        elif output_extension.upper() == ".JSON":
            with open(args.output, 'w') as f:
                f.write(df.to_json())
        else:
            print(df)


if __name__ == '__main__':
    prog = "ML Boilerplate"
    descr = "Boilerplate for ML"
    parser = argparse.ArgumentParser(prog=prog, description=descr)
    parser.add_argument("--input", type=str, default="./data/input.csv", help="Input Dataset")
    parser.add_argument("--output", type=str, default="./data/output.csv", help="Output Directory (of Model or Data)")
    parser.add_argument("--config", type=str, default="config.json", help="Config file")
    parser.add_argument("--test", default=False, help="Test Labeled Data")
    parser.add_argument("--predict", default=False, help="Generate Predictions on Input")
    parser.add_argument("--train", default=False, help="Train new Model")

    args = parser.parse_args()
    config = json.load(open(args.config)) if args.config else {"columns": ""}
    model_dir_path="./models"


    run(config)
