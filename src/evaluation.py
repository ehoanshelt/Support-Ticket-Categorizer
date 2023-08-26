import pickle
import os
import yaml
import json
import pandas as pd
from sklearn.metrics import classification_report

params = yaml.safe_load(open("params.yaml"))["evaluate"]
model_file = params["model_file"]
test_data = pd.read_csv(params["test_file"], index_col=0)

features = test_data["Ticket Description"]
targets = test_data["Ticket Type"]

with open(model_file, "rb") as fd:
    model = pickle.load(fd)

def evaluate(m, X, y):
    predictions = m.predict(X)
    report = classification_report(y, predictions, output_dict=True)

    return report


def create_output_file(data, dir, filename):
    if not os.path.exists(dir):
        os.mkdir(dir)

    fullname = os.path.join(dir, filename)

    with open("metrics/test_metrics.json", "w") as outfile:
        json.dump(data, outfile, indent = 4)

test_report = evaluate(model, features, targets)
create_output_file(test_report, "metrics", "test_metrics.json")

    