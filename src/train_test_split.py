import os
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

params = yaml.safe_load(open("params.yaml"))["train_test_split"]


def create_output_file(data, dir, filename):
    if not os.path.exists(dir):
        os.mkdir(dir)

    fullname = os.path.join(dir, filename)    

    data.to_csv(fullname)

processed_data = pd.read_csv(params["processed_file"], index_col=0)

X_train, X_test, y_train, y_test = train_test_split(processed_data["Ticket Description"], \
                                                    processed_data["Ticket Type"], \
                                                    test_size=params["test_size"])

training_data = pd.concat([X_train, y_train], axis=1)
testing_data = pd.concat([X_test, y_test], axis=1)

create_output_file(training_data, "data/training", "training.csv")
create_output_file(training_data, "data/testing", "test.csv")
