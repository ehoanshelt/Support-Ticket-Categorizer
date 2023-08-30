import pickle
import yaml
import numpy as np
import pandas as pd
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

params = yaml.safe_load(open("params.yaml"))["train"]

training_data = pd.read_csv(params["training_file"], index_col=0)
output = params["output_file"]
X_train = training_data["Ticket Description"]
y_train = training_data["Ticket Type"]
parameter_grid = {
    "vect__max_df": (0.2, 0.4, 0.6, 0.8, 1.0),
    "vect__min_df": (1, 3, 5, 10),
    "vect__norm": ("l1", "l2"),
}

pipeline = Pipeline(
    [
        ("vect", TfidfVectorizer(stop_words="english")),
        ("clf", LinearSVC()),
    ]
)

tuned_model = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=parameter_grid,
    n_iter=40,
    random_state=0,
    n_jobs=2,
    verbose=1,
)

tuned_model.fit(X_train, y_train)

print("Best parameters combination found:")
best_parameters = tuned_model.best_estimator_.get_params()
for param_name in sorted(parameter_grid.keys()):
    print(f"{param_name}: {best_parameters[param_name]}")

with open(output, "wb") as fd:
    pickle.dump(tuned_model, fd)
