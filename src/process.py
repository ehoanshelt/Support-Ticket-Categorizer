import numpy as np
import pandas as pd
import os
import yaml
import sys
from sklearn.preprocessing import LabelEncoder

params = yaml.safe_load(open("params.yaml"))["process"]


raw_data = pd.read_csv(params["raw_file"])
target = "Ticket Type"
ticket_descr= "Ticket Description"
pp = "Product Purchased"


#Check if expected column are in the raw data
if not pd.Series([target, ticket_descr, pp]).isin(raw_data.columns).all():
    print(f"Expect columns in raw data not found. Need {target}, {ticket_descr}, and {pp}")
    sys.exit()



# Ticket Description has {product_purchased} in the text. Need to replace that with the actual product purchased
def add_product_purchased(data):
    product_purchased:str = data[0]
    description:str = data[1]
    
    description = description.replace("{product_purchased}", product_purchased)
    
    return description

def encode_label(data):
    le = LabelEncoder()
    data[target] = le.fit_transform(data[target])
    return data

def create_output_file(data, dir, filename):
    if not os.path.exists(dir):
        os.mkdir(dir)

    fullname = os.path.join(dir, filename)    

    data.to_csv(fullname)


# This loops through the data and apply the above function
def create_processed_data(data:pd.DataFrame, updated_column, columns) -> pd.DataFrame:
    data = encode_label(data)
    data[updated_column] = data[columns].apply(add_product_purchased, axis=1)
    data["Ticket Description"] = data["Ticket Subject"] + " " + data["Ticket Description"]
    data.drop("Ticket Subject", axis=1, inplace=True)
    return data



processed_data = create_processed_data(raw_data, ticket_descr, [pp, ticket_descr])
processed_data = processed_data[[ticket_descr, target]]
create_output_file(processed_data,"data/processed", "customer_support_tickets_processed.csv")


