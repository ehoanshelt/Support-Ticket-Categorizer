import numpy as np
import pandas as pd
import os
import yaml

params = yaml.safe_load(open("params.yaml"))["process_data"]


raw_data = pd.read_csv(params["raw_file"])
target = "Ticket Type"
column_for_processing = "Ticket Description"
columns_to_process = ["Product Purchased","Ticket Description"]

def add_product_purchased(data):
    product_purchased:str = data[0]
    description:str = data[1]
    
    description = description.replace("{product_purchased}", product_purchased)
    
    return description

def create_processed_data(data:pd.DataFrame, updated_column, columns) -> pd.DataFrame:
    data[updated_column] = data[columns].apply(add_product_purchased, axis=1)
    return data

def create_output_file(data, dir, filename):
    if not os.path.exists(dir):
        os.mkdir(dir)

    fullname = os.path.join(dir, filename)    

    data.to_csv(fullname)


processed_data = create_processed_data(raw_data, column_for_processing, columns_to_process)
processed_data= processed_data[["Ticket Subject", column_for_processing, target]]
create_output_file(processed_data,"data/processed", "customer_support_tickets_processed.csv")


