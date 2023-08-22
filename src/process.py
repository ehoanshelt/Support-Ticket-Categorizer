import numpy as np
import pandas as pd

raw_data = pd.read_csv("data/raw/customer_support_tickets.csv")
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

raw_data = create_processed_data(raw_data, column_for_processing, columns_to_process)
raw_data = raw_data[["Ticket Subject", column_for_processing, target]]
raw_data.to_csv("data/processed/customer_support_tickets_processed.csv")

