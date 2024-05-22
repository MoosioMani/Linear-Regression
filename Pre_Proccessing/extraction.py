import pandas as pd

def extract_from_csv(file_path): 
    return pd.read_csv(file_path, parse_dates=['loan_start'], dayfirst=True, infer_datetime_format=True)