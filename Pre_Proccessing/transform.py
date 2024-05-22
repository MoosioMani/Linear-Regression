import pandas as pd

# Drop Nan Columns
def all_nan(df):
    return df.dropna(how='all', axis=0)

# Fill NaN 
def fillna(data):
    data.fillna(value={
        'client_id':data.client_id.mode()[0],
        'loan_type':data.loan_type.mode()[0],
        'loan_amount':data.loan_amount.mean(),
        'repaid':data.repaid.mode()[0],
        'loan_start':data.loan_start.mode()[0],
        'loan_end':data.loan_end.mode()[0],
        'rate':data.rate.mean()
    },inplace=True)
    return data

# Noisy Data
def noisy_data(data):
    data.loc[data.loan_amount.astype('str').str.isalpha(),'loan_amount']=data.loan_amount.mode()[0]
    return data

# Convert Types
def data_types(data):
    data['loan_start'] = pd.to_datetime(data['loan_start'], format = '%Y-%m-%d')
    data['loan_end'] = pd.to_datetime(data['loan_end'], format = '%Y-%m-%d')
    data['days'] = (data['loan_end'] - data['loan_start']).dt.days
    data.drop(columns=['loan_start', 'loan_end'], inplace=True)
    return data

# One-Hand Encoder
def one_hand_encoder(data, columns):
    data = pd.get_dummies(data, columns=columns)
    return data

#Label Encoder
from sklearn.preprocessing import LabelEncoder
def label_encoder(data, columns):
    le = LabelEncoder()
    for col in columns:    
        data[col] = le.fit_transform(data[col])
    return data

# Discret 
from sklearn.preprocessing import KBinsDiscretizer
def k_bins_discretizer(data, columns):
    dis = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='uniform')
    for col in columns:
        data[col] = dis.fit_transform(data[[col]])   
    return data

# Outlier Plot 
import plotly.express as px
def outlier_columns_by_plotly(data, columns):
    fig = px.box(data, y=columns)
    fig.show()

# Remove Outlier Rate
def remove_outlier_rate(data, min_w, max_w):
    df = pd.DataFrame(data)
    data = df[(df['rate'] >= min_w) & (df['rate'] <= max_w)]
    return data

# Normal Scaler
from sklearn.preprocessing import MinMaxScaler
def min_max_scaler(data, columns):
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data, columns)
    data = pd.DataFrame(data)
    data.columns=columns
    return data

# مStandard Scaler
from sklearn.preprocessing import StandardScaler
def standard_scaler(data, columns):
    scaler = StandardScaler()
    data = scaler.fit_transform(data, columns)
    data = pd.DataFrame(data)
    data.columns=columns
    return data