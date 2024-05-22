from extraction import *
from transform import *
from load import *

def seperate(data):
    print(data)
    print(80*'*')

data = extract_from_csv('./loans.csv')
seperate(data)

data = all_nan(data)
seperate(data)

data = fillna(data)
seperate(data)

data = noisy_data(data)
seperate(data)

data = data_types(data)
seperate(data)

data = one_hand_encoder(data, ['repaid'])
seperate(data)

data = label_encoder(data, ['loan_type'])
seperate(data)

data = k_bins_discretizer(data, ['client_id'])
seperate(data)

# outlier_columns_by_plotly(data, ['rate'])

data = remove_outlier_rate(data, 0, 5)
seperate(data)

data = min_max_scaler(data, ['client_id', 'loan_type', 'loan_amount', 'loan_id', 'days', 'rate', 'repaid_0', 'repaid_1', ])
seperate(data)

load(data, './target.csv')