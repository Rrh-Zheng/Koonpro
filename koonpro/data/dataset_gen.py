import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

scaler = StandardScaler()

def takens_embed(dataset, time_shifts, window):
    data_wd = []
    for i in range(dataset.shape[0] - window + 1):
        temp = []
        for j in range(window):
            temp.append(dataset[i + j, :])
        temp = np.stack(temp, 1)
        data_wd.append(temp)
    data_wd = np.array(data_wd)
    data = []
    for i in range(dataset.shape[0] - time_shifts - window + 1):
        temp = []
        for j in range(time_shifts + 1):
            temp.append(data_wd[i + j, :, :])
        data.append(temp)
    data = np.array(data)
    return data

def read_data(args, flag):
    data_path = 'data/' + args.data_name + '.csv'
    df_raw = pd.read_csv(data_path)
    cols_data = df_raw.columns[1:]
    df_data = df_raw[cols_data]
    data = df_data.values

    scaler.fit(data)
    data_std = scaler.fit_transform(data)

    if flag == 'test':
        st, ed = args.test_size[0], args.test_size[1]
    elif flag == 'train':
        st, ed = args.train_size[0], args.train_size[1]
    elif flag == 'val':
        st, ed = args.val_size[0], args.val_size[1]

    st = int(st * data.shape[0])
    ed = int(ed * data.shape[0])

    if args.scale:
        data_std = data_std[st: ed, :]
        print(flag, ' size:', data_std.shape)
        return data_std
    else:
        data = data[st: ed, :]
        return data