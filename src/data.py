"""
Functions for reading preprocessed datasets.
"""
import numpy as np
import os
import pandas as pd

def load_examples(data_path, trn_date, val_date, test_date, seq=2, verbose=False, with_index=False):
    """
    Load examples of a dataset.

    :param data_path: a path that contains preprocessed feature vectors.
    :param trn_date: a starting date of training data.
    :param val_date: a starting date of validation data.
    :param test_date: a starting date of test data.
    :param seq: the length of time series sequences (the same as the window parameter).
    :param verbose: whether to print the detailed process.
    :param with_index: whether to read examples with the corresponding index.
    :return: the pair of stock symbols and loaded data.
    """
    ticks=sorted(pd.read_csv(os.path.join(data_path, '..', 'stocks.csv'))['stock'].tolist())
    fnames=['%s.csv'%tick for tick in ticks]


    data = []
    for index, fname in enumerate(fnames):
        path = os.path.join(data_path, fname)
        arr = np.genfromtxt(path, dtype=float, delimiter=',', skip_header=False)
        data.append(arr)
    x_dim = data[0].shape[1] - 2

    if with_index:
        path = os.path.join(data_path, '..', 'index.csv')
        data_index = np.genfromtxt(path, dtype=float, delimiter=',', skip_header=False)

    path = os.path.join(data_path, '..', 'trading_dates.csv')
    trading_dates = np.genfromtxt(path, dtype=str, delimiter=',', skip_header=False)
    dates_idx = {date: index for index, date in enumerate(trading_dates)}

    trn_idx = dates_idx[trn_date]
    val_idx = dates_idx[val_date]
    test_idx = dates_idx[test_date]

    stocks = len(fnames)
    trn_num = val_idx - max(trn_idx, seq)
    val_num = test_idx - val_idx
    test_num = len(trading_dates) - test_idx

    def generate_instances(size, date_start, date_end):
        all_stocks = stocks + 1 if with_index else stocks
        out_x = np.zeros([size, all_stocks, seq, x_dim], dtype=np.float32)
        out_y = np.zeros([size, stocks], dtype=np.float32)
        out_xmask = np.zeros([size, stocks], dtype=bool)
        out_ymask = np.zeros([size, stocks], dtype=bool)

        for tic_idx in range(stocks):
            ins_idx = 0
            for date_idx in range(date_start, date_end):
                if date_idx < seq:  # filter out instances without length enough history
                    continue

                out_x[ins_idx, tic_idx] = data[tic_idx][date_idx - seq:date_idx, :-2]
                out_xmask[ins_idx, tic_idx] = True

                out_y[ins_idx, tic_idx] = data[tic_idx][date_idx][-2]
                out_ymask[ins_idx, tic_idx] = True

                ins_idx += 1

        if with_index:
            tic_idx = -1
            ins_idx = 0
            for date_idx in range(date_start, date_end):
                if date_idx < seq:  # filter out instances without length enough history
                    continue


                out_x[ins_idx, tic_idx] = data_index[date_idx - seq:date_idx, :-2]

                ins_idx += 1

        survive = out_ymask.sum(1) > 0
        out_x = out_x[survive]
        out_y = out_y[survive]
        out_xmask = out_xmask[survive]
        out_ymask = out_ymask[survive]
        return out_x, out_y, out_xmask, out_ymask

    trn_out = generate_instances(trn_num, trn_idx, val_idx)
    val_out = generate_instances(val_num, val_idx, test_idx)
    test_out = generate_instances(test_num, test_idx, len(trading_dates))
    data = *trn_out, *val_out, *test_out

    if verbose:
        print(len(fnames), ' tickers selected')
        print(len(trading_dates), 'trading dates:')
        print(trn_num, ' training instances')
        print(val_num, ' validation instances')
        print(test_num, ' testing instances')
    return ticks, data


def load_data(data, seq, verbose=False, with_ticks=False):
    """
    Load a dataset by the name.

    :param data: the name of a dataset to load.
    :param seq: the length of time series sequences (the same as the window parameter).
    :param verbose: whether to print the detailed process.
    :param with_ticks: whether to return the data with the list of stock symbols.
    :return: the loaded data.
    """
    if data=='kr':
        tra_date = '2014-01-02'
        val_date = '2020-07-01'
        tes_date = '2021-01-04'
    elif data=='us':
        tra_date = '2003-01-02'
        val_date = '2017-01-03'
        tes_date = '2019-01-02'
    elif data=='chn':
        tra_date = '2011-01-06'
        val_date = '2018-01-02'
        tes_date = '2019-01-02'
    elif data=='uk':
        tra_date='2014-01-06'
        val_date = '2021-01-04'
        tes_date = '2022-01-04'


    else:
        raise ValueError(data)

    path = '../data/{}/ourpped'.format(data)
    ticks, data = load_examples(path, tra_date, val_date, tes_date, seq, verbose=verbose, with_index=True)
    
    if with_ticks:
        return ticks, data
    else:
        return data
