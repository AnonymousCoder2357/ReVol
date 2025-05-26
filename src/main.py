"""
Main function to train our ReVol model and to evaluate it.
"""
import argparse
import io
import os
import shutil
import pandas as pd

import torch
from torch import optim, nn
import numpy as np

import models
from data import load_data
from utils import to_device, to_loader
import scipy.stats as stats
import matplotlib.pyplot as plt

def read_closing_prices(args, ticks):
    """
    Read closing prices for given tickers from CSV files.

    param args: An object containing the path to the data directory.
    param ticks: A list of ticker symbols to load.

    return: A DataFrame with closing prices, where rows are dates and columns are tickers.
    """
    data_path = f'../data/{args.data}/ourpped'
    path = os.path.join(data_path, '..', 'trading_dates.csv')
    dates = np.genfromtxt(path, dtype=str, delimiter=',', skip_header=False)
    data = []
    for index, tick in enumerate(ticks):
        path = os.path.join(data_path, str(tick) + '.csv')
        arr = np.genfromtxt(path, dtype=float, delimiter=',', skip_header=False)
        data.append(arr[:, -1])
    data = np.array(data).transpose()
    return pd.DataFrame(data, columns=ticks, index=dates)

def calc_ic(pred, label):
    """
    Calculate Information Coefficient (IC) and Rank IC (RIC) between prediction and ground truth.

    param pred: A 1D array or Series of model predictions.
    param label: A 1D array or Series of true target values.

    return: A tuple of (IC, RIC), where IC is the Pearson correlation and RIC is the Spearman correlation.
    """
    df = pd.DataFrame({'pred':pred, 'label':label})
    ic = df['pred'].corr(df['label'])
    ric = df['pred'].corr(df['label'], method='spearman')
    return ic, ric

class Trainer:
    """
    A class for training and evaluation our model.
    """
    def __init__(self, model, optimizer, device, guid=0):
        """
        Initializer function.

        :param model: a PyTorch module of ReVol.
        :param optimizer: an optimizer for the model.
        :param device: a device where the model is stored.
        """
        self.guid=guid
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.loss = nn.MSELoss(reduction='none')

    def train(self, loader):
        """
        Train the model by a single epoch given a data loader.

        :param loader: a data loader for training.
        :return: the training loss.
        """
        self.model.train()
        loss_sum, count = 0, 0
        for x, y, xm, ym in loader:
            x = x.to(self.device)
            y = y.to(self.device)
            xm = xm.to(self.device)
            ym = ym.to(self.device)
            y_pred,mu,sigma = self.model(x, xm,with_mu=True)
            loss = self.loss(torch.exp(y_pred), torch.exp(y.float()))[ym].mean() + self.guid * self.loss(mu,torch.mean(x[...,3],-1)).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_sum += loss.item() * ym.sum()
            count += ym.sum()
        return loss_sum / count


    def evaluate(self, loader):
        """
        Evaluate the model given a data loader.

        :param loader: a data loader for evaluation.
        :return: the accuracy (ACC) and MCC.
        """
        self.model.eval()
        ic,ric=[],[]
        for x, y, xm, ym in loader:
            x = x.to(self.device)
            y = y
            xm = xm.to(self.device)
            y_pred = self.model(x, xm)

            for label, pred in zip(torch.exp(y).detach().numpy(),torch.exp(y_pred).cpu().detach().numpy()):
                daily_ic,daily_ric=calc_ic(pred,label)
                ic.append(daily_ic)
                ric.append(daily_ric)

        metrics = {
            'IC': np.mean(ic),
            'ICIR': np.mean(ic)/np.std(ic),
            'RIC': np.mean(ric),
            'RICIR': np.mean(ric)/np.std(ric)
        }

        return metrics
    

def evaluate_investment(args, model, device, top_k):
    """
    Simulate an actual investment and report the return.

    This functions needs not to be run for new datasets; do not set the `invest` argument of the
    main function to True.

    :param args: an argument parser.
    :param model: a trained ReVol model.
    :param device: a device where the model is stored.
    :param top_k: the number of stocks to invest at each day.
    :return: the return of the investment.
    """
    if args.data=='kr':
        date_start = '2021-01-04'
    elif args.data=='us':
        date_start = '2019-01-02'
    elif args.data=='chn':
        date_start = '2019-01-02'
    elif args.data=='uk':
        date_start = '2022-01-04'
    else:
        raise ValueError(args.data)
    
    ticks, data = load_data(args.data, args.statwindow, with_ticks=True)
    test_data = data[-4:]
    test_loader = to_loader(*test_data, batch_size=128)

    path = f'../data/{args.data}/trading_dates.csv'
    dates = np.genfromtxt(path, dtype=str, delimiter=',', skip_header=False)
    dates = dates[list(dates).index(date_start):]
    prices = read_closing_prices(args, ticks)

    model=model.to('cpu')
    model.device='cpu'
    model.eval()
    predictions = []

    for idx, (x, y, xm, ym) in enumerate(test_loader):
        assert False not in xm
        y_pred = model(x, xm)
        predictions.append(y_pred)
    predictions = torch.cat(predictions)

    model=model.to(device)
    model.device=device
    
    money = 1
    money_list = []
    prev_values = []
    return_list=[]
    for idx, pred in enumerate(predictions):
        date = dates[idx]
        if idx > 0:
            portfolio = [ticks[i] for i in torch.topk(pred, k=top_k)[1]]
            diff = prices.loc[date, portfolio] / prev_values[portfolio]
            return_list.append(diff.mean()-1)
            money *= diff.mean()
        prev_values = prices.loc[date, :]
        money_list.append(money)

    return money_list[-1]**(252/len(predictions))-1, np.mean(return_list)/np.std(return_list)*np.sqrt(252)


def get_open_close_ratio(data):
    """
    Calculate the ratio of open-to-close return variance to close-to-close return variance.

    param data: A string representing the market identifier (e.g., 'kr', 'us', 'chn', 'uk').

    return: A float value representing the ratio used for return scaling.
    """
    if data=='kr':
        tra_date = '2014-01-02'
        val_date = '2020-07-01'
    elif data=='us':
        tra_date = '2003-01-02'
        val_date = '2017-01-03'
    elif data=='chn':
        tra_date = '2011-01-06'
        val_date = '2018-01-02'
    elif data=='uk':
        tra_date='2014-01-06'
        val_date = '2021-01-04'
    stocks=pd.read_csv('../data/%s/stocks.csv'%data)['stock'].tolist()
    datas_tr = [pd.read_csv('../data/%s/raw/%s.csv'%(data,stock)).iloc[:,1:].to_numpy() for stock in stocks]
    dates=pd.read_csv('../data/%s/trading_dates.csv'%data,header=None)
    dates=dates[dates.columns[0]].tolist()
    datas_tr=[i[dates.index(tra_date):dates.index(val_date)] for i in datas_tr]
    datas=[]
    for ii in range(len(stocks)):
        process=datas_tr[ii]
        process2=np.empty([len(process)-1,2])
        process2[:,1]=np.log(process[1:,3])-np.log(process[:-1,3])
        process2[:,0]=np.log(process[1:,0])-np.log(process[:-1,3])
        datas.append(process2)
    datas=np.concatenate(datas,axis=0)
    return (datas[:,0]*datas[:,1]).sum()/(datas[:,1]**2).sum()


def parse_args():
    """
    Parse command line arguments for the script.

    :return: the parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='us')
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--guid', type=float, default=0.25)
    parser.add_argument('--window', type=int, default=8)
    parser.add_argument('--statwindow', type=int, default=256)

    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--units', type=int, default=256)
    parser.add_argument('--rve',type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)

    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--patience', type=int, default=35)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--tnhead', type=int, default=4)
    parser.add_argument('--snhead', type=int, default=2)
    parser.add_argument('--beta', type=int, default=5)



    parser.add_argument('--invest', action='store_true', default=True)
    parser.add_argument('--load', action='store_true', default=False)
    parser.add_argument('--no-stop', action='store_true', default=False)
    parser.add_argument('--out', type=str, default='../out')
    parser.add_argument('--device', type=int, default=None)
    parser.add_argument('--save', action='store_true', default=True)
    parser.add_argument('--silent', action='store_true', default=False)
    parser.add_argument('--fm', type=str, default='lstm')
    return parser.parse_args()


def main():
    """
    Main function for training and evaluation.

    :return: None.
    """

    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    trn_x, trn_y, trn_xm, trn_ym, val_x, val_y, val_xm, val_ym, test_x, test_y, test_xm, test_ym = \
        load_data(args.data, args.statwindow)
    num_stocks = trn_x.shape[1] - 1
    oc_ratio = get_open_close_ratio(args.data)
    device = to_device(args.device)

    if args.fm == 'lstm':
        fm = models.LSTM(args.units)

    
    model = models.ReVol(args.rve,oc_ratio, args.window, device,fm)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    trainer = Trainer(model, optimizer, device, guid=args.guid)

    trn_loader = to_loader(trn_x, trn_y, trn_xm, trn_ym, args.batch_size, shuffle=True)
    val_loader = to_loader(val_x, val_y, val_xm, val_ym, args.batch_size)
    test_loader = to_loader(test_x, test_y, test_xm, test_ym, args.batch_size)

    out_path = os.path.join(args.out, str(args.seed))

    if args.load:
        save_path = os.path.join(out_path, 'model.pth')
        model.load_state_dict(torch.load(save_path))
    else:
        if os.path.exists(out_path):
            shutil.rmtree(out_path)
        os.makedirs(out_path, exist_ok=True)

        saved_model, best_epoch, best_ic = io.BytesIO(), 0, -np.inf
        for epoch in range(args.epochs + 1):
            trn_loss = 0
            if epoch > 0:
                trn_loss = trainer.train(trn_loader)
            val_metrics = trainer.evaluate(val_loader)
            log = '{:3d}\t{:7.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}'.format(
                epoch, trn_loss, val_metrics['IC'],val_metrics['ICIR'],val_metrics['RIC'],val_metrics['RICIR'])
            if epoch >= 10 and val_metrics['IC'] > best_ic:
                best_epoch = epoch
                best_ic = val_metrics['IC']
                saved_model.seek(0)
                torch.save(model.state_dict(), saved_model)
                log += '\tBEST'
            with open(os.path.join(out_path, 'log.tsv'), 'a') as f:
                f.write(log + '\n')
            if not args.silent:
                print(log)
            if args.patience > 0 and epoch >= best_epoch + args.patience:
                break

        if not args.no_stop:
            saved_model.seek(0)
            model.load_state_dict(torch.load(saved_model))

        if args.save:
            save_path = os.path.join(out_path, 'model.pth')
            torch.save(model.state_dict(), save_path)

    val_res = trainer.evaluate(val_loader)
    test_res = trainer.evaluate(test_loader)
    log = '{}\t{}\t{}\t{}\t{}'.format(val_res['IC'],test_res['IC'],test_res['ICIR'],test_res['RIC'],test_res['RICIR'])

    if args.invest:
        ar,sr=evaluate_investment(args, model, device, top_k=5)
        log += '\t{}\t{}'.format(ar,sr)


    if not args.silent:
        print(log)

    with open(os.path.join(out_path, 'out.tsv'), 'w') as f:
        f.write(log + '\n')



if __name__ == '__main__':
    main()
