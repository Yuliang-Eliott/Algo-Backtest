
import sys
import json
import os
import copy
import multiprocessing
import pandas as pd
import numpy as np
sys.path.append('/usr/local/anaconda3/lib/python3.8/site-packages')
from datahub_api import get
from pathlib import Path
from utils import *
current_folder = os.path.dirname(os.path.abspath(__file__))
class Backtest:
    def __init__(self, config_file,date):
        self.config_file = config_file
        self.date = date
        self.load_data()
        self.load_market_data()
        self.init_trade()

    def load_data(self):
       
        with open(f'{current_folder}/backtest_config.json','r') as file:
            self.config = json.load(file)
        self.parent_path = self.config["parent_path"]
        self.snap_data_url = self.config["snap_data_url"]
        self.load_score()
        self.load_market_data()

    def load_score(self):
        # load_score
        self.score_path = self.config["score_path"]
        self.timestamp_path = self.config["timestamp_path"]
        self.localtime_path = self.config["localtime_path"]
        self.pred_score = np.load(f"{self.score_path}/{self.date}.npy")
        self.used_label_idx = self.config["used_label_idx"]
        self.ticker_idx_path = self.config["ticker_idx_path"]
        if len(self.used_label_idx)==1:
            self.pred_score = self.pred_score[:,self.used_label_idx].reshape(-1)
        else:
            self.pred_score = np.mean(np.vstack([self.pred_score[:,x] for x in self.used_label_idx]),axis=0)
        # load_time_stamp
        self.timestamps = np.load(f'{self.timestamp_path}/{self.date}.npy')
        self.timestamps = [int(self.date+('0'+str(int(x)))[-6:]+'000') for x in self.timestamps]
        self.localtime = np.load(f'{self.localtime_path}/{self.date}.npy')
        self.localtime = [int(self.date+('0'+str(int(x*1000)))[-9:]) for x in self.localtime]
        #load tickers
        with open(f"{self.ticker_idx_path}/{self.date}.json",'r') as file:
            self.ticker_idx = json.load(file)
        # save_score
        self.ticker_score_map = {}
        for ticker in self.ticker_idx.keys():
            self.ticker_score_map[ticker] = {}
            for sample_idx in range(self.ticker_idx[ticker][0],self.ticker_idx[ticker][1]):
                self.ticker_score_map[ticker][self.localtime[sample_idx]] = self.pred_score[sample_idx]

    def load_market_data(self):
        self.parent_order_df = pd.read_csv(f'{self.parent_path}/27519_{self.date}_parent.csv')
        self.need_tickers_all = list(self.parent_order_df['ticker'].unique())
        self.snap_sub_all = get(self.snap_data_url,date=self.date,use_codes=self.need_tickers_all)

    def init_trade(self):
        self.traderes = []
        self.save_dir = self.config["save_dir"]
        
        self.transfer_map = {}
        self.transfer_map["a"] = self.config["a"]
        self.transfer_map["b"] = self.config["b"]
        self.transfer_map["c"] = self.config["c"]
        self.transfer_map["d"] = self.config["d"]
        self.transfer_map["buy_take_threshold"] = self.config["buy_take_threshold"]
        self.transfer_map["sell_take_threshold"] = self.config["sell_take_threshold"]
        self.transfer_map["multivol"] = self.config["multivol"]
        self.transfer_map["sleepmulti"] = self.config["sleepmulti"]
    
    def run_backtest_parent(self):
    
        for ticker_num,value in self.parent_order_df.iterrows():
            parent_order = Parentmanager(self.date,value,self.ticker_score_map[value["ticker"]],self.transfer_map,self.snap_sub_all[value["ticker"]])
            parent_order.run_backtest_unit()
            self.run_analysis(parent_order)

    def run_analysis(self,parent_order):
        temp_parent_dict = {}
        temp_parent_dict["direct"] = parent_order.direct
        temp_parent_dict["original_volume"] = parent_order.original_volume
        temp_parent_dict["ticker"] = parent_order.ticker
        temp_parent_dict["timetick"] = parent_order.timetick
        temp_parent_dict["traderecord"] = parent_order.traderecord
        snap_sub = parent_order.snap_sub
        temp_parent_dict['15minvwap'] = (snap_sub[min(len(snap_sub)-1,parent_order.snap_first_tic+300),col_idx_map["AccTurnover"]]-snap_sub[parent_order.snap_first_tic,col_idx_map["AccTurnover"]])/(snap_sub[min(len(snap_sub)-1,parent_order.snap_first_tic+300),col_idx_map["AccVolume"]]-snap_sub[parent_order.snap_first_tic,col_idx_map["AccVolume"]])
        temp_parent_dict['30minvwap'] = (snap_sub[min(len(snap_sub)-1,parent_order.snap_first_tic+600),col_idx_map["AccTurnover"]]-snap_sub[parent_order.snap_first_tic,col_idx_map["AccTurnover"]])/(snap_sub[min(len(snap_sub)-1,parent_order.snap_first_tic+600),col_idx_map["AccVolume"]]-snap_sub[parent_order.snap_first_tic,col_idx_map["AccVolume"]])
        temp_parent_dict['pct10amount'] = get_vwap(snap_sub[parent_order.snap_first_tic:,:].copy(),order_volume = abs(parent_order.original_volume),times = 10)
        if len(temp_parent_dict['traderecord'])!=0:
            try:
                temp_parent_dict['tradeprice'] = np.sum(np.array(temp_parent_dict['traderecord'])[:,1]*np.array(temp_parent_dict['traderecord'])[:,2])/np.sum(np.array(temp_parent_dict['traderecord'])[:,1])
                temp_parent_dict['trade_volume'] = np.sum(np.array(temp_parent_dict['traderecord'])[:,1])
                self.traderes.append(temp_parent_dict)
            except:
                print(temp_parent_dict['traderecord'])
        return temp_parent_dict

    def save_result(self):
        with open(f'{self.save_dir}/{self.date}.json','w') as f:
            json.dump(self.traderes,f)
        pass

class Parentmanager:
    def __init__(self,date,value,pred_score_map,transfer_map,snap_sub) -> None:
        self.date = date
        self.traderecord = []
        self.direct = "buy" if value['dir_list']==1 else "sell"
        self.original_volume = value['all_order_volume']
        self.ticker = value['ticker']
        self.timetick = date+('0'+str(int(value['update_time'])))[-6:]
        self.pred_score_map = pred_score_map
        self.pred_localtime_list = list(self.pred_score_map.keys())
        self.pred_localtime_list.sort()
        self.transfer_map = transfer_map
        self.leftvol = abs(self.original_volume)
        self.last_tick = 0
        self.snap_first_tic = 0
        self.tradevol = 0
        self.snap_sub = snap_sub

    def update_start_acc(self):
        self.start_acc_volume = self.snap_sub[self.snap_first_tic,col_idx_map["AccVolume"]]
        self.start_acc_turnover = self.snap_sub[self.snap_first_tic,col_idx_map["AccTurnover"]]

    def update_last_acc(self):
        self.last_acc_volume = self.snap_sub[self.last_tick,col_idx_map["AccVolume"]]
        self.last_acc_turnover = self.snap_sub[self.last_tick,col_idx_map["AccTurnover"]]

    def ontick(self,snap_list,snap_id):
        #发信号后的一个tic，记录下id
        if (self.leftvol==0)|((self.direct == "buy")&(snap_list[col_idx_map["AskPrice1"]]==0))|((self.direct == "sell")&(snap_list[col_idx_map["BidPrice1"]]==0)):
            return False
        punish_point = 0
        time_tic = snap_list[0]
        mid = (snap_list[col_idx_map["AskPrice1"]]+snap_list[col_idx_map["BidPrice1"]]) /(int(snap_list[col_idx_map["AskPrice1"]]>0)+int(snap_list[col_idx_map["BidPrice1"]]>0)) 
        pred_idx = np.where(self.pred_localtime_list<=time_tic)[0]
        if len(pred_idx)==0:
            return False
        else:
            pred_idx = pred_idx[-1]
        if (mid ==0)|(np.isnan(mid))|(snap_list[0]<int(self.timetick+'000'))|(snap_list[0]>=int(self.date+'145500000')):
            return False
        if (self.snap_first_tic==0)&(snap_list[0]>=int(self.timetick+'000')):
            self.snap_first_tic = snap_id
            self.last_tick = snap_id
        if (snap_list[0]>=int(self.date+'144000000')):
            punish_point = 3
        if (snap_list[0]>=int(self.date+'145000000')):
            punish_point = 10
        if (snap_list[0]>=int(self.date+'145300000')):
            punish_point = 100
        if self.direct == "buy":
           
            order_price = mid*(1+1*cubic_func(self.pred_score_map[self.pred_localtime_list[pred_idx]],self.transfer_map["a"],self.transfer_map["b"],self.transfer_map["c"],self.transfer_map["d"])-(self.transfer_map["buy_take_threshold"]-punish_point)/1e4)
            if order_price>=snap_list[col_idx_map["AskPrice1"]]:
                if (self.tradevol*self.transfer_map["multivol"]<=(snap_list[col_idx_map["AccVolume"]]-self.start_acc_volume))&(self.tradevol*self.transfer_map["sleepmulti"]<=(snap_list[col_idx_map["AccVolume"]]-self.last_acc_volume))&(self.leftvol>0):
                    self.traderecord.append([snap_list[0],min(self.leftvol,snap_list[col_idx_map["AskVol1"]]),snap_list[col_idx_map["AskPrice1"]]])
                    self.leftvol-=min(self.leftvol,snap_list[col_idx_map["AskVol1"]])
                    self.tradevol+=min(self.leftvol,snap_list[col_idx_map["AskVol1"]])
                    self.last_tick = copy.copy(snap_id)
        elif self.direct == "sell":
            order_price = mid*(1+1*cubic_func(self.pred_score_map[self.pred_localtime_list[pred_idx]],self.transfer_map["a"],self.transfer_map["b"],self.transfer_map["c"],self.transfer_map["d"])+(self.transfer_map["sell_take_threshold"]-punish_point)/1e4)
            if order_price<=snap_list[col_idx_map["BidPrice1"]]:
                if (self.tradevol*self.transfer_map["multivol"]<=(snap_list[col_idx_map["AccVolume"]]-self.start_acc_volume))&(self.tradevol*self.transfer_map["sleepmulti"]<=(snap_list[col_idx_map["AccVolume"]]-self.last_acc_volume))&(self.leftvol>0):
                    self.traderecord.append([snap_list[0],min(self.leftvol,snap_list[col_idx_map["BidVol1"]]),snap_list[col_idx_map["BidPrice1"]]])
                    self.leftvol-=min(self.leftvol,snap_list[col_idx_map["BidVol1"]])
                    self.tradevol+=min(self.leftvol,snap_list[col_idx_map["BidVol1"]])
                    self.last_tick = copy.copy(snap_id)

    def run_backtest_unit(self):
        if (len(self.snap_sub)==0)|(self.original_volume==0):
            return {}
        for j in range(len(self.snap_sub)):
            self.update_start_acc()
            self.update_last_acc()
            self.ontick(self.snap_sub[j],j)

if __name__ == '__main__':
    date = sys.argv[1]
    bktst = Backtest("backtest_config.json",date)
    bktst.run_backtest_parent()
    bktst.save_result()

    


