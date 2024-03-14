import sys
import json
import os
import copy
import multiprocessing
import pandas as pd
import numpy as np
# Append a directory to sys.path to include external Python packages
sys.path.append('/usr/local/anaconda3/lib/python3.8/site-packages')
from datahub_api import get  # Import a custom function to fetch data
from pathlib import Path
from utils import *  # Import all functions and classes from a utility script

# Set the current working directory to the script's location
current_folder = os.path.dirname(os.path.abspath(__file__))

class Backtest:
    def __init__(self, config_file, date):
        self.config_file = config_file  # The path to the configuration file
        # Load the configuration file to set up paths and parameters for backtesting
        with open(f'{current_folder}/backtest_config_fy.json', 'r') as file:
            self.config = json.load(file)
        self.date = date  # The date for which to run the backtest
        # Extract necessary paths from the config to locate data files
        self.parent_path = self.config["parent_path"]
        self.base_data_path = self.config["base_data_path"]
        self.score_path = self.config["score_path"]
        self.used_label_idx = self.config["used_label_idx"]  # Indices of labels to use
        # Load the parent order data for the given date
        self.parent_order_df = pd.read_csv(f'{self.parent_path}/27519_{self.date}_parent.csv')
        # Extract unique ticker symbols from the parent orders
        self.need_tickers_all = list(self.parent_order_df['ticker'].unique())
        self.snap_data_url = self.config["snap_data_url"]  # URL to fetch market snapshot data
        # Load various datasets required for backtesting
        self.load_data()
        self.load_market_data()
        self.init_trade()  # Initialize trading parameters and structures
        self.load_ticker_and_score()  # Load scoring data for tickers

    def load_data(self):
        # Load predictive scores and transform them based on selected indices
        self.pred_score = np.load(self.score_path+f"score_{self.date}.npy")[0].T
        self.pred_score = np.mean(np.vstack([self.pred_score[:, x] for x in self.used_label_idx]), axis=0)
        # Load timestamp and ticker information for the trading day
        self.timestamps = np.load(f'{self.base_data_path}/{self.date}_time.npy')
        self.timestamps = [int(self.date + ('0'+str(int(x)))[-9:]) for x in self.timestamps]
        self.tickers = np.load(f'{self.base_data_path}/{self.date}_ticker.npy')
        self.load_market_data()  # Load market data for the given date

    def load_ticker_and_score(self):
        # Create a mapping between tickers and their predictive scores
        self.ticker_score_map = {}
        for pred_idx in range(len(self.pred_score)):
            if self.tickers[pred_idx] not in self.ticker_score_map.keys():
                self.ticker_score_map[self.tickers[pred_idx]] = {}
            self.ticker_score_map[self.tickers[pred_idx]][self.timestamps[pred_idx]] = self.pred_score[pred_idx]

    def load_market_data(self):
        # Fetch market snapshot data for all required tickers on the given date
        self.snap_sub_all = get(self.snap_data_url, date=self.date, use_codes=self.need_tickers_all)

    def init_trade(self):
        # Initialize trading result structures and configuration parameters
        self.traderes = []
        self.position_dict = {}
        self.save_dir = self.config["save_dir"]
        # Setup conversion parameters from score to predictive return
        self.transfer_map = {key: self.config[key] for key in ["a", "b", "c", "d", "buy_take_threshold", "sell_take_threshold", "multivol", "sleepmulti"]}

    def run_backtest_parent(self):
        # Iterate over each parent order and simulate its execution
        for ticker_num, value in self.parent_order_df.iterrows():
            parent_order = Parentmanager(self.date, value, self.ticker_score_map[value["ticker"]], self.transfer_map, self.snap_sub_all[value["ticker"]])
            parent_order.run_backtest_unit()
            self.run_analysis(parent_order)

    def run_analysis(self, parent_order):
        # Analyze the execution of a parent order and compile results
        temp_parent_dict = {"direct": parent_order.direct, "original_volume": parent_order.original_volume, "ticker": parent_order.ticker, "timetick": parent_order.timetick, "traderecord": parent_order.traderecord}
        snap_sub = parent_order.snap_sub
        # Calculate VWAP (Volume Weighted Average Price) over different time intervals
        # Further analysis and calculations omitted for brevity
        if len(temp_parent_dict['traderecord']) != 0:
            try:
                # Calculate the average trade price and total trade volume
                trade_prices_volumes = np.array(temp_parent_dict['traderecord'])[:, 1:3].astype(float)
                temp_parent_dict['tradeprice'] = np.sum(trade_prices_volumes[:, 0] * trade_prices_volumes[:, 1]) / np.sum(trade_prices_volumes[:, 0])
                temp_parent_dict['trade_volume'] = np.sum(trade_prices_volumes[:, 0])
                self.traderes.append(temp_parent_dict)
            except Exception as e:
                print(e, temp_parent_dict['traderecord'])
        return temp_parent_dict

    def save_result(self):
        # Save the backtesting results to a file
        with open(f'{self.save_dir}/{self.date}.json', 'w') as f:
            json.dump(self.traderes, f)

class Parentmanager:
    def __init__(self, date, value, pred_score_map, transfer_map, snap_sub):
        # Initialize parent order with given parameters and setup for backtesting
        self.traderecord = []
        self.direct = "buy" if value['dir_list'] == 1 else "sell"
        self.date = date
        self.original_volume = value['all_order_volume']
        self.ticker = value['ticker']
        self.timetick = date + ('0'+str(int(value['update_time'])))[-6:]
        self.pred_score_map = pred_score_map
        self.transfer_map = transfer_map
        self.leftvol = abs(self.original_volume)
        self.last_tick = 0
        self.snap_first_tic = 0
        self.tradevol = 0
        self.snap_sub = snap_sub 

    def update_start_acc(self):
        # Updates the starting accumulated volume and turnover based on the first tick data
        self.start_acc_volume = self.snap_sub[self.snap_first_tic, col_idx_map["AccVolume"]]
        self.start_acc_turnover = self.snap_sub[self.snap_first_tic, col_idx_map["AccTurnover"]]

    def update_last_acc(self):
        # Updates the last accumulated volume and turnover based on the current tick data
        self.last_acc_volume = self.snap_sub[self.last_tick, col_idx_map["AccVolume"]]
        self.last_acc_turnover = self.snap_sub[self.last_tick, col_idx_map["AccTurnover"]]

    def ontick(self, snap_list, snap_id):
        # Processes market data for each tick and decides whether to execute a trade
        if self.leftvol == 0:
            return False  # No volume left to trade
        punish_point = 0  # Initialize a penalty variable for late trades
        
        time_tic = snap_list[0]  # The current timestamp
        # Calculate the mid price based on the best ask and bid prices
        mid = (snap_list[col_idx_map["AskPrice1"]] + snap_list[col_idx_map["BidPrice1"]]) / (int(snap_list[col_idx_map["AskPrice1"]] > 0) + int(snap_list[col_idx_map["BidPrice1"]] > 0))
        # If the mid price is invalid or the current tick is out of the trading window, return
        if (mid == 0) | (np.isnan(mid)) | (snap_list[0] < int(self.timetick+'000')) | (snap_list[0] >= int(self.date+'145500000')):
            return False
        # Mark the first tick within the trading window
        if (self.snap_first_tic == 0) & (snap_list[0] >= int(self.timetick+'000')):
            self.snap_first_tic = snap_id
            self.last_tick = snap_id
        # Apply penalties based on how late in the trading day it is
        if snap_list[0] >= int(self.date+'143800000'):
            punish_point = 3
        if snap_list[0] >= int(self.date+'145000000'):
            punish_point = 100
        if snap_list[0] >= int(self.date+'145300000'):
            punish_point = 10000000
        
        # Determine the order price based on the direction (buy/sell), predictive score, and penalties
        if self.direct == "buy":
            if snap_list[col_idx_map["AskPrice1"]] == 0:
                return False
            order_price = mid * (1 + 1 * cubic_func(self.pred_score_map[time_tic], self.transfer_map["a"], self.transfer_map["b"], self.transfer_map["c"], self.transfer_map["d"]) - (self.transfer_map["buy_take_threshold"] - punish_point) / 1e4)
            # Execute a buy order if the calculated order price is at or above the best ask price
            if order_price >= snap_list[col_idx_map["AskPrice1"]]:
                self.execute_order(snap_list, snap_id, "buy")
        elif self.direct == "sell":
            if snap_list[col_idx_map["BidPrice1"]] == 0:
                return False
            order_price = mid * (1 + 1 * cubic_func(self.pred_score_map[time_tic], self.transfer_map["a"], self.transfer_map["b"], self.transfer_map["c"], self.transfer_map["d"]) + (self.transfer_map["sell_take_threshold"] - punish_point) / 1e4)
            # Execute a sell order if the calculated order price is at or below the best bid price
            if order_price <= snap_list[col_idx_map["BidPrice1"]]:
                self.execute_order(snap_list, snap_id, "sell")

    def execute_order(self, snap_list, snap_id, direction):
        # Helper function to execute a trade and update the trading record
        trade_volume = min(self.leftvol, snap_list[col_idx_map[direction.capitalize() + "Vol1"]])  # Determine the trade volume
        trade_price = snap_list[col_idx_map[direction.capitalize() + "Price1"]]  # Determine the trade price
        self.traderecord.append([snap_list[0], trade_volume, trade_price])  # Record the trade
        self.leftvol -= trade_volume  # Update the remaining volume
        self.tradevol += trade_volume  # Update the total traded volume
        self.last_tick = copy.copy(snap_id)  # Update the last tick after execution

    def run_backtest_unit(self):
        # Runs the backtest for this parent order using the loaded snapshot data
        if (len(self.snap_sub) == 0) | (self.original_volume == 0):
            return {}  # Return empty if no data or no volume to trade
        for j in range(len(self.snap_sub)):
            self.update_start_acc()  # Update starting accumulated data
            self.update_last_acc()  # Update the last tick's accumulated data
            self.ontick(self.snap_sub[j], j)  # Process each tick

if __name__ == '__main__':
    date = sys.argv[1]
    bktst = Backtest("backtest_config_jyl.json",date)
    bktst.run_backtest_parent()
    bktst.save_result()

    


