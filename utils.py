import numpy as np
from datahub_api import get


dates = get("datahub:/common/dates/data")
tickers_npy = get("datahub:/common/tickers/data")
volumes = get("datahub:/common/wind/archive/ASHAREEODPRICES/data?table=S_DQ_VOLUME")
use_cols = get("datahub:/high_frequency/stock/common/tonglian/snapshot/meta")["available_cols"]
col_idx_map = {}
for col_idx, col_name in enumerate(use_cols):
    col_idx_map[col_name] = col_idx

def cubic_func(x,a,b,c,d):
    return a*x**3+b*x**2+c*x+d

def get_vwap(df_tmp,order_volume,times = 10):
    df_tmp[:,col_idx_map["AccTurnover"]] = df_tmp[:,col_idx_map["AccTurnover"]] -df_tmp[0,col_idx_map["AccTurnover"]] 
    df_tmp[:,col_idx_map["AccVolume"]] = df_tmp[:,col_idx_map["AccVolume"]] -df_tmp[0,col_idx_map["AccVolume"]] 
    order_volume_adj = abs(order_volume)*times
    df_tmp_after = df_tmp[df_tmp[:,col_idx_map["AccVolume"]]>=order_volume_adj]
    if len(df_tmp_after)==0:
        return df_tmp[-1,col_idx_map["AccTurnover"]]/df_tmp[-1,col_idx_map["AccVolume"]]
    else:
        return df_tmp_after[0,col_idx_map["AccTurnover"]]/df_tmp_after[0,col_idx_map["AccVolume"]]
            