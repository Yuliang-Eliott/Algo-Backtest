import time
import os
from ai_queue_client.rancher_job_client import run
import datetime
import wy_cms_api.wycluster.wyjob as wyjob
from datahub_api import get
all_dates = get("datahub:/common/dates/data")
dateslist = all_dates[(all_dates>='20231220')&(all_dates<='20240112')]

for date in dateslist:
    command = f"python /scratch/wy_dev/develop_containers_home/fany/execution/git/algo_backtest/AlphaParentOrderBTGan.py {date}"
    jobconfig = {
        "jobname": f"lgb"+datetime.datetime.now().strftime("%Y%m%d%H%M%S%f"),
        "namespace": "fany",
        "gpu_mig_type": 10,
        "cpu": 1,
        "mem": 300,
        "gpu": 0,
        "command": command,
        "image": "registry.wy.com:5000/wydevbase:3.0",
        "job_type": "normal"

    }

    status = run(**jobconfig)
    print(status)
    print(status.message)
    # 表示调度是否成功
    print(status.success)
    time.sleep(0.1)

