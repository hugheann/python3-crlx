from datetime import datetime, timezone, timedelta
import matplotlib.pyplot as plt
import requests
from requests.compat import urljoin
import time

API_URL = "https://coriolix.sikuliaq.alaska.edu/api/"
REQ_DT_FMT = '%Y-%m-%d %H:%M:%S'

def get_recent_gps(hours):
    edt = datetime.now(timezone.utc).replace(tzinfo = None)
    bdt = edt - timedelta(hours = hours)
    params = {'date_after': bdt.strftime(REQ_DT_FMT),
              'date_before': edt.strftime(REQ_DT_FMT),
              'format': 'json'}
    url = urljoin(API_URL,'gnss_gga_bow')
    response = requests.get(url, params = params)
    return response.elapsed.total_seconds()



hours = range(1,73)

data = []
for hour in hours:
    avg = []
    for i in range(3):
        avg.append(get_recent_gps(hour))
        time.sleep(1)
    data.append(sum(avg)/3)
    print(data)





fig, ax = plt.subplots(1,1, figsize = (6,6), constrained_layout = True)

ax.plot(hours, data)
ax.set_xlabel('Last X Hours')
ax.set_ylabel('Average Response Time')