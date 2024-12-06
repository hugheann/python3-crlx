from datetime import datetime, timezone, timedelta
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
import numpy as np

from crlx import SIKULIAQ




def main():
    SKQ = SIKULIAQ(verify=True, verbose=True)
    edt = datetime.now(timezone.utc)
    bdt = edt - timedelta(days=3)
    gilla = SKQ.get_fwd_mast_wind_a(bdt, edt)




if __name__ == "__main__":
    main()