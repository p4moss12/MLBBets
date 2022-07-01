import pandas
from pybaseball import statcast_batter, playerid_lookup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def main():
    #
    # h = statcast_batter(start_dt='2022-06-09', end_dt='2022-06-14', player_id=519317)
    #
    # print(h.head(2).to_string())
    tb = pandas.read_html('https://baseballsavant.mlb.com/probable-pitchers/?date=2022-06-15')
    print(len(tb))
    print(tb[0].to_string())


if __name__ == '__main__':
    main()
