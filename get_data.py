# https://github.com/jldbc/pybaseball
# https://github.com/JeMorriso/PySBR
import re
import warnings  # Ignore FutureWarnings cause by pybaseball for table.drop and also from df.mean
warnings.simplefilter(action='ignore', category=FutureWarning)
from pybaseball import pitching_stats_range, batting_stats_range
import pandas as pd
import pysbr  # pip3_install requests_toolbelt
from datetime import datetime, timedelta
from scrape import matchups
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm
import sys
import subprocess

BOOK = 'Bovada'
MARKETS = {398: '1H OU',
           402: 'OU',
           91: '1H ML',
           83: 'ML',
           401: 'Point Spread'}

# # To append new data to existing data
# # LOAD = True
# # START_DATE = datetime.strptime('2022-06-11', '%Y-%m-%d')  # start date, last date new data was gathered for
# # END_DATE = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)  # current date
#
# # To gather new data from beginning of season to a date or to current date
# LOAD = False
# START_DATE = datetime.strptime('2022-04-08', '%Y-%m-%d')  # day after opening day, set load to false
# # END_DATE = datetime.strptime('2022-04-10', '%Y-%m-%d')  # date get training data up to
# END_DATE = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)  # current date

# Names to replace that show up from matchup scrape to match hit_stats
PLAYER_REPLACEMENTS = {
    'Albert Almora Jr.': 'Albert Almora',
    'Michael Harris II': 'Michael Harris',
    'Cal Mitchell': 'Calvin Mitchell',
    'Josh H. Smith': 'Josh Smith',
    'Bobby Witt Jr.': 'Bobby Witt'
}

# Unicode characters to replace in hitting/pitching stats for consistency with matchup scrape
REPLACE_CHARS = {
    '├®': 'e',
    'ă©': 'e',
    'é': 'e',
    'Á': 'A',
    '├ü': 'a',
    '├Ī': 'a',
    'á': 'a',
    '├│': 'o',
    'ó': 'o',
    '├Ł': 'i',
    'í': 'i',
    'ñ': 'n',
    '├▒': 'n'
}


def get_data(start, end, load=True):
    # Prevent sleep
    if 'darwin' in sys.platform:
        print('Running \'caffeinate\' on MacOSX to prevent the system from sleeping')
        subprocess.Popen('caffeinate')

    if load:
        # Data and label lists
        data = np.load('data.npy', allow_pickle=True).tolist()
        labels = np.load('labels.npy', allow_pickle=True).tolist()

        print(f'Length of loaded data: {len(data)}')
        print(f'Length of loaded labels: {len(labels)}')

    else:
        # Data and label lists
        data = []
        labels = []

    # Pysbr configs
    mlb = pysbr.MLB()  # MLB config class
    sb = pysbr.Sportsbook()  # Sportsbook config class

    # Get events up to range
    print('Gathering Events...')
    events = pysbr.EventsByDateRange(mlb.league_id, start, end)  # get events in range
    # print(events.dataframe().to_string())

    # Dataframe for all lines for each event in the desired markets
    cl = pysbr.CurrentLines(events.ids(), mlb.market_ids(list(MARKETS.keys())), sb.ids([BOOK])).dataframe(events)
    # print(cl.to_string())  # Print current lines dataframe

    # CLEANING LINES DATA
    print('Cleaning data...')
    # Dict to store event ids and their counts
    e_ids = defaultdict(int)
    # Only keep events with all/clean data available (10 lines, 10 rows for each event)
    for index, row in cl.iterrows():
        e_ids[row['event id']] += 1

    cl_clean = pd.DataFrame(columns=cl.columns)  # Clean dataframe that will be used for data
    for index, row in cl.iterrows():
        if e_ids[row['event id']] == 10:  # if exactly 10 lines available for this ID
            cl_clean = pd.concat([cl_clean, cl.iloc[[index]]], ignore_index=True)  # add row to clean dataframe

    # Drop rows with NaN outcomes
    cl_clean = cl_clean.dropna(subset=['profit'])  # drop rows with NaN values
    cl_clean = cl_clean.reset_index(drop=True)  # reset the index

    print('Clean CL Dataframe:')
    print(cl_clean.to_string())  # print clean dataframe
    print(f'Number of games with 10 Lines: {len(cl_clean.index) / 10}')  # 10 lines for each game

    # Iterate over the rows and add training data
    count = 0
    cur_label = []

    # Variables for input data
    home_in = []
    away_in = []
    got_in = False  # whether input data has been gathered

    # Dictionary used to identify and insert push bets, keys are markets, values are index in output list
    # Once a winner is assigned, the value for that market is set to True, if a winner for the market is never assigned,
    # push is inserted at the index
    push_dict = {
        'ML': 0,
        '1H ML': 1,
        '1H OU': 2,
        'Point Spread': 3,
        'OU': 4
    }
    print('Gathering training data...')
    total = len(cl_clean.index)  # total iterations for tqdm pbar
    for row in tqdm(cl_clean.itertuples(), total=total):

        if count == 10:  # End of event lines
            count = 0

            if got_in and home_in and away_in and cur_label:
                # Check for pushes
                for market, val in push_dict.items():
                    if type(val) == int:  # if val is still an index instead of True
                        cur_label.insert(val, market + ' push')

                # Labels
                labels.append(cur_label.copy())  # append event labels to label list
                cur_label.clear()  # reset cur_label
                push_dict = {  # reset push_dict
                    'ML': 0,
                    '1H ML': 1,
                    '1H OU': 2,
                    'Point Spread': 3,
                    'OU': 4
                }

                # Input data
                in_data = home_in + away_in  # combine home and away inputs
                data.append(in_data)  # append input data list to data
                home_in.clear()  # reset home input
                away_in.clear()  # reset away input
                got_in = False  # reset input bool

            else:  # input not gathered, reset lists/dicts without appending data/labels
                # Labels
                cur_label.clear()  # reset cur_label
                push_dict = {  # reset push_dict
                    'ML': 0,
                    '1H ML': 1,
                    '1H OU': 2,
                    'Point Spread': 3,
                    'OU': 4
                }

                # Input data
                home_in = []  # reset home input
                away_in = []  # reset away input
                got_in = False  # reset input bool

        # Get market results and input data
        if getattr(row, '_1') == 398:  # 1H OU  # _1 = market id
            if getattr(row, 'result') == 'W':
                cur_label.append(MARKETS[getattr(row, '_1')] + ' ' + getattr(row, 'participant'))  # append result
                push_dict[MARKETS[getattr(row, '_1')]] = True  # Confirm no push for this market
        elif getattr(row, '_1') == 402:  # OU  # _1 = market id
            if getattr(row, 'result') == 'W':
                cur_label.append(MARKETS[getattr(row, '_1')] + ' ' + getattr(row, 'participant'))  # append result
                push_dict[MARKETS[getattr(row, '_1')]] = True  # Confirm no push for this market
        else:  # 1H ML, ML, Spread
            # Decipher whether current team is the home or away team
            is_home = False
            event_df = pysbr.EventsByEventIds(getattr(row, '_2')).dataframe()  # _2 = event id
            if getattr(row, '_5') == event_df.at[0, 'participants.1.participant id']:  # _5 = participant id
                is_home = event_df.at[0, 'participants.1.is home']
            elif getattr(row, '_5') == event_df.at[0, 'participants.2.participant id']:  # _5 = participant id
                is_home = event_df.at[0, 'participants.2.is home']

            # Get input data for home/away team if not already gathered
            if not got_in:
                # gather input data given team and whether team is home or away (both home and away data)
                e = pysbr.EventsByEventIds(getattr(row, '_2')).list()[0]  # _2 = event id; get event id for game
                dt = e['datetime']  # get datetime for game, must use event for matching start time
                team = getattr(row, '_17')  # _17 = participant full name

                if 'Indians' in team:  # check for Cleveland which isn't updated
                    team = 'Cleveland Guardians'

                home_in, away_in = get_input_data(dt, team, is_home)

                # Check if input values are valid
                if (not home_in) or (not away_in):  # Invalid input data available
                    got_in = False
                    count += 1
                    continue  # continue to next row
                else:
                    got_in = True
                # break

            # If L, continue to next iteration, no info needed
            if getattr(row, 'result') == 'L':
                count += 1
                continue  # Loss result, go to next row

            # Otherwise...
            # Append market result; _1 = market id
            if is_home:
                cur_label.append(MARKETS[getattr(row, '_1')] + ' home win')  # append home win as result for market
                push_dict[MARKETS[getattr(row, '_1')]] = True  # Confirm no push for ths market
            else:
                cur_label.append(MARKETS[getattr(row, '_1')] + ' away win')  # append away win as result for market
                push_dict[MARKETS[getattr(row, '_1')]] = True  # Confirm no push for this market

        count += 1
    # Append final event
    if home_in and away_in and got_in:
        in_data = home_in + away_in  # combine home and away inputs
        if in_data is not None and cur_label is not None:
            # Data
            data.append(in_data)  # append input data list to data

            # Labels
            # Check for pushes
            for market, val in push_dict.items():
                if type(val) == int:  # if val is still an index instead of True
                    cur_label.insert(val, market + ' push')
            labels.append(cur_label)

        else:
            print('In data or cur_label was none at the end.')

    print(f'Length of data after additions: {len(data)}')
    print(f'Length of labels after additions: {len(labels)}')

    # Np array
    data = np.array(data)
    labels = np.array(labels)

    print(f'Shape of data after additions: {data.shape}')
    print(f'Shape of labels after additions: {labels.shape}')

    # Save data and labels
    np.save('data', data, allow_pickle=True)
    np.save('labels', labels, allow_pickle=True)


def get_input_data(dt, team, is_home):
    home_data = []
    away_data = []

    # Columns for what stats are relevant/desired from stats dataframes
    pitch_cols = ['Name', 'BB', 'SO', 'ERA', 'Str', 'StL', 'StS', 'GB/FB', 'LD', 'PU', 'WHIP', 'BAbip', 'SO9', 'SO/W']
    hit_cols = ['Name', 'BA', 'OBP', 'SLG', 'OPS']

    matchup_date = datetime.fromisoformat(dt).date()  # get matchup date from given datetime as datetime object
    matchup_df = matchups(matchup_date.strftime('%Y-%m-%d'))  # get matchups on desired date

    # Stats range start dates
    # Hitting stats start
    if matchup_date < datetime.strptime('2022-04-22', '%Y-%m-%d').date():  # If date within 15 days of opening day
        start_hit = '2022-04-07'  # opening day
    else:
        start_hit = (matchup_date - timedelta(days=15)).strftime('%Y-%m-%d')  # last 15 days prior to game

    # Pitching stats start
    if matchup_date < datetime.strptime('2022-05-07', '%Y-%m-%d').date():  # If date within 30 days of opening day
        start_pitch = '2022-04-07'
    else:
        start_pitch = (matchup_date - timedelta(days=30)).strftime('%Y-%m-%d')  # last 30 days prior to game

    # Get pitching/batting data up to the date of the game, subtract 1 from matchup date
    pitch_stats = pitching_stats_range(start_pitch, (matchup_date - timedelta(days=1)).strftime('%Y-%m-%d'))
    hit_stats = batting_stats_range(start_hit, (matchup_date - timedelta(days=1)).strftime('%Y-%m-%d'))

    # Deal with accented strings
    for key, value in REPLACE_CHARS.items():
        hit_stats['Name'] = hit_stats['Name'].str.replace(key, value)
        pitch_stats['Name'] = pitch_stats['Name'].str.replace(key, value)
    hit_stats['Name'] = hit_stats['Name'].str.title()
    pitch_stats['Name'] = pitch_stats['Name'].str.title()

    # Find matchup
    if is_home:  # Find exact matchup via datetime and give data given home team
        # Find exact matchup
        # Lambda checks if team in col is in given team string, second half of & checks the datetime of the matchup
        row = matchup_df[(matchup_df['Home'].apply(lambda x: x in team)) & (matchup_df['datetime'] == dt)]
    else:  # Find exact matchup via datetime and give data given away team
        # Find exact matchup
        # Lambda checks if team in col is in given team string, second half of & checks the datetime of the matchup
        row = matchup_df[(matchup_df['Away'].apply(lambda x: x in team)) & (matchup_df['datetime'] == dt)]

    if row.empty:
        print(f'Matchup for {team} on {matchup_date} when given {dt} not found. Ishome: {is_home}')
        print(matchup_df.to_string())
        return None, None

    # Starting pitchers
    # Home
    home_starter = row['Home Starter'].iloc[0]
    if not home_starter:  # Empty string for home starter
        return None, None

    hs_stats = pitch_stats[pitch_stats['Name'].str.contains(home_starter, case=False)]
    hs_stats = hs_stats[pitch_cols]  # keep desired columns

    if hs_stats.empty:  # no stats on this starter
        return None, None  # incomplete stats for game, cannot use

    # Away
    away_starter = row['Away Starter'].iloc[0]
    if not away_starter:  # empty string for away starter
        return None, None

    as_stats = pitch_stats[pitch_stats['Name'].str.contains(away_starter, case=False)]
    as_stats = as_stats[pitch_cols]  # keep desired columns

    if as_stats.empty:  # no stats on this starter
        return None, None  # incomplete stats for game, cannot use

    # Starting lineups
    # Home
    hl = row.iloc[0, 7:16].to_list()  # home lineup
    hl = [x if x not in PLAYER_REPLACEMENTS else PLAYER_REPLACEMENTS[x] for x in hl]  # replace player names
    if not all(hl):  # check for empty strings in home lineup
        return None, None

    hl_stats = hit_stats[hit_stats['Name'].str.contains('|'.join(hl), case=False)]  # rows from stats containing lineup
    hl_stats = hl_stats[hit_cols]  # keep desired columns

    if hl_stats.empty or len(hl_stats.index) < 9:  # incomplete stats for lineup
        return None, None  # incomplete stats for game, cannot use

    # Away
    al = row.iloc[0, 16:].to_list()  # away lineup
    al = [x if x not in PLAYER_REPLACEMENTS else PLAYER_REPLACEMENTS[x] for x in al]  # replace player names
    if not all(al):  # check for empty strings in away lineup
        return None, None

    al_stats = hit_stats[hit_stats['Name'].str.contains('|'.join(al), case=False)]  # rows from stats containing lineup
    al_stats = al_stats[hit_cols]  # keep desired columns

    if al_stats.empty or len(al_stats.index) < 9:  # incomplete stats for lineup
        return None, None  # incomplete stats for game, cannot use

    # Check for NaN pitching stats
    # Home
    if pd.isna(hs_stats['SO/W'].iloc[0]):  # div by 0
        hs_stats['SO/W'].iat[0] = hs_stats['SO'].iat[0]  # set to their current strikeout total

    # Away
    if pd.isna(as_stats['SO/W'].iloc[0]):  # div by 0
        as_stats['SO/W'].iat[0] = as_stats['SO'].iat[0]  # set to their current strikeout total

    # Append stats to return lists
    # Home
    home_data.extend(hs_stats.iloc[0, 3:].tolist())  # Pitcher
    home_data.extend(hl_stats.mean().tolist())  # Lineup averages

    # Away
    away_data.extend(as_stats.iloc[0, 3:].tolist())  # Pitcher
    away_data.extend(al_stats.mean().tolist())  # Lineup averages

    return home_data, away_data


def load_data():
    # Load data from saved csv
    pass


def main():
    ans = input('Append to existing data? Type \'n\' if you wish to gather new data starting on Opening Day.(Y/n): ')

    if ans.casefold() == 'Y'.casefold() or ans.casefold() == 'yes'.casefold():
        # To append new data to existing data
        while True:
            date = input('Enter the start date for the new data (YYYY-MM-DD): ')
            if not re.match('^\d{4}\-(0[1-9]|1[012])\-(0[1-9]|[12][0-9]|3[01])$', date):  # check for proper date format
                print('Error. Please enter date in YYYY-MM-DD format.')
            else:
                start_date = datetime.strptime(date, '%Y-%m-%d')  # start date, last date new data was gathered for

                if start_date.date() < datetime.strptime('2022-04-08', '%Y-%m-%d').date():
                    print('Error. Please enter a date after Opening Day 2022 (2022-04-07)')
                else:
                    break

        load = True
        end_date = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)  # current date

    elif ans.casefold() == 'N'.casefold() or ans.casefold() == 'no'.casefold():
        # To gather new data from beginning of season to a date or to current date
        load = False
        start_date = datetime.strptime('2022-04-08', '%Y-%m-%d')  # day after opening day, set load to false
        # END_DATE = datetime.strptime('2022-04-10', '%Y-%m-%d')  # date get training data up to
        end_date = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)  # current date

    else:
        print('Invalid response. Exiting...')
        exit(0)

    get_data(start_date, end_date, load=load)  # if START_DATE is the beginning of the season, set load to false


if __name__ == '__main__':
    main()
