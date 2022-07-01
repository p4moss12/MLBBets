import warnings  # Ignore FutureWarnings cause by pybaseball for table.drop and also from df.mean
warnings.simplefilter(action='ignore', category=FutureWarning)
from pybaseball import pitching_stats_range, batting_stats_range
import pandas as pd
import typing
import tensorflow as tf
import tensorflow_addons as tfa
# For code completion
from tensorflow import keras
if typing.TYPE_CHECKING:
    from keras.api._v2 import keras
from sklearn.preprocessing import MultiLabelBinarizer
from datetime import datetime, timedelta
from scrape import matchups
import numpy as np
from tqdm import tqdm
import pickle


MODEL_PATH = 'saved_model/MLBModel'
BOOK = 'Bovada'
MARKETS = {398: '1H OU',
           402: 'OU',
           91: '1H ML',
           83: 'ML',
           401: 'Point Spread'}
TODAY = datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)  # current date
# TODAY = (datetime.today() - timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)  # yesterday

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


def make_preds():
    # Load model
    model = keras.models.load_model(MODEL_PATH)
    # Print summary
    print(model.summary())

    # Load Binarizer
    binarizer = pickle.load(open('binarizer.pkl', 'rb'))
    # print(f'Binarizer classes: {binarizer.classes_}')

    # Get input data for today
    X, games = get_input()

    print(f'Shape of todays input: {X.shape}')

    print('Making predictions...')
    preds = model.predict(X)
    preds[preds >= 0.9] = 1  # set to 1 if output from model is greater than 0.5 (model output uses sigmoid activation)
    preds[preds < 0.9] = 0  # set to 0 otherwise
    preds_ib = binarizer.inverse_transform(preds)  # get preds as strings

    print(f'MLBBets predictions for {TODAY.date()}:')
    for idx in range(len(preds)):
        print(f'{games[idx]}: {preds_ib[idx]}')


def get_input():
    X = []
    games = []  # list of strings to keep track of which game is which

    # Columns for what stats are relevant/desired from stats dataframes
    pitch_cols = ['Name', 'BB', 'SO', 'ERA', 'Str', 'StL', 'StS', 'GB/FB', 'LD', 'PU', 'WHIP', 'BAbip', 'SO9', 'SO/W']
    hit_cols = ['Name', 'BA', 'OBP', 'SLG', 'OPS']

    matchup_date = TODAY.date()  # get matchup date from given datetime as datetime object
    matchup_df = matchups(matchup_date.strftime('%Y-%m-%d'))  # get matchups on desired date
    print('MATCHUPS FOR TODAY: ')
    print(matchup_df.to_string())

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
    print('Getting stats up to today...')
    pitch_stats = pitching_stats_range(start_pitch, (matchup_date - timedelta(days=1)).strftime('%Y-%m-%d'))
    hit_stats = batting_stats_range(start_hit, (matchup_date - timedelta(days=1)).strftime('%Y-%m-%d'))

    # Deal with accented strings
    for key, value in REPLACE_CHARS.items():
        hit_stats['Name'] = hit_stats['Name'].str.replace(key, value)
        pitch_stats['Name'] = pitch_stats['Name'].str.replace(key, value)
    hit_stats['Name'] = hit_stats['Name'].str.title()
    pitch_stats['Name'] = pitch_stats['Name'].str.title()

    # Get input data
    total = len(matchup_df.index)  # total matchups
    print(f'Gathering todays input data for {total} games...')
    for row in matchup_df.itertuples():
        home_data = []
        away_data = []

        # Starting pitchers
        # Home
        # home_starter = row['Home Starter'].iloc[0]
        home_starter = getattr(row, '_6')  # _6 = home starter
        # home_team = row['Home'].iloc[0]
        home_team = getattr(row, 'Home')
        hs_stats = pitch_stats[pitch_stats['Name'].str.contains(home_starter, case=False)]
        hs_stats = hs_stats[pitch_cols]  # keep desired columns

        if hs_stats.empty:  # no stats on this starter
            print(f'Missing stats for {home_starter} for {home_team}')
            continue  # incomplete stats for game, cannot use, go to next row

        # Away
        # away_starter = row['Away Starter'].iloc[0]
        away_starter = getattr(row, '_7')  # _7 = away starter
        # away_team = row['Away'].iloc[0]
        away_team = getattr(row, 'Away')
        as_stats = pitch_stats[pitch_stats['Name'].str.contains(away_starter, case=False)]
        as_stats = as_stats[pitch_cols]  # keep desired columns

        if as_stats.empty:  # no stats on this starter
            print(f'Missing stats for {away_starter} for {away_team}')
            continue  # incomplete stats for game, cannot use, go to next row

        # Starting lineups
        # Home
        hl = list(row[8:17])  # difference here from get_input in get_data because row is a tuple with Index at row[0]
        hl = [x if x not in PLAYER_REPLACEMENTS else PLAYER_REPLACEMENTS[x] for x in hl]  # replace player names
        if not all(hl):  # check for empty strings in home lineup
            print(f'Incomplete lineup for {home_team}.')
            continue  # incomplete stats for game, cannot use, continue to next row

        hl_stats = hit_stats[
            hit_stats['Name'].str.contains('|'.join(hl), case=False)]  # get rows from stats containing lineup
        hl_stats = hl_stats[hit_cols]  # keep desired columns

        if hl_stats.empty or len(hl_stats.index) < 9:  # incomplete stats for lineup
            print(f'Incomplete lineup stats for {home_team}. '
                  f'Missing stats for '
                  f'{[s for s in hl if not any(s.casefold() in x.casefold() for x in hl_stats.Name.values)]}')
            continue  # incomplete stats for game, cannot use, continue to next row

        # Away
        al = list(row[17:])  # difference here from get_input in get_data because row is a tuple with Index at row[0]
        al = [x if x not in PLAYER_REPLACEMENTS else PLAYER_REPLACEMENTS[x] for x in al]  # replace player names

        if not all(al):  # check for empty strings in away lineup
            print(f'Incomplete lineup for {away_team}.')
            continue  # incomplete stats for game, cannot use, continue to next row

        al_stats = hit_stats[
            hit_stats['Name'].str.contains('|'.join(al), case=False)]  # get rows from stats containing lineup
        al_stats = al_stats[hit_cols]  # keep desired columns

        if al_stats.empty or len(al_stats.index) < 9:  # incomplete stats for lineup
            print(f'Incomplete lineup stats for {away_team}. '
                  f'Missing stats for '
                  f'{[s for s in al if not any(s.casefold() in x.casefold() for x in al_stats.Name.values)]}')
            continue  # incomplete stats for game, cannot use, continue to next row

        # Check for NaN pitching stats
        # Home
        if pd.isna(hs_stats['SO/W'].iloc[0]):  # div by 0
            hs_stats['SO/W'].iat[0] = hs_stats['SO'].iat[0]  # set to their current strikeout total

        # Away
        if pd.isna(as_stats['SO/W'].iloc[0]):  # div by 0
            as_stats['SO/W'].iat[0] = as_stats['SO'].iat[0]  # set to their current strikeout total

        # Append stats to lists
        # Home
        home_data.extend(hs_stats.iloc[0, 3:].tolist())  # Pitcher
        home_data.extend(hl_stats.mean().tolist())  # Lineup averages

        # Away
        away_data.extend(as_stats.iloc[0, 3:].tolist())  # Pitcher
        away_data.extend(al_stats.mean().tolist())  # Lineup averages

        data = home_data + away_data
        game = away_team + '@' + home_team + ' ' + matchup_date.strftime('%Y-%m-%d')
        X.append(data)
        games.append(game)

    X = np.array(X)

    return X, games


def main():
    make_preds()


if __name__ == '__main__':
    main()
