import requests
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime, timezone
from collections import defaultdict

bref = 'https://www.baseball-reference.com/leagues/MLB-standings.shtml'
lineups_mlb = 'https://www.mlb.com/starting-lineups/'
# Dictionary to replace spanish names that appear randomly in requests, also replace D-backs
replacements = {
    'Atléticos': 'Athletics',
    'Azulejos': 'Blue Jays',
    'Bravos': 'Braves',
    'Tigres': 'Tigers',
    'Reales': 'Royals',
    'Gigantes': 'Giants',
    'Nacionales': 'Nationals',
    'Piratas': 'Pirates',
    'Cardenales': 'Cardinals',
    'Cerveceros': 'Brewers',
    'Cachorros': 'Cubs',
    'Marineros': 'Mariners',
    'Rojos': 'Reds',
    'D-backs': 'Diamondbacks'
}


def exp_standings():
    pass


def matchups(date):
    """
    Function to acquire the starting pitchers on a given date; can improve to scrape starting lineups using same webpage
    :param date: Date of the games in question as a string
    :return: Dataframe of each of the dates matchups and the starting pitchers
    """
    df = pd.DataFrame(columns=['Matchup', 'gamepk', 'datetime', 'Home', 'Away', 'Home Starter', 'Away Starter',
                               'Home C', 'Home 1B', 'Home 2B', 'Home 3B', 'Home SS', 'Home LF', 'Home CF', 'Home RF',
                               'Home DH',
                               'Away C', 'Away 1B', 'Away 2B', 'Away 3B', 'Away SS', 'Away LF', 'Away CF', 'Away RF',
                               'Away DH'])  # page dataframe

    # Beautiful soup
    page = requests.get(lineups_mlb + date)
    soup = BeautifulSoup(page.content, "html.parser")

    # Find matchups
    matchups = soup.find_all('div', class_='starting-lineups__matchup')
    # print(len(matchups))

    # Populate dataframe with each matchup
    for matchup in matchups:
        # Get gamepk
        gamepk = matchup['data-gamepk']

        # Get datetime; time from site is UTC, convert to local time
        time_el = matchup.find_all('time')[0]  # get time element
        dt = datetime.fromisoformat(time_el['datetime'].strip('Z'))  # get utc datetime object in iso format
        dt = dt.replace(tzinfo=timezone.utc).astimezone(tz=None)  # convert to local time
        dt = dt.isoformat()  # return datetime to iso format

        # Get home team
        home_el = matchup.find_all('span',
                                   attrs={'class': 'starting-lineups__team-name starting-lineups__team-name--home'})
        home_team = home_el[0].text.strip()  # strip to get rid of extra white space
        if home_team in replacements:
            home_team = replacements[home_team]
        # home_team = home_el[0].find_all('a')[0].text.strip()  # strip to get rid of extra white space

        # Get away team
        away_el = matchup.find_all('span',
                                   attrs={'class': 'starting-lineups__team-name starting-lineups__team-name--away'})
        away_team = away_el[0].text.strip()  # strip to get rid of extra white space
        if away_team in replacements:
            away_team = replacements[away_team]
        # away_team = away_el[0].find_all('a')[0].text.strip()  # strip to get rid of extra white space

        # Get pitchers
        pitchers = matchup.find_all('div',
                                    attrs={'class': 'starting-lineups__pitcher-name'})
        home_pitcher = pitchers[1].text.strip()  # strip to get rid of extra white space
        away_pitcher = pitchers[0].text.strip()  # strip to get rid of extra white space

        # Get lineups
        awaylineup_dict = defaultdict(str)
        homelineup_dict = defaultdict(str)
        # Away
        away_lineup = matchup.find_all('ol',
                                    attrs={'class': 'starting-lineups__team starting-lineups__team--away'})[0]
        away_players = away_lineup.find_all('li',
                                    attrs={'class': 'starting-lineups__player'})
        for player in away_players:
            position = str(player.find_all('span')[0].text)[4:].strip()  # player position
            name = player.find_all('a')[0].text
            awaylineup_dict[position] = name

        # Home
        home_lineup = matchup.find_all('ol',
                                    attrs={'class': 'starting-lineups__team starting-lineups__team--home'})[0]
        home_players = home_lineup.find_all('li',
                                    attrs={'class': 'starting-lineups__player'})
        for player in home_players:
            position = str(player.find_all('span')[0].text)[4:].strip()  # player position
            name = player.find_all('a')[0].text
            homelineup_dict[position] = name

        # Populate full matchup string
        matchup_str = away_team + ' @ ' + home_team

        # print(f'Away pitcher: {away_pitcher}')
        # print(f'Home pitcher: {home_pitcher}')
        # print(away_team)
        # print(home_team)
        # print(matchup_str)

        # Populate dataframe
        # df = df.append({'Matchup': matchup_str,
        #            'gamepk': gamepk,
        #            'Home': home_team,
        #            'Away': away_team,
        #            'Home Starter': home_pitcher,
        #            'Away Starter': away_pitcher}, ignore_index=True)
        vals = {'Matchup': matchup_str,
                'gamepk': gamepk,
                'datetime': dt,
                'Home': home_team,
                'Away': away_team,
                'Home Starter': home_pitcher,
                'Away Starter': away_pitcher,
                'Home C': homelineup_dict['C'],
                'Home 1B': homelineup_dict['1B'],
                'Home 2B': homelineup_dict['2B'],
                'Home 3B': homelineup_dict['3B'],
                'Home SS': homelineup_dict['SS'],
                'Home LF': homelineup_dict['LF'],
                'Home CF': homelineup_dict['CF'],
                'Home RF': homelineup_dict['RF'],
                'Home DH': homelineup_dict['DH'] if homelineup_dict['DH'] else homelineup_dict['P'],  # Ohtani lol
                'Away C': awaylineup_dict['C'],
                'Away 1B': awaylineup_dict['1B'],
                'Away 2B': awaylineup_dict['2B'],
                'Away 3B': awaylineup_dict['3B'],
                'Away SS': awaylineup_dict['SS'],
                'Away LF': awaylineup_dict['LF'],
                'Away CF': awaylineup_dict['CF'],
                'Away RF': awaylineup_dict['RF'],
                'Away DH': awaylineup_dict['DH'] if awaylineup_dict['DH'] else awaylineup_dict['P']  # Ohtani lol
                }
        df = pd.concat([df, pd.DataFrame(vals, index=[0])], ignore_index=True)

    return df


def main():
    # Print dataframe of today's starting pitchers
    # data = starters(datetime.today().strftime('%Y-%m-%d'))
    data = matchups('2022-04-19')
    print(data.to_string())


if __name__ == '__main__':
    main()
