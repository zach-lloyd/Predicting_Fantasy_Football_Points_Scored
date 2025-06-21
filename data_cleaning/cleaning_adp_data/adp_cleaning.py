import re
import pandas as pd

years = range(2012, 2025)
list_of_dataframes = []
# Read ADP info for each year into a separate dataframe.
for year in years:
    # I had issues with the encoding of the CSV files after pasting in the data, so I ended up
    # having to iterate over a few different encodings and try each of those to be able to read
    # the files.
    for enc in ['cp1252', 'latin-1', 'iso-8859-1']:
        try:
            df = pd.read_csv('ADP With Position/' + str(year) + ' PPR ADP.csv', encoding=enc)
            break                
        except UnicodeDecodeError:
            pass
    # The ADP files do not contain a column specifying the season that the data is from, so add
    # it here.
    df['Year'] = year
    # Rename position column to ensure conformity with statistics file (this will be one of the
    # keys that I will join on).
    if 'POS' in df.columns:
        df.rename(columns = {'POS':'Pos'}, inplace = True)
        
    list_of_dataframes.append(df)

final_df = pd.concat(list_of_dataframes).reset_index(drop=True)
# Many of the player names contain extraneous data like team names and injury designations in 
# parentheses at the end of their names, so use Regex to strip out text in parentheses from 
# player names.
stripped_names = []

for idx, row in final_df.iterrows():
    stripped_names.append(re.sub(r"\(.*\)", "", str(row['PLAYER'])))
    
final_df['PLAYER'] = stripped_names


def strip_whitespace(column_name, df):
    """Strip trailing whitespace from the specified column

    Args:
        df: A Pandas dataframe.
        column_name: A string representing the name of the column to be cleaned.

    Returns:
        Null. Sets the specified column equal to the cleaned data.
    """
    stripped_whitespace = []
    for idx, row in df.iterrows():
        stripped_whitespace.append(row[column_name].strip())
    df[column_name] = stripped_whitespace


strip_whitespace('PLAYER', final_df)
# Some of the ADP data contains injuries and other extraneous characters appended
# to player names that are not in parentheses, so the previous code for stripping
# text in parentheses will not be sufficient. Best approach I could think of here
# was to go through and list out each unique instance of this that I saw and then
# define some functions to strip these specific strings from player names.
injuries = ['Knee Surgery', 'Hip Surgery', 'Torn ACL', 'Arrested',
            'Torn Achilles', 'Dislocated Hip', 'Broken Leg', 'COV-IR',
            'O', 'IR', 'AK', 'HU', 'N', 'C', 'S', 'WA', 'K', 'CI', 'JA',
            'MI', 'DE', 'TE']


def strip_string(str_to_strip, slice_of_str):
    """Remove the specified slice from the specified string.

    Args:
        str_to_strip: The string to be cleaned.
        slice_of_str: The slice of text to be removed from the string.

    Returns:
        str_to_strip with slice_of_str removed from the end of it.
    """
    if str_to_strip.endswith(slice_of_str):
        s1 = slice(-len(slice_of_str))
        str_to_strip = str_to_strip[s1]
        
    return str_to_strip 


def strip_df(string, df):
    """Iterate over player names and remove the specified string.

    Args:
        string: The string to be stripped from player names.
        df: A Pandas dataframe containing player names.

    Returns:
        Null. Sets the player name column to contain the cleaned names.
    """
    stripped_entries = []
    
    for idx, row in df.iterrows():
        stripped_entries.append(strip_string(row['PLAYER'], string))
        
    df['PLAYER'] = stripped_entries


for injury in injuries:
    strip_df(injury, final_df)
# Some player names also contain bye week information, represented by a comma
# followed by an integer representing the player's bye week in the given season.
# The below code stores player names in a temporary dataframe and strips this information.
temp = final_df['PLAYER']
temp_series = temp.squeeze()
final_df['PLAYER'] = temp_series.str.rstrip(', 0123456789')
# Given that whitespace was stripped above, there shouldn't be any, but given the amount of 
# cleaning that has been done to player names, strip whitespace again as a safety check.
strip_whitespace('PLAYER', final_df)
# Player names also sometimes contain their team abbreviation, so that needs to be stripped too. 
# Note that 'O' is included because 'AK' gets stripped from 'OAK' above in injuries.
teams = ['PHI', 'DAL', 'WAS', 'NYG', 'DET', 'GB', 'MIN', 'CHI', 'NO',
         'TB', 'CAR', 'ATL', 'ARI', 'SEA', 'SF', 'LAR', 'STL', 'NE',
         'MIA', 'NYJ', 'BUF', 'CIN', 'CLE', 'PIT', 'BAL', 'TEN', 'HOU',
         'IND', 'LA', 'JAC', 'KC', 'DEN', 'SD', 'LAC', 'OAK', 'LV', 'O', 
         'FA']

for team in teams:
    strip_df(team, final_df)
# Strip any leftover underscores, and to be safe, strip whitespace one more time. 
strip_df('_', final_df)
strip_whitespace('PLAYER', final_df)
# Rename the ranking column for clarity because there is already a 'Rank' column in the statistics
# sheet.
final_df.rename(columns={'Rank': 'Ovr_ADP_Rk'}, inplace=True)
# In the ADP sheets, player positions and their positional ranks are combined in the format 'QB7',
# 'WR13', etc., so split position and positional rank into separate columns.
final_df[['Pos', 'Pos_ADP_Rk']] = final_df['Pos'].str.extract(r"([A-Z]+)\s*(\d+)", expand=True)
# Some of the sheets have extraneous rows where the Player name is NaN or is just a single
# letter like P or Q. The below code removes these rows.
final_df = (
    final_df.loc[~final_df['PLAYER'].str.fullmatch(r"[A-Za-z]", na=False)]
      .dropna(subset=['PLAYER', 'ADP'])
      .copy()
)

final_df.to_csv('adp.csv')