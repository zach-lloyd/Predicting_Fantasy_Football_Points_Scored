import pandas as pd
from functools import reduce


def remove_duplicates(df):
    """Remove duplicate rows of players who played for more than one team in a season. 

    Args:
        df: A Pandas dataframe with NFL player stats.
    
    Returns:
        A revised version of the dataframe that removes duplicate rows for players who 
        played for more than one team in a single season and keeps just the aggregate 
        row showing their full stats for that season.
    """
    # PFR handles players playing for multiple teams in the same season by listing the 
    # player's stats with each team, as well as their aggregate stats for the season in 
    # a separate row with their team listed as '2TM', '3TM', etc.
    agg_pat   = r'^(?:[234]TM|TOT)$'
    # Track whether a player played for multiple teams in the same year.
    df['is_agg'] = df['Team'].str.match(agg_pat)
    # Sort by is_agg in descending order so for any players who played for more than
    # one team, the aggregate stats row is the first row listed for that player,
    # then drop the other, team-specific rows for that player.
    cleaned_df = (df
         .sort_values('is_agg', ascending=False)      
         .drop_duplicates(subset=['Player', 'Season', 'Pos'], keep='first')
         .drop(columns='is_agg') # Can remove the is_agg column after this.                  
         .reset_index(drop=True))
    
    return cleaned_df


def join_sheets(pass_csv, rush_csv, recv_csv):
    """Joins the passing, rushing, and receiving stat sheets for a given season.

    Args:
        pass_csv: A csv pulled from Pro Football Reference with passing stats from an 
                  NFL season.
        rush_csv: A csv pulled from Pro Football Reference with rushing stats from an 
                  NFL season.
        recv_csv: A csv pulled from Pro Football Reference with receiving stats from an 
                  NFL season.
    
    Returns:
        A Pandas dataframe with the three csv files combined into one sheet.
    """
    pass_df = pd.read_csv(pass_csv)
    rush_df = pd.read_csv(rush_csv)
    recv_df = pd.read_csv(recv_csv)
    # For some reason the Season column in the rushing and receiving tables got
    # added as 'Season_', but the same did not happen in the passing tables.
    rush_df.rename(columns={'Season_': 'Season'}, inplace=True)
    recv_df.rename(columns={'Season_': 'Season'}, inplace=True)

    pass_cols_to_rename = [
        'Att', 'Yds', 'TD', 'TD%', '1D', 'Succ%', 'Lng', 'Y/A', 'AY/A', 'Y/C', 
        'Y/G', 'NY/A', 'ANY/A'
    ]
    # Rename some generically-named columns in the passing dataframe to avoid 
    # confusion. This is not necessary for the rushing and receiving tables
    # because they have column names of the format 'Rushing_{stat}'/'Receiving_{stat}'.
    for col in pass_df.columns:
        if col == 'Yds.1':
            pass_df.rename(columns={col: 'SkYdsLst'}, inplace=True)
        elif col in pass_cols_to_rename:
            pass_df.rename(columns={col: f'Pass_{col}'}, inplace=True)
        else:
            continue

    dfs = [
        remove_duplicates(pass_df), 
        remove_duplicates(rush_df), 
        remove_duplicates(recv_df)
    ]

    keys  = ['Player', 'Season', 'Team', 'Pos', 'Age', 'GS', 'G', 'Awards']
    combined_df  = reduce(lambda left,right: pd.merge(left, right, how='outer', on=keys), dfs)
    # Handle duplicate fumble columns from the rushing and receiving sheets. Joining on 
    # this column doesn't work because the passing sheet doesn't have a Fumbles column.
    if 'Fmb_x' in combined_df.columns:
        combined_df.rename(columns={'Fmb_x': 'Fumbles'}, inplace=True)
        combined_df = combined_df.drop(columns='Fmb_y')
    else:
        combined_df['Fumbles'] = pd.NA
        
    return combined_df


if __name__ == "__main__":
    start_year = 1932
    end_year = 2024
    sheets = []
    
    for yr in range(start_year, end_year + 1):
        passing_stats = f'passing_stats/passing_stats_{yr}.csv'
        rushing_stats = f'rushing_stats/rushing_stats_{yr}.csv'
        receiving_stats = f'receiving_stats/receiving_stats_{yr}.csv'
        combined_sheet = join_sheets(passing_stats, rushing_stats, receiving_stats)
        sheets.append(combined_sheet)
    
    complete_sheet = pd.concat(sheets, axis=0, ignore_index=True, sort=False)
    complete_sheet.to_csv('leviathan_stats.csv', index=False)
