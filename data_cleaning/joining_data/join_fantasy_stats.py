import pandas as pd
import numpy as np


def calculate_fantasy_points(df):
    """Calculate PPR fantasy points scored for players in the input dataframe. 

    Args:
        df: A pandas dataframe containing NFL stats.
    
    Returns:
        Column of fantasy point values.
    """
    cols = ['Pass_Yds','Pass_TD','Int','Rushing_Yds','Rushing_TD',
            'Fumbles','Receiving_Rec','Receiving_Yds','Receiving_TD']
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce').fillna(0)

    return (
        df['Pass_Yds']   * 0.04     +          
        df['Pass_TD']    * 4        -          
        df['Int']        * 2        +          
        df['Rushing_Yds']* 0.10     +          
        df['Rushing_TD'] * 6        -          
        df['Fumbles']    * 2        +          
        df['Receiving_Rec']         +   
        df['Receiving_Yds']* 0.10   +      
        df['Receiving_TD'] * 6            
    )


def join_fantasy_sheets():
    """Join all fantasy stat sheets from 1970 onward in one df.

    Returns: combined dataframe consisting of all fantasy stat sheets.
    """
    start_year = 1970
    # Adjust as needed for subsequent years.
    end_year = 2024
    dfs = []

    for yr in range(start_year, end_year + 1):
        df = pd.read_csv(f'fantasy_stats/fantasy_stats_{yr}.csv')
        # Remove duplicate header rows that PFR includes every so often in their stat sheets.
        df = df.dropna(subset=['Rk'])
        # Can remove many columns because they are duplicates of columns already contained
        # in the passing, rushing, and receiving stat sheets, but there are some stats in
        # the fantasy sheets that are not contained in the normal stat sheets, so want to
        # keep those.
        cols_to_keep = [
            'Rk', 'Player', 'Tm', 'FantPos', 'Age', 'Fumbles_FL', 'Scoring_TD', 
            'Scoring_2PM', 'Scoring_2PP', 'Fantasy_FantPt', 'Fantasy_PPR', 
            'Fantasy_DKPt', 'Fantasy_FDPt', 'Fantasy_VBD', 'Fantasy_PosRank',
            'Fantasy_OvRank', 'Season_'
        ]
        df = df[cols_to_keep]
        # Rename these columns to be consistent with the corresponding names in the master
        # sheet, because these are the keys that will be used to join these stats to the
        # master sheet.
        df.rename(
            columns={
                'Tm': 'Team',
                'FantPos': 'Pos',
                'Season_': 'Season'
            },
            inplace=True
        )
        dfs.append(df)
    
    complete_df = pd.concat(dfs, axis=0, ignore_index=True, sort=False)

    return complete_df


if __name__ == "__main__":
    complete_sheet = join_fantasy_sheets()
    leviathan_stats = pd.read_csv('leviathan_stats.csv')
    key_cols = ['Team', 'Player', 'Age', 'Season', 'Pos']

    merged = (
        leviathan_stats
        .merge(complete_sheet,            
               on=key_cols,         
               how='left',
               suffixes=('', '_cs')  # Make sure suffixes are appropriate; it threw an error without this.       
        )
    )
    # Handle seasons prior to 1970, which do not have fantasy points data on PFR so it needs to be 
    # calculated manually.
    merged['Fantasy_PPR'].replace('', np.nan, inplace=True)
    needs_ppr_points = merged['Fantasy_PPR'].isna()
    merged.loc[needs_ppr_points, 'Fantasy_PPR'] = calculate_fantasy_points(merged.loc[needs_ppr_points])

    merged.to_csv('leviathan_stats_fantasy.csv', index=False)
