import pandas as pd

MAIN_DF = pd.read_csv(
        'combine_data/leviathan_plus_combine.csv',
        dtype={'Season': 'int64', 
               'Pos': 'string'}
        )


def create_positional_df(pos):
    """Create new dataframe with players at the specified position in the main dataframe.

    Args:
        pos: A string that is either 'QB', 'RB', 'WR', or 'TE'.
    
    Returns:
        Null. Saves a new data frame with all player seasons from and after 1987 for the
        specified position.
    """
    pos_df = MAIN_DF.loc[
        (MAIN_DF['Pos'] == pos) &     
        (MAIN_DF['Season'] >= 1987)    
    ].copy()
    # When shifting previous season data, don't shift columns that are unchanged from season
    # to season.
    cols_to_exclude = [
        'Player', 'Season', 'Team', 'Pos', 'College', 'first_season_year', 'Draft_Class', 
        'Height(in)', 'Weight(lbs)', 'Hand Size(in)', 'Arm Length(in)', 'Wonderlic', 
        '40Yard', 'Bench Press', 'Vert Leap(in)', 'Broad Jump(in)', 'Shuttle', '3Cone'
    ]

    stat_cols = pos_df.columns.difference(cols_to_exclude)
    # Copy the stat columns from the original df and increment season number so these rows
    # can be joined as new columns to the season following the season in which the stats were
    # accumulated.
    prev_df = (
        pos_df[['Player', 'Season', 'Team', *stat_cols]]
        .copy()
        .assign(Season=lambda d: d['Season'] + 1)        
        .rename(columns={c: f'{c}_prev' for c in stat_cols})
    )

    join_keys = ['Player', 'Season', 'Team']
    # When I first wrote and ran this code, I was getting some errors indicating some 
    # duplicate entries, so I included the below code to find those.
    dup_left = (
        pos_df[pos_df.duplicated(subset=join_keys, keep=False)]
        .sort_values(join_keys)
    )

    if not dup_left.empty:
        print(f'[{pos}] duplicates in CURRENT-season data:', dup_left.shape[0])
        print(
            dup_left.groupby(join_keys, as_index=False)
            .size()
            .head(10)      
            .to_string(index=False)
        )
        dup_left.to_csv(f'{pos}_dup_left.csv', index=False)

    dup_right = (
        prev_df[prev_df.duplicated(subset=join_keys, keep=False)]
        .sort_values(join_keys)
    )

    if not dup_right.empty:
        print(f'[{pos}] duplicates in PREVIOUS-season data:', dup_right.shape[0])
        print(
            dup_right.groupby(join_keys, as_index=False)
            .size()
            .head(10)
            .to_string(index=False)
        )
        dup_right.to_csv(f'{pos}_dup_right.csv', index=False)

    overlap_counts = (
        pos_df[join_keys]
        .merge(prev_df[join_keys], on=join_keys, how='inner')
        .value_counts()
    )

    if (overlap_counts > 1).any():
        print(f'[{pos}] potential one-to-many keys across sides:')
        print(overlap_counts[overlap_counts > 1].head(10).to_string())
    # Fill previous season statistics with 0's if the values are NaN because this indicates
    # that the player either was not in the league, was injured, or just did not see the field
    # for some other reason in the previous season.
    pos_df_with_prev = pos_df.merge(
        prev_df,
        on=['Player', 'Season', 'Team'],      
        how='left',                  
        validate='1:1'              
    )

    prev_cols = [f'{c}_prev' for c in stat_cols]
    pos_df_with_prev[prev_cols] = pos_df_with_prev[prev_cols].fillna(0)

    pos_df_with_prev.to_csv(f'{pos}_with_prev.csv')


if __name__ == "__main__":
    positions = ['QB', 'RB', 'WR', 'TE']

    for position in positions:
        create_positional_df(position)
