import pandas as pd


def clean_name(s):
    """Lowercase, drop punctuation & extra spaces.
    
    Args:
        player: A string representing a player's name.

    Returns:
        A string representing the player's name that is all lowercase with punctuation
        and extraneous spaces removed.
    """
    return (s.str.lower()
              .str.replace(r'[^a-z\s]', '', regex=True)
              .str.replace(r'\s+', ' ',  regex=True)
              .str.strip())


def add_combine_data(df_stats, df_combine):
    """Join NFL Scouting Combine data to main CSV file.

    Args:
        df_stats: Main Pandas dataframe containing passing, rushing, and receiving
        statistics, average draft position, and fantasy scoring data.

        df_combine: Pandas dataframe containing NFL Scouting Combine data dating back to
        1987.

    Returns:
        Combined dataframe with df_combine joined to df_stats.
    """
    for df in (df_stats, df_combine):
        df['name_clean'] = clean_name(df['Player'])
        df['pos_clean']  = df['Pos'].str.upper().str.strip()
    # Get the earliest season for each player in the main file.
    first_season = (
        df_stats
            .loc[:, ['name_clean', 'pos_clean', 'Season']]
            .groupby(['name_clean', 'pos_clean'])
            .Season.min()               
            .rename('first_season_year')
            .reset_index()
    )
    # Join the "first season" table to the scouting combine sheet.
    candidates = (first_season
                  .merge(df_combine,
                         on=['name_clean', 'pos_clean'],
                         how='left', suffixes=('', '_cmb')))   
    # Calculate how far the applicable combine’s draft year is from a player's first season.
    candidates['year_gap'] = (candidates['first_season_year']
                              - candidates['Season']).abs()
    # For each (name, pos) keep the combine data from the combine with the smallest gap.
    best_match = (candidates
                  .sort_values('year_gap')
                  .drop_duplicates(subset=['name_clean', 'pos_clean'], keep='first'))
    
    df_stats_enriched = (df_stats
        .merge(best_match,
               on=['name_clean', 'pos_clean'],
               how='left')        
    )

    return df_stats_enriched


if __name__ == "__main__":
    leviathan = pd.read_csv('../adp/leviathan_plus_adp.csv')
    combine_data = pd.read_csv('full_combine_data.csv')
    combined_sheet = add_combine_data(leviathan, combine_data)
    combined_sheet.to_csv('leviathan_plus_combine.csv', index=False)