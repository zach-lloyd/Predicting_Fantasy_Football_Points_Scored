import pandas as pd


def concat_combine_data(combine_dfs):
    """Concatenate Scouting Combine data for offensive players into one df.

    Args:
        combine_dfs: A list of dataframes containing NFL Scouting Combine data from 1987 to 2025.

    Returns:
        A dataframe containing NFL Scouting Combine measurements from all offensive players from 1987 to 2025.
    """
    master_df = pd.concat(combine_dfs, axis=0, ignore_index=True, sort=False)

    offensive_positions = {'QB', 'RB', 'FB', 'WR', 'TE',}
    df_offense = master_df[master_df['POS'].str.upper().isin(offensive_positions)].copy()
    df_offense.rename(columns={'Year': 'Season', 'POS': 'Pos', 'Name': 'Player'})   

    return df_offense


if __name__ == "__main__":
    start_year = 1987
    end_year = 2025

    dfs = [
        pd.read_csv(f'combine_csvs/combine_data_{yr}.csv') 
        for yr in range(start_year, end_year + 1)
    ]

    complete_sheet = concat_combine_data(dfs)
    complete_sheet.to_csv('full_combine_data.csv', index=False)
