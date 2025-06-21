import pandas as pd


def flatten_multind_cols(cols):
    """Flatten any column names that are multi-indexed.

    Args:
        cols: A list of column names in a Pandas dataframe.
    
    Returns:
        A list of column names with any multi-indexed column names flattened.
    """
    if isinstance(cols, pd.MultiIndex):
        new_cols = []

        for col_tuple in cols:
            # If the first level of the MultiIndex is 'Unnamed: X_level_0' or empty,
            # just use the second level name.
            if 'Unnamed:' in col_tuple[0] or col_tuple[0].strip() == '':
                new_cols.append(col_tuple[1])
            else:
                # Otherwise, combine them. If both levels are the same, just use one.
                if col_tuple[0] == col_tuple[1]:
                    new_cols.append(col_tuple[0])
                else:
                    new_cols.append(f'{col_tuple[0]}_{col_tuple[1]}')

        cols = new_cols
        print('Successfully flattened column headers.')
        
        return new_cols
    else:
        print('No columns need to be flattened.')

        return cols
    

def remove_dup_headers(df):
    """Remove duplicate header rows.

    Args:
        df: A Pandas dataframe containing NFL stats.

    Return:
        A dataframe with any duplicate header rows removed.
    """
    # PFR fantasy tables typically have a 'Rk' column, so can use that to remove
    # duplicate header rows.
    if 'Rk' in df.columns:
        # Convert the entire 'Rk' column to strings and then keep only the rows
        # where the rank column only contains digits. If a row has any non-digit
        # characters, it must be a duplicate header row and it can be removed.
        df = df[df['Rk'].astype(str).str.isdigit()]
    else:
        # Handle case where the table doesn't have a 'Rk' column by assuming the
        # first column is the 'Rk' column.
        print("Warning: 'Rk' column not found as expected. Attempting to filter based on the first column.")

        if not df.empty:
            first_col_name = df.columns[0]
            df = df[df[first_col_name].astype(str).str.isdigit()]
    
    return df


def clean_player_names(df):
    """Strip extraneous characters from player names.

    Args:
        df: A Pandas dataframe containing NFL stats.

    Returns:
        A dataframe with extraneous characters removed from the end of player names.
    """
    if 'Player' in df.columns:
        df['Player'] = df['Player'].astype(str).str.replace(r'[*+]', '', regex=True).str.strip()
    
    return df


def convert_to_numeric(df):
    """Convert columns that should be numbers to numeric.
    
    Args:
        df: A Pandas dataframe containing NFL stats.

    Returns:
        A dataframe with all columns that should be numeric converted to numbers.
    """
    for col in df.columns:
        # Skip columns that should remain as text, convert other columns to numbers.
        # errors='coerce' changes any non-numeric values in the converted columns to NaN.
        if col.lower() not in ['player', 'team', 'pos', 'awards', 'tm', 'fantpos']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df
