import re
import pandas as pd
from unicodedata import normalize as unormalize 

RX_INITIAL  = re.compile(r'\b([A-Z])\.\s*')                     
RX_SUFFIX   = re.compile(r'\s+(jr|sr|ii|iii|iv|v|vi)\.?$', re.I)
RX_MULTISP  = re.compile(r'\s+')


def normalize_player(raw):
    """Cleans and normalizes player names to prevent duplicate rows.

    Args:
        raw: A string representing a player's name.
    
    Returns:
        A string with the player's name cleaned and normalized. Removes periods from 
        initials, removes suffixes, makes player names lowercase, etc.
    """
    # Ignore NaNs.
    if pd.isna(raw):                    
        return raw
    # Handle odd unicode charaters: ẞ -> SS, é -> e, etc.
    s = unormalize('NFKD', raw) 
    # Remove dots from initials.        
    s = RX_INITIAL.sub(r'\1', s)
    # Remove any stray periods.       
    s = s.replace('.', '')  
    # Strip name suffixes like Jr., Sr., II, III, etc.            
    s = RX_SUFFIX.sub('', s)
    # Collapse whitespace.
    s = RX_MULTISP.sub(' ', s)  
    # Make names case-insensitive.      
    return s.casefold().strip()         


def add_adp(main_sheet, adp_sheet):
    """Add ADP data to my main stats sheet.

    Args:
        main_sheet: A Pandas dataframe containing NFL statistics from Pro Football Reference.
        adp_sheet:  A Pandas dataframe containing average draft position data from FantasyPros.
    
    Returns:
        The combined dataframe that results from performing an outer join on the two 
        sheets on player name, position, and season.
    """
    main_df = pd.read_csv(main_sheet)
    adp_df = pd.read_csv(adp_sheet)
    # Remove extraneous text that starts with a square root symbol and also strip white space.
    adp_df['Player'] = (
        adp_df['Player']
        .str.replace(r"[^A-Za-z.\-'\s].*$", '', regex=True)  
        .str.rstrip()                                    
    )

    for df in (main_df, adp_df):
        df['_name_key'] = df['Player'].map(normalize_player)

    keys = ['_name_key', 'Pos', 'Season']
    combined = pd.merge(
        main_df,
        adp_df,
        left_on=keys,
        right_on=keys,
        how='outer',
        suffixes=('_main', '_adp'),
    )
    # When the sheets are merged, the names of any players from the ADP sheet who are 
    # not in the main PFR sheet are added as a separate column from the names of the 
    # players in the PFR sheet. This merges those columns so all player names are in one
    # column.
    combined['Player'] = combined['Player_main'].combine_first(combined['Player_adp'])
    combined = combined.drop(columns=['Player_main', 'Player_adp', '_name_key'])

    return combined


if __name__ == "__main__":
    adp = 'adp.csv'
    leviathan = '../leviathan_stats_fantasy.csv'
    combined_df = add_adp(leviathan, adp)
    combined_df.to_csv('leviathan_plus_adp.csv', index=False)
