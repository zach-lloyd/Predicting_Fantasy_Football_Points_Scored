import os, time, random, requests
import pandas as pd
import data_cleaning as dc      


def get_stats_csv(year, stats_type, output_dir):
    """Gets the specified stats for the specified year and saves them to a CSV file.

    Args:
        year:           An integer representing the season for which you want to fetch stats.
        stats_type:     String representing the type of stats to fetch ('passing', 'rushing',
                        'receiving', 'fantasy', 'kicking', 'return', or 'defense').
        output_dir:     String representing the directory to save the CSV file in.

    Returns:
        Null. Writes the stats sheet to a CSV file and saves it in output_dir.
    
    Raises:
        HTTPError:          URL generates an invalid response.
        RequestException:   Request to Pro Football Reference was unsuccessful.
        ValueError:         Unable to locate the table in the response returned from PFR.
        Exception:          For general errors.
    """
    # Use a header to avoid getting blocked.
    headers = {
        'User-Agent': (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
            'AppleWebKit/537.36 (KHTML, like Gecko) '
            'Chrome/91.0.4472.124 Safari/537.36'
        )
    }
    # Make sure to use the url of the table itself rather than the url of the page.
    # The response you get from using the url of the table itself is much easier to parse.
    url = f'https://www.pro-football-reference.com/years/{year}/{stats_type}.htm#{stats_type}'

    try:
        response = requests.get(url, headers=headers, timeout=10)
        # Check for errors.
        response.raise_for_status()
        response_text = response.text
        print('Successfully fetched HTML content.')
        # Tables on PFR should typically have an id tag equal to the type of stats stored
        # in the table.
        tables = pd.read_html(response_text, attrs={'id': stats_type})

        if not tables:
            print(f'No table with id={stats_type} found for the year {year} at {url}.')
            print('The page structure might have changed, or data for this year might not be available in the expected format.')
            return
        # Get the first table (there shouldn't be more than one).
        df = tables[0]
        # Add a column reflecting the season that these stats are from.
        df['Season'] = year
        
        print('Successfully parsed the fantasy stats table.')
        # Many columns from PFR are multiindex, so this is needed to flatten them.
        df.columns = dc.flatten_multind_cols(df.columns)
        print('cols flattened')
        # PFR columns sometimes have extraneous characters tacked onto the end of
        # player names. This function will strip those extraneous characters.
        df = dc.clean_player_names(df)
        print('names cleaned')
        # Make sure that integer/float columns are properly converted to numeric
        # values.
        df = dc.convert_to_numeric(df)
        print('converted to numeric')
        # Drop any rows where all the values are NaN.
        df.dropna(how='all', inplace=True)
        print('NaN rows dropped')
        # If the df includes a separate row for the league averages, remove it.
        df.drop(df[df['Player'] == 'League Average'].index, inplace=True)
        # Reset df index after filtering and dorpping rows.
        df.reset_index(drop=True, inplace=True)

        print('Data cleaning complete.')

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        csv_filename = os.path.join(output_dir, f'{stats_type}_stats_{year}.csv')
        df.to_csv(csv_filename, index=False)

        print(f'Successfully saved data to {csv_filename}')

    except requests.exceptions.HTTPError as e:
        print(f'HTTP error occurred for year {year}: {e}')
        print(f'URL attempted: {url}')

    except requests.exceptions.RequestException as e:
        print(f'Error during HTTP request for year {year}: {e}')

    except ValueError as e:
        # This can happen if pd.read_html fails (e.g., "No tables found" if attrs don't match).
        print(f'Error parsing HTML table for year {year}: {e}')

    except Exception as e:
        print(f'An unexpected error occurred for year {year}: {e}')

if __name__ == "__main__":
    start_year = 1932
    end_year = 2024
    stats_types = ['passing', 'rushing', 'receiving']
    # For future years, I can just remove the outer loop and change this code to
    # just get the most recent year.
    for yr in range(start_year, end_year + 1):
        for stat in stats_types: 
            print(f'Starting {stat} stats download for year {yr}...')
            get_stats_csv(yr, stat, f'{stat}_stats')
            print(f'Pausing for a few seconds...')
            # Jitter the PFR requests to avoid getting rate limited.
            time.sleep(random.uniform(5.5, 10.5))
