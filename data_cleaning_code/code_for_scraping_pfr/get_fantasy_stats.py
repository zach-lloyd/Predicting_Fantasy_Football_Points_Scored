import time, random
from get_stats_csv import get_stats_csv 


def get_fantasy_stats(year):
    """Get fantasy stats for the specified year.

    Args:
        year: An integer representing the year to get fantasy stats for.
    
    Returns:
        Null. Calls get_stats_csv and writes the results to a CSV file.
    """
    stats_type = 'fantasy'
    output_dir = '../fantasy_stats'
    get_stats_csv(year, stats_type, output_dir)

if __name__ == "__main__":
    start_year = 1970
    end_year = 2024
    # For future years, I can just remove the outer loop and change this code to
    # just get the most recent year.
    for yr in range(start_year, end_year + 1):
        print(f'Starting fantasy stats download for year {yr}...')
        get_fantasy_stats(yr)
        print(f'Pausing for a few seconds...')
        # Jitter the requests to PFR to lessen the likelihood of getting rate-limited.
        time.sleep(random.uniform(5.5, 10.5))
