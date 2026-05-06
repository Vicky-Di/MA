import os
import random
import pandas as pd  
import numpy as np
  
def organize_timestamps(df: pd.DataFrame, col_timestamp: str) -> pd.DataFrame:  
    """  
    Organize timestamps in a pandas DataFrame.  
  
    This function converts "col_timestamp" to datetime type, renames it as "timestamp",
    sets it as the index, and sorts the index. 
  
    Args:  
        df (pd.DataFrame): The DataFrame to process.  
        col_timestamp (str): The name of the column containing the timestamps.  
  
    Returns:  
        pd.DataFrame: The processed DataFrame with sorted "timestamp" as the index.  
    """  
    # Check if the current index is the same as the col_timestamp column  
    if df.index.name == col_timestamp:  
        df.index = pd.to_datetime(df.index)  # Set type
        df.index.name = "timestamp"  # Rename
    else:  
        df["timestamp"] = pd.to_datetime(df[col_timestamp])  # Set type and rename
        df = df.set_index("timestamp")  # Set as index
    
    # Sort the index  
    df = df.sort_index()  
  
    return df 


def example_data(seed: int = 1) -> pd.DataFrame:  
    """  
    Generates a modified dataset based on an example Parquet file, with voltage values altered randomly.  
  
    This function reads a base dataset from a Parquet file (`data_example.parquet`) located in the same  
    package directory as the script. It modifies the `voltage` column by applying a random offset, which  
    is determined by a provided or default seed for reproducibility.  
  
    Args:  
        seed (int, optional): A seed value for the random number generator to ensure reproducibility.  
                              Defaults to 1.  
  
    Returns:  
        pd.DataFrame: A modified DataFrame containing the original data with adjusted `voltage` values.  
    """  
    package_dir = os.path.dirname(__file__)
    file_path = os.path.join(package_dir, 'data_example.parquet')
    base_data = pd.read_parquet(file_path)
    random.seed(seed)
    # modify the voltage values in base_data based on the seed
    start_offset = random.uniform(-0.05, 0.05)
    overtime_offset = random.uniform(0, 0.05)
    offset = np.arange(start_offset, start_offset + overtime_offset, len(base_data))
    data = base_data.copy()
    data.voltage = base_data.voltage + offset
    return data


def validate_columns(df: pd.DataFrame, required_columns: list):  
    """  
    Validates that the required columns exist in a DataFrame, including checking the index.  
  
    Parameters  
    ----------  
    df : pd.DataFrame  
        The DataFrame to validate.  
    required_columns : list  
        A list of column names that must exist in the DataFrame or as the index.  
  
    Raises  
    ------  
    ValueError  
        If any required column is missing.  
    """  
    # Combine DataFrame columns and the index name (if it exists)  
    existing_columns = list(df.columns)  
    if df.index.name:  # Add index name to the list if it exists (avoid df.reset_index because it creates a copy of the dataframe)
        existing_columns.append(df.index.name)  
      
    # Find missing columns  
    missing_columns = [col for col in required_columns if col not in existing_columns]  
    if missing_columns:  
        raise ValueError(f"The following required columns are missing from the DataFrame: {missing_columns}")  
    

def fill_gaps(data_with_gap, max_gap_h=1):
    """
    Fill data gaps in a DataFrame using forward-fill, but only for gaps less than 'max_gap_h' hour.  
      
    Parameters:  
        data_with_gap (pd.DataFrame): A DataFrame with a timestamps as index.  
      
    Returns:  
        pd.DataFrame: A DataFrame with gaps filled based on the specified conditions.  
    """
    index_desired =  pd.to_datetime(data_with_gap.index)
    data = pd.DataFrame(index=index_desired)
    for col in data_with_gap.columns:
        # first forward fill all gaps, later convert back to nan for large gaps.
        data[col] = data_with_gap[col].ffill()
        index_with_data = data_with_gap[col].dropna().index
        time_diff_compared_with_previous_timestamp = index_with_data.to_series().diff().dt.total_seconds().fillna(0)
        time_diff_compared_with_next_timestamp = index_with_data.to_series().diff(-1).dt.total_seconds().fillna(0)

        large_gaps_from = time_diff_compared_with_next_timestamp[abs(time_diff_compared_with_next_timestamp) > max_gap_h*3600].index
        large_gaps_to = time_diff_compared_with_previous_timestamp[abs(time_diff_compared_with_previous_timestamp) > max_gap_h*3600].index
        
        for gap_from, gap_to in zip(large_gaps_from, large_gaps_to):
            try:  
                next_timestamp_after_gap_from = data.index[data.index.get_loc(gap_from) + 1]  # Get the next timestamp  
                previous_timestamp_before_gap_to = data.index[data.index.get_loc(gap_to) - 1]
                # Convert back to nan if the gap is large.
                data.loc[next_timestamp_after_gap_from:previous_timestamp_before_gap_to, col] = np.nan # "loc" function: "contrary to usual python slices, both the start and the stop are included."
            except IndexError:  
                # Todo: Handle cases where `gap_from` is the last timestamp or `gap_to` is the first timestamp  
                continue  
    return data
