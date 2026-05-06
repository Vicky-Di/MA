import pandas as pd
import numpy as np


def find_interval(data_no_gap: pd.DataFrame, condition: str, min_length: int) -> pd.DataFrame: 
    """
    Find operation intervals that fulfills the 'condition' and duration > 'min_length'.
    :param data_no_gap: We need to temporarily fill the nan values. Otherwise, each missing value
    will be recognized as a start because of "indices - indices.shift(1)) != 1". The data must contain a column
    named "timestamp".
    :param condition: A string representing the condition for the query (e.g., "I > 0.1" or "U > 1.5").
    :param min_length: An integer representing the minimum length (in terms of the number of data points)  
                       for an operating interval to be considered valid.  
    :return: A DataFrame with columns start_time, end_time, and interval_h. 
    If no interval is found, return an empty dataframe.
    """
    data = data_no_gap
    ts_conditioned = data.reset_index(
        drop=False, inplace=False).query(condition).copy()

    indices = ts_conditioned.index.to_series()

    ts_conditioned['group'] = ((indices - indices.shift(1)) != 1).cumsum()
    group_size = ts_conditioned.groupby(
        ['group']).size().reset_index(name='counts')

    ts_conditioned.set_index('timestamp', inplace=True)

    intervals = list()
    for _, row in group_size.query(f"counts > {min_length}").iterrows():
        group = row.group
        s = ts_conditioned.query('group == @group').index[0]
        e = ts_conditioned.query('group == @group').index[-1]
        intervals.append([s, e])

    if len(intervals) < 1:
        print(f"No interval is found that fulfills the condition {condition}.")
        return pd.DataFrame(columns=['start_time', 'end_time', 'interval_h'])  

    intervals_df = pd.DataFrame(intervals).iloc[:, 0:2]
    intervals_df.rename(columns={0: 'start_time', 1: 'end_time'}, inplace=True)
    intervals_df["interval_h"] = (
        intervals_df["end_time"] - intervals_df["start_time"]) / np.timedelta64(1, 'h')
    return intervals_df


def calc_h_since_last_start(data: pd.DataFrame, condition_op: str, min_length: int = 1) -> pd.DataFrame:
    """  
    Calculate the hours since the last start of an operation interval.  
    :param data: A DataFrame indexed and sorted by datetime  
    :param condition_op: A string representing the condition for the query (e.g., "I > 0.1" or "U > 1.5").  
    :param min_length: An integer representing the minimum length (in terms of the number of data points)  
                       for an operating interval to be considered valid. Default is 1.  
    :return: A DataFrame with an additional column 'h_since_last_start' indicating the hours since the last start.  
    :raises ValueError: If no operating intervals are found that satisfy the given condition and minimum length.  
    """  
    # # original
    #  # We need to temporarily fill the nan values. Otherwise, each missing value will be recognized as a start
    # # because of "indices - indices.shift(1)) != 1" in the function find_interval.
    # operating_intervals = find_interval(data.ffill(), condition_op, min_length)





    # Fill NaN with 0 (not ffill!) so that shutdown periods (where voltage/current are NaN)
    # are correctly treated as non-operating. Using ffill would propagate the last operating
    # voltage into shutdown rows, causing conditions like "voltage>1.3" to remain True
    # across shutdowns and h_since_last_start to accumulate without resetting.
    operating_intervals = find_interval(data.fillna(0), condition_op, min_length)
    if operating_intervals.empty:
        raise ValueError("No operating intervals are found.")
    
    for _, row in operating_intervals.iterrows():
        data.loc[row["start_time"]:row["end_time"],
                "last_start_time"] = row["start_time"]
    # Calculate "h_since_last_start"
    data["h_since_last_start"] = (
        data.index - data.last_start_time) / np.timedelta64(1, 'h')
    
    # Handle the special case where the data begins with an operation  
    data = _calc_h_since_last_start_special_case(data, operating_intervals)  
  
    return data  
  
  
def _calc_h_since_last_start_special_case(data: pd.DataFrame, operating_intervals: pd.DataFrame) -> pd.DataFrame:  
    """  
    Handle the special case where the data begins with an operation interval.  
  
    If the data begins with an operation (no start at the beginning), set the 'h_since_last_start'  
    for the first interval to infinity and adjust the 'last_start_time' accordingly.  
  
    :param data: The original DataFrame containing the 'last_start_time' and 'h_since_last_start' columns.  
    :param operating_intervals: A DataFrame containing the operating intervals.  
    :return: The modified DataFrame with the special-case adjustments applied.  
    """  
    first_data = data.iloc[0]  
    if first_data.name == first_data.last_start_time:  
        # Find the corresponding interval  
        corresponding_interval = operating_intervals.query("start_time == @first_data.name").iloc[0]  
        # Modify data within this interval  
        data.loc[corresponding_interval["start_time"]:corresponding_interval["end_time"], "last_start_time"] = np.nan  
        data.loc[corresponding_interval["start_time"]:corresponding_interval["end_time"], "h_since_last_start"] = np.inf  
    return data  


def temp_correction(
    df: pd.DataFrame,   
    i: str,   
    cell_vot: str,   
    temp: str,   
    temperature_correction_target: float = 60 
    ) -> pd.DataFrame:  
    """
    Correct the cell votage to 'temperature_correction_target'.

    Parameters
    ----------
    df : Data Frame
        Data Frame with cell voltage and temperature data
    i : String
        Column containing the current density data.
    cell_vot : String
        Column containing the cell votage data.
    temp : String
        Column containing the temperature data.
    temperature_correction_target : float, optional  
        The temperature target (in the same units as the `temp` column)   
        to which the cell voltage should be corrected. Default is 60.  

    Returns
    -------
    df: Data Frame
        with an additional column with corrected cell votage.

    """
    new_name = cell_vot + '_corrected'
    df[new_name] = df[cell_vot] - (df[i]*1.211+2.4591)/1000*(temperature_correction_target - df[temp])

    return df

def pressure_correction(
    df: pd.DataFrame,   
    col_i: str,   
    col_U: str,   
    col_pressure_gauge_bar: str,   
    pressure_correction_target_bar: float = 0 
    ) -> pd.DataFrame:  
    """
    Correct the cell votage to 'pressure_correction_target_bar'.

    Parameters
    ----------
    df : Data Frame
        Data Frame with cell voltage and temperature data
    col_i : String
        Column containing the current density data.
    col_U : String
        Column containing the cell votage data.
    col_pressure_gauge_bar : String
        Column containing the gauge pressure data in [bar].
    pressure_correction_target_bar : float, optional  
        The pressure target to which the cell voltage should be corrected. 
        Default is 0, i.e. atmospheric pressure.  

    Returns
    -------
    df: Data Frame
        with an additional column with corrected cell votage.

    """
    new_name = col_U + '_P_corrected'
    df[new_name] = df[col_U] + (df[col_i]*0.948-2.44) / 1000 *(df[col_pressure_gauge_bar] - pressure_correction_target_bar)
    return df