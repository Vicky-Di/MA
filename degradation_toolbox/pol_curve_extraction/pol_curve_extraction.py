import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from degradation_toolbox.utils.data_helpers import example_data, organize_timestamps, validate_columns, fill_gaps
from degradation_toolbox.utils.calculation_basics import find_interval, calc_h_since_last_start, temp_correction


class PolarizationCurve:
    def __init__(self, 
                 df: pd.DataFrame, 
                 col_timestamp: str, 
                 col_I: str, 
                 col_voltage: str, 
                 col_temperature: str,  
                 temperature_correction_target: float = 60,
                 cell_area: float = 5000, 
                 current_tolerance: float = 50, 
                 allowed_min_temperature: float = 50,  
                 allowed_min_h_since_last_start: float = 1, 
                 data_stability_look_back: str = "5min",  
                 voltage_stability_threshold: float = 0.01, 
                 method_to_find_steps: str = "plateaus",  
                 bins_min: float = 950, 
                 bins_max: float = 8050, 
                 bins_width: float = 100, 
                 min_count: float = 30):
        """
        Function to find a polarization curve in the given dataframe "df".  
        
        Parameters  
        ----------  
        df : DataFrame  
            DataFrame with cell voltage, current, and temperature data.  
        col_timestamp : str  
            Column name containing timestamps.  
        col_I : str  
            Column name containing the current data (in A).  
        col_voltage : str  
            Column name containing the cell voltage data (in V).  
        col_temperature : str  
            Column name containing the temperature data (in °C).  
        temperature_correction_target: float, optional
            To which degree celsius should the voltage be corrected to (default is 60).
        cell_area : float, optional  
            Cell area in cm^2 (default is 5000).  
        current_tolerance : float, optional  
            Tolerance for current stability during polarization data extraction (default is 50).  
        allowed_min_temperature : float, optional  
            Minimum allowable cell temperature for data to be considered (default is 50°C).  
        allowed_min_h_since_last_start : float, optional  
            Minimum hours since the last start for data to be considered (default is 1 hour).  
        data_stability_look_back : str, optional  
            Time interval to look back for data stability checks (default is "5min").  
        voltage_stability_threshold : float, optional  
            Threshold for voltage stability in volts (default is 0.01 V).  
        method_to_find_steps : str, optional  
            Method to identify steps in current data; can be either 'histogram' or 'plateaus' (default is 'plateaus').  
        bins_min : int, optional  
            Minimum current value in mA for binning (default is 950).  
        bins_max : int, optional  
            Maximum current value in mA for binning (default is 8050).  
        bins_width : int, optional  
            Width of each current bin in mA (default is 100).  
        min_count : int, optional  
            Minimum count of data points required within a bin (default is 30).  
        """
        self.data_prepared = False  # flag to assure unprepared data will not be calculated.

        # Check if required columns exist in the DataFrame  
        required_columns = [col_timestamp, col_I, col_voltage, col_temperature]
        validate_columns(df, required_columns)
        self.df = df.copy()

        self.col_timestamp = col_timestamp
        self.col_I = col_I
        self.col_voltage = col_voltage
        self.col_temperature = col_temperature
        self.temperature_correction_target = temperature_correction_target
        self.cell_area = cell_area  
        self.current_tolerance = current_tolerance  
        self.allowed_min_temperature = allowed_min_temperature  
        self.allowed_min_h_since_last_start = allowed_min_h_since_last_start  
        self.data_stability_look_back = data_stability_look_back  
        self.voltage_stability_threshold = voltage_stability_threshold  
        self.method_to_find_steps = method_to_find_steps  
        self.bins_min = bins_min  
        self.bins_max = bins_max  
        self.bins_width = bins_width  
        self.min_count = min_count 
                                    
    def prepare_data(self) -> bool:
        """Prepare the data for polarization curve extraction.
        """
        # Align name, format, order of the timestamp column. (The 'find_steps' function requires the column name to be 'timestamp'.)
        self.df = organize_timestamps(self.df, self.col_timestamp)

        # Fill gaps to handle cases where data is only archived "on change" (like Trailblazer)
        # fill_gap function requires that the dataframe has timestamps as index
        self.df = fill_gaps(self.df).copy()

        if not self.df.empty:
            self.data_prepared = True
            return True
        return False

    def calculate(self) -> None:
        """Run the calculations to extract the polarization curve data.
        """
        if not self.data_prepared:
            raise ValueError("Data is not prepared. Please call prepare_data() before calculate().")
        
        self.steps = self.find_steps()
        
        if len(self.steps) < 1:
            raise ValueError("No polarization curve is found.")  # Remove print, make it something else.. 
        
        # Preprocessing: Rename columns, temperature correction, calculate hours since last start etc.
        self.df = self.data_preprocessing()

        # Find pol curve data by looping through each step
        self.pol_curve_data = self.get_pol_curve_data()

        # Don't do like below: if recalculate the curve (re-filter the data) with "get_pol_curve_data_without_temperature_correction", 
        # the "start_timestamp" and "end_timestamp" might be different. So the merge will fail.
        # But "pol_curve_data_without_temperature_correction" is still kept so that user can still get the results.
        self.pol_curve_data_without_temperature_correction = self.get_pol_curve_data_without_temperature_correction()

    def find_steps(self) -> list[dict]:
        """
        Detect discrete steps in the dataset based on the configured detection method.

        This method delegates step detection to a specific implementation
        based on the value of `self.method_to_find_steps`.

        Supported methods:
            - "histogram": Uses a histogram-based binning approach.
            - "plateaus": Uses plateau detection logic.

        Returns
        -------
        list or np.ndarray
            Identified steps as returned by the selected detection method.

        Raises
        ------
        ValueError
            If `self.method_to_find_steps` is not 'histogram' or 'plateaus'.
        """
        if self.method_to_find_steps == "histogram":
            self.steps = self.find_steps_with_histogram()
        elif self.method_to_find_steps == "plateaus":
            self.steps = self.find_steps_with_plateaus()
        else:
            raise ValueError(
                "Invalid value for method_to_find_steps. "
                "Allowed values: 'histogram' or 'plateaus'."
            )

        return self.steps
    
    def find_steps_with_histogram(self):
        """
        Detect steps using a histogram binning approach.

        This method bins the current (`self.col_I` in `self.df`) into specified
        ranges and identifies bins whose counts exceed `self.min_count`.
        The midpoint of each qualifying bin is used to approximate the step level.

        Notes
        -----
        - Limitation:
        If the current fluctuates around a bin edge (e.g., bins [4000, 4100),
        [4100, 4200) and the current fluctuates at ~4100 ± 50 A), the data will 
        be split into two bins, producing two points on the polarization curve 
        that are very close together in both current and time.
        - Default behavior:
        Bins are created from `self.bins_min` to `self.bins_max` with a width 
        of `self.bins_width`. `self.min_count` determines the minimum data 
        points per bin (assuming 1 sample per minute, this approximates minutes).

        Returns
        -------
        np.ndarray
            Array of bin center values for bins meeting the `self.min_count` criterion.
        """
        current = self.df[self.col_I]

        counts, bins = np.histogram(
            current,
            bins=np.arange(self.bins_min, self.bins_max, self.bins_width)
        )

        # Lower bounds of bins that have enough samples
        lower_bound = bins[:-1][counts > self.min_count]

        # Return the midpoint of the bins
        return lower_bound + self.bins_width / 2

    def find_steps_with_plateaus(self) -> list[dict]:
        """
        Identify stable plateaus in the current signal based on the absence of large jumps
        over a specified rolling time window.

        This method analyses the current profile to detect periods where the current remains
        stable (plateaus) and the voltage is above a defined threshold. The detection is based
        on comparing the current in each time step with its rolling maximum and minimum over
        a defined look-back period (`data_stability_look_back`). A plateau is considered valid
        if:
        - The difference between the current value and the rolling extrema is within 
            ±`current_tolerance` Amperes (i.e., no current jumps exceeding 2 x tolerance).
        - Stack voltage during the interval is greater than 1.2 V.
        - The duration of the plateau is at least the length of `data_stability_look_back`.

        Returns
        -------
        list of dict
            A list of dictionaries, each containing:
                - 'start_timestamp': datetime of plateau start
                - 'end_timestamp': datetime of plateau end

        Raises
        ------
        ValueError
            If no stable intervals are found based on the configured parameters.

        Notes
        -----
        - This method is slower than the histogram-based approach, because it examines
        rolling statistics for every point in the profile.
        - Reducing `data_stability_look_back` or `current_tolerance` may increase sensitivity
        and help detect shorter plateaus, but may also increase false positives.
        """
        # Compute rolling max and min for the given stability look-back period
        self.df["max_I_previous"] = (
            self.df[self.col_I].rolling(self.data_stability_look_back).max()
        )
        self.df["min_I_previous"] = (
            self.df[self.col_I].rolling(self.data_stability_look_back).min()
        )

        # Mark points as "jumps" if they exceed ±current_tolerance from rolling extrema
        self.df["is_jump"] = (
            (abs(self.df["max_I_previous"] - self.df[self.col_I])
            > (2 * self.current_tolerance))
            |
            (abs(self.df[self.col_I] - self.df["min_I_previous"])
            > (2 * self.current_tolerance))
        )

        # Use helper to find continuous intervals where no jump occurred and voltage > 1.2 V
        static_intervals = find_interval(
            self.df,
            f"is_jump == False and `{self.col_voltage}` > 1.2",
            1
        )

        if static_intervals.empty:
            raise ValueError(
                f"No stable intervals found. "
                f"Consider reducing 'data_stability_look_back' "
                f"(current: {self.data_stability_look_back}) or "
                f"'current_tolerance' (current: {self.current_tolerance} A)."
            )

        # Keep only intervals longer than the stability look-back period
        static_intervals["long_enough"] = (
            (static_intervals.end_time - static_intervals.start_time)
            >= pd.Timedelta(self.data_stability_look_back)
        )

        plateaus = static_intervals.query("long_enough == True")[
            ["start_time", "end_time"]
        ].rename(columns={"start_time": "start_timestamp",
                        "end_time": "end_timestamp"})

        # Return plateaus as a list of dictionaries: [{'start_timestamp':..., 'end_timestamp':...}, ...]
        return plateaus.to_dict(orient='records')

        
    def data_preprocessing(self):
        # rename the columns so that "df.query" won't fail due to hyphens (will be interpertd as "minus")
        self.df = self.df.rename(columns={self.col_I: "I", self.col_voltage: "U",
                    self.col_temperature: "T_stack"}).copy()
        
        # Temperature correction to 60°C
        self.df["current_density"] = self.df["I"] / self.cell_area
        self.df = temp_correction(self.df, "current_density", "U", "T_stack", self.temperature_correction_target)

        # Calculate hours since last start
        self.df = calc_h_since_last_start(self.df, "current_density>0.1")

        # Calculate min/max current/voltage in the past
        self.df.loc[:, "max_I_previous"] = self.df["I"].rolling(self.data_stability_look_back).max()
        self.df.loc[:, "min_I_previous"] = self.df["I"].rolling(self.data_stability_look_back).min()
        self.df.loc[:, "max_U_previous"] = self.df["U"].rolling(self.data_stability_look_back).max()
        self.df.loc[:, "min_U_previous"] = self.df["U"].rolling(self.data_stability_look_back).min()
        self.df.loc[:, "max_U_corrected_previous"] = self.df["U_corrected"].rolling(self.data_stability_look_back).max()
        self.df.loc[:, "min_U_corrected_previous"] = self.df["U_corrected"].rolling(self.data_stability_look_back).min()

        return self.df

    
    def get_pol_curve_data(self):
        """  
        Gathers polarization curve data based on the identified current steps.  
    
        This method processes the steps found in the dataset to construct the polarization curve data.  
        It operates differently based on the method used to find steps, which can be 'histogram' or 'plateaus'.  
        For each step, it filters the data according to current stability, temperature constraints,  
        and voltage stability, then calculates the average values for current density, voltage,  
        and temperature within the step. The results are compiled into a list of dictionaries, which  
        is then converted into a DataFrame.  
    
        Raises  
        ------  
        Exception  
            If no current steps are found, an exception is raised, suggesting the user to modify  
            the 'method_to_find_steps' and/or its related parameters.  
    
        Returns  
        -------  
        pd.DataFrame  
            A DataFrame containing the polarization curve data. Each row represents a step  
            with columns for start_timestamp, end_timestamp, average_current_density_A_cm2,  
            average_voltage_V, average_temperature_deg_C, and hours_since_last_start. If no steps  
            are found or if no data meet the criteria within the steps, an empty DataFrame is returned.  
        """  
        if len(self.steps) < 1:
            raise ValueError("No current steps are found. You can try to modify the 'method_to_find_steps' and/or its related parameters.")
        else:
            self.pol_curve_data = [] # pol curve data is a list of dictionaries and will be converted to DataFrame in the end

        for step in self.steps:
            if self.method_to_find_steps == "histogram":
                data_this_bin = self.df.query(f"{step-self.bins_width/2}<I<{step+self.bins_width/2}")
                current_this_bin = np.mean(data_this_bin.I)
                data_this_step = data_this_bin.query(f"{current_this_bin-self.current_tolerance}<I<{current_this_bin+self.current_tolerance}"
                                        f" and T_stack>{self.allowed_min_temperature} "
                                        f" and h_since_last_start>{self.allowed_min_h_since_last_start} "
                                        f" and max_I_previous<{current_this_bin+self.current_tolerance} "
                                        f" and min_I_previous>{current_this_bin-self.current_tolerance}"
                                        f" and (max_U_corrected_previous - min_U_corrected_previous) < {self.voltage_stability_threshold}"
                                        )
                if len(data_this_step) >= 1:
                    step_data = {  
                        "start_timestamp": np.min(data_this_step.index),
                        "end_timestamp": np.max(data_this_step.index),
                        "average_current_density_A_cm2": np.mean(data_this_step.I) / self.cell_area,  
                        "average_voltage_V": np.mean(data_this_step.U_corrected),
                        "average_temperature_deg_C": np.mean(data_this_step.T_stack),
                        "hours_since_last_start": np.min(data_this_step.h_since_last_start),
                        "temperature_correction_target": self.temperature_correction_target,
                        "average_voltage_V_uncorrected": np.mean(data_this_step.U)
                    }  
                    self.pol_curve_data.append(step_data) 
            elif self.method_to_find_steps == "plateaus":
                data_this_plateau = self.df.query(f"'{step['start_timestamp']}'<timestamp<'{step['end_timestamp']}'")
                current_this_plateau = np.mean(data_this_plateau.I)
                data_this_step = data_this_plateau.query(f"{current_this_plateau-self.current_tolerance}<I<{current_this_plateau+self.current_tolerance}"
                            f" and T_stack>{self.allowed_min_temperature} "
                            f" and h_since_last_start>{self.allowed_min_h_since_last_start} "
                            f" and max_I_previous<{current_this_plateau+self.current_tolerance} "
                            f" and min_I_previous>{current_this_plateau-self.current_tolerance}"
                            f" and (max_U_corrected_previous - min_U_corrected_previous) < {self.voltage_stability_threshold}"
                            )
                if len(data_this_step) >= 1:
                    step_data = {  
                        "start_timestamp": np.min(data_this_step.index),
                        "end_timestamp": np.max(data_this_step.index),
                        "average_current_density_A_cm2": np.mean(data_this_step.I) / self.cell_area,  
                        "average_voltage_V": np.mean(data_this_step.U_corrected),
                        "average_temperature_deg_C": np.mean(data_this_step.T_stack),
                        "hours_since_last_start": np.min(data_this_step.h_since_last_start),
                        "temperature_correction_target": self.temperature_correction_target,
                        "average_voltage_V_uncorrected": np.mean(data_this_step.U)
                    }  
                    self.pol_curve_data.append(step_data) 
        return pd.DataFrame(self.pol_curve_data)


    def get_pol_curve_data_without_temperature_correction(self):
        """  
        Gathers polarization curve data (without temperature correction) based on the identified current steps.  
    
        This method processes the steps found in the dataset to construct the polarization curve data.  
        It operates differently based on the method used to find steps, which can be 'histogram' or 'plateaus'.  
        For each step, it filters the data according to current stability, temperature constraints,  
        and voltage stability, then calculates the average values for current density, voltage,  
        and temperature within the step. The results are compiled into a list of dictionaries, which  
        is then converted into a DataFrame.  
    
        Raises  
        ------  
        Exception  
            If no current steps are found, an exception is raised, suggesting the user to modify  
            the 'method_to_find_steps' and/or its related parameters.  
    
        Returns  
        -------  
        pd.DataFrame  
            A DataFrame containing the polarization curve data. Each row represents a step  
            with columns for start_timestamp, end_timestamp, average_current_density_A_cm2,  
            average_voltage_V, average_temperature_deg_C, and hours_since_last_start. If no steps  
            are found or if no data meet the criteria within the steps, an empty DataFrame is returned.  
        """  
        if len(self.steps) < 1:
            raise ValueError("No current steps are found. You can try to modify the 'method_to_find_steps' and/or its related parameters.")
        else:
            self.pol_curve_data_without_temperature_correction = [] # pol curve data is a list of dictionaries and will be converted to DataFrame in the end

        for step in self.steps:
            if self.method_to_find_steps == "histogram":
                data_this_bin = self.df.query(f"{step-self.bins_width/2}<I<{step+self.bins_width/2}")
                current_this_bin = np.mean(data_this_bin.I)
                data_this_step = data_this_bin.query(f"{current_this_bin-self.current_tolerance}<I<{current_this_bin+self.current_tolerance}"
                                        f" and T_stack>{self.allowed_min_temperature} "
                                        f" and h_since_last_start>{self.allowed_min_h_since_last_start} "
                                        f" and max_I_previous<{current_this_bin+self.current_tolerance} "
                                        f" and min_I_previous>{current_this_bin-self.current_tolerance}"
                                        f" and (max_U_previous - min_U_previous) < {self.voltage_stability_threshold}"
                                        )
                if len(data_this_step) >= 1:
                    step_data = {  
                        "start_timestamp": np.min(data_this_step.index),
                        "end_timestamp": np.max(data_this_step.index),
                        "average_current_density_A_cm2": np.mean(data_this_step.I) / self.cell_area,  
                        "average_voltage_V": np.mean(data_this_step.U),
                        "average_temperature_deg_C": np.mean(data_this_step.T_stack),
                        "hours_since_last_start": np.min(data_this_step.h_since_last_start)
                    }  
                    self.pol_curve_data_without_temperature_correction.append(step_data) 
            elif self.method_to_find_steps == "plateaus":
                data_this_plateau = self.df.query(f"'{step['start_timestamp']}'<timestamp<'{step['end_timestamp']}'")
                current_this_plateau = np.mean(data_this_plateau.I)
                data_this_step = data_this_plateau.query(f"{current_this_plateau-self.current_tolerance}<I<{current_this_plateau+self.current_tolerance}"
                            f" and T_stack>{self.allowed_min_temperature} "
                            f" and h_since_last_start>{self.allowed_min_h_since_last_start} "
                            f" and max_I_previous<{current_this_plateau+self.current_tolerance} "
                            f" and min_I_previous>{current_this_plateau-self.current_tolerance}"
                            f" and (max_U_previous - min_U_previous) < {self.voltage_stability_threshold}"
                            )
                if len(data_this_step) >= 1:
                    step_data = {  
                        "start_timestamp": np.min(data_this_step.index),
                        "end_timestamp": np.max(data_this_step.index),
                        "average_current_density_A_cm2": np.mean(data_this_step.I) / self.cell_area,  
                        "average_voltage_V": np.mean(data_this_step.U),
                        "average_temperature_deg_C": np.mean(data_this_step.T_stack),
                        "hours_since_last_start": np.min(data_this_step.h_since_last_start)
                    }  
                    self.pol_curve_data_without_temperature_correction.append(step_data) 
        return pd.DataFrame(self.pol_curve_data_without_temperature_correction)




if __name__ == '__main__':
    data = example_data()
    data_staircase_profile = data.query("'2021-08-03 09:00'<timestamp<'2021-08-03 20:15'")
    data_staircase_profile["I"] = data_staircase_profile["currentDensity"] * 5000

    pol_curve_histogram = PolarizationCurve(data_staircase_profile, "timestamp", "I", "voltage", "temperature", method_to_find_steps="histogram") #.pol_curve_data
    pol_curve_histogram.prepare_data()
    pol_curve_histogram.calculate()
    print("Use histogram")
    print(pol_curve_histogram.pol_curve_data)
    plt.scatter(pol_curve_histogram.pol_curve_data.average_current_density_A_cm2, pol_curve_histogram.pol_curve_data.average_voltage_V, label="histogram")

    pol_curve_plateau = PolarizationCurve(data_staircase_profile, "timestamp", "I", "voltage", "temperature", method_to_find_steps="plateaus") #.pol_curve_data
    pol_curve_plateau.prepare_data()
    pol_curve_plateau.calculate()
    print("Use plateaus")
    print(pol_curve_plateau.pol_curve_data)
    plt.scatter(pol_curve_plateau.pol_curve_data.average_current_density_A_cm2, pol_curve_plateau.pol_curve_data.average_voltage_V, label="plateaus")
    plt.legend()
    plt.savefig("UI.png")
