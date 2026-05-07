import pandas as pd
import matplotlib.pyplot as plt
import os
import datetime
from pathlib import Path
import sys

class GMpreprocess:
    def __init__(self, file_path, output_dir):
        """
        Initializes the pipeline for processing URC data.
        
        Args:
            file_path (str): Path to the input parquet file.
            output_dir (str): Directory where plots and results will be saved.
        """
        self.file_path = file_path
        self.output_dir = output_dir
        self.name = Path(file_path).stem 
        self.df = None
        self.urc_instance = None
        self.results = None

        # Create output directory if it does not exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f">> [Init] Created output directory: {self.output_dir}")

    def load_and_preprocess(self):
        """
        Loads the parquet file, detects column names dynamically, 
        renames them to standard formats, and performs unit conversion.
        """
        print(f"=== 1. Loading & Preprocessing: {self.name} ===")  
        
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File {self.file_path} not found.")

        try:
            self.df = pd.read_parquet(self.file_path)
            print(">> Data loaded successfully.")
        except Exception as e:
            raise Exception(f"Failed to read parquet file. Reason: {e}")

        # --- Column Detection Logic ---
        # Logic: Find 'Temp_XX_ID', extract ID, then find matching Voltage/Current
        module_id = None
        found_cols = {"temp": None, "volt": None, "curr": None}

        # 1. Identify Temperature Column and extract ID
        for col in self.df.columns:
            if col.startswith("Temp_"):
                module_id = col[-1] # Assuming ID is the last character
                found_cols["temp"] = col
                print(f"   [Detected] Temperature Col: '{col}' -> ID: '{module_id}'")
                break
        
        # 2. Identify Voltage and Current based on ID
        if module_id:
            for col in self.df.columns:
                if col.startswith("Mean_Voltage") and col.endswith(module_id):
                    found_cols["volt"] = col
                elif col.startswith("Current_Row") and col.endswith(module_id):
                    found_cols["curr"] = col
        
        # 3. Rename columns
        if all(found_cols.values()):
            rename_mapping = {
                found_cols["curr"]: "currentDensity",
                found_cols["temp"]: "temperature",
                found_cols["volt"]: "voltage",
                "TIMESTAMP": "timestamp"
            }
            self.df = self.df.rename(columns=rename_mapping)
            print(">> Columns renamed to standard format.")
        else:
            print(f"[Warning] Could not automatically map columns for ID {module_id}. Checking for standard names...")

        # 4. Set Index and Convert Units
        if "timestamp" in self.df.columns:
            self.df = self.df.set_index("timestamp")
        
        if "currentDensity" in self.df.columns:
            # calculate currentdensity: current (A) / size (5000 cm2)
            self.df["currentDensity"] = self.df["currentDensity"] / 5000.0
            
        # 5. Validation
        required = ['currentDensity', 'temperature', 'voltage']
        missing = [c for c in required if c not in self.df.columns]
        
        if missing:
            raise ValueError(f"Missing required columns: {missing}. Cannot proceed.")
        
        self.df = self.df[required]
        print(f">> Columns filtered. Retained: {required}")

    # def inspect_data(self):
    #     """
    #     Generates and saves an overview plot of the raw data.
    #     """
    #     if self.df is None or self.df.empty:
    #         print("[Skip] No data to plot.")
    #         return

    #     print("\n=== 2. Visualizing Data ===")
    #     fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
    #     axes[0].plot(self.df.index, self.df['voltage'], color='blue', linewidth=0.5)
    #     axes[0].set_ylabel('Voltage [V]')
    #     axes[0].set_title(f'{self.name} - Data Overview')
        
    #     axes[1].plot(self.df.index, self.df['currentDensity'], color='orange', linewidth=0.5)
    #     axes[1].set_ylabel('Current Density [A/cm2]')
        
    #     axes[2].plot(self.df.index, self.df['temperature'], color='red', linewidth=0.5)
    #     axes[2].set_ylabel('Temperature [°C]')
        
    #     plt.tight_layout()
        
    #     # Add timestamp to filename
    #     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    #     filename = f"{self.name}_raw_overview_{timestamp}.png"
    #     plot_path = os.path.join(self.output_dir, filename)
        
    #     plt.savefig(plot_path)
    #     plt.close(fig) # Close to free memory
    #     print(f">> Overview plot saved to: {plot_path}")

    # def run_algorithm(self): # not used any more
    #     """
    #     Instantiates the Urc class and runs the calculation.
    #     """
    #     print("\n=== 3. Running Urc Algorithm ===")
    #     if self.df is None:
    #         raise ValueError("Data not loaded. Call load_and_preprocess() first.")

    #     try:
    #         # Initialize Urc (Assuming it runs calculation upon initialization)
    #         self.urc_instance = Urc(
    #             data=self.df, 
    #             name=self.name, 
    #             configDict={'plotSummary': True}
    #         )
    #         print(">> Urc algorithm executed successfully.")
            
            # # Retrieve results
            # # Note: Verify if 'fitting_results' is the correct attribute in your toolbox
            # if hasattr(self.urc_instance, 'fitting_results'):
            #     self.results = self.urc_instance.fitting_results
            # else:
            #     print("[Info] 'fitting_results' attribute not found. Please check Urc class definition.")
            #     self.results = None 

        # except Exception as e:
        #     print(f"[Error] Urc execution failed: {e}")
        #     raise e

    def save_results(self):
        """
        Saves the preprocessd data to a parquet file.
        """
        print("\n=== 3. Saving Preprocessd data in .parquet format ===")
        if self.df is None:
            print("[Error] No preprocessd data to save.")
            return

        # Generate filename with date
        date_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_name = f"{self.name}_{date_str}.parquet"
        save_path = os.path.join(self.output_dir, save_name)

        try:
            self.df.to_parquet(save_path)
            print(f">> ✅ Final Results saved successfully to:\n   {save_path}")
        except Exception as e:
            print(f"[Error] Failed to save data: {e}")

    def run(self):
        """
        Master method to execute the full pipeline sequentially.
        """
        try:
            self.load_and_preprocess()
            # self.inspect_data()
            # self.run_algorithm()
            self.save_results()
            print("\n=== GMpreprocess Pipeline Completed Successfully ===")
            return self.df
        except Exception as e:
            print(f"\n[Terminated] Process stopped due to error: {e}")
            return None
