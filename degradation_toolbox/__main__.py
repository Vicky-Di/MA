from degradation_toolbox import Urc, example_data, deg_rate_multi_electrolyzers

import numpy as np

Urc_multi_cell = []
for cell in range(1, 3):
    # Prepare data: a Pandas DataFrame containing columns
    # "timestamp" (as index, formatted as datetime using pandas.to_datetime),
    # "currentDensity" in [A/cm2],
    # "temperature" in [degree C],
    # "voltage" in [V] (both cell voltage and average cell voltage are fine.)
    # Example datasets can be generated with the "example_data" function:
    data_cell = example_data(cell)

    # Calculated Urc and save plots (optional, controlled by 'plotSummary').
    # Default reference load levels are [0.6, 1.5]. It can be customized with "Iref".
    Urc_single_cell = Urc(data_cell, name=f"Cell_{cell}", configDict={'plotSummary': True})

    # Append the Urc results to a list.
    Urc_multi_cell.append(Urc_single_cell)

# Calculate degradation rate using Urc results of multiple cells.
degradation_multi_cell = deg_rate_multi_electrolyzers(Urc_multi_cell, name="Degradation rate of example cells")

# Get degradation results for a specific load level (e.g. 1.5A/cm2)
print("Degradation rate")
print(f"{round(degradation_multi_cell.degradation_results[1.5].aging_uV_per_h, 1)}uV/h")

print("Degradation rate uncertainty, 1 sigma")
print(f"{round(np.sqrt(degradation_multi_cell.degradation_results[1.5].uncertainty_uV2), 1)}uV/h")

print("Begin-of-life voltage")
# Not accurate. Because Urc is dedicated for calculating the degradation rate, not the absolute voltage level,
# which is impacted by the predefined reference condition. The official BOL voltage should refer to FAT or SAT results.
print(f"{round(degradation_multi_cell.degradation_results[1.5].bol_V, 2)}V")
