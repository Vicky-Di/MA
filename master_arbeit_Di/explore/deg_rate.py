"""Utilities for aggregate degradation-rate estimation across electrolyzers.

This module combines per-cell Urc degradation outputs into weighted aggregate
degradation rates and optional diagnostic plots.
"""

import logging
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from degradation_toolbox.Urc.Urc import Urc
from degradation_toolbox.Urc.helpers import degradation_result


logger = logging.getLogger(__name__)


class deg_rate_multi_electrolyzers:
    """Aggregate degradation-rate estimates across multiple Urc instances."""

    def __init__(self, Urc_multi_electrolyzers_list, name="Degradation rate of multi cells", plot=True):
        # TODO: check all cells have the same reference condition: Tref, OHref.
        self.name = name

        # If only one Urc instance is provided, not a list --> turn it into a list.
        if isinstance(Urc_multi_electrolyzers_list, Urc):
            Urc_multi_electrolyzers_list = [Urc_multi_electrolyzers_list]

        if len(Urc_multi_electrolyzers_list) == 1:
            logger.warning(
                "Only one cell was provided. Combined-rate uncertainty may be underestimated "
                "because inter-cell variation is unavailable."
            )

        if plot:
            plt.figure()
            if not os.path.exists("plots"):
                os.makedirs("plots")

        # Find Iref for all cells
        i_all = set()
        for cell in Urc_multi_electrolyzers_list:
            i_all.update(cell.Iref)

        self.degradation_results = dict()

        for i in i_all:
            aging_uV_per_h, bol_V, uncertainty_uV2, weights = [], [], [], []
            x_all = np.array([])  # for final plotting

            for cell in Urc_multi_electrolyzers_list:
                try:
                    deg_result = cell.degradation_results[i]
                    cell.get_Urc_results(i)
                    aging_uV_per_h.append(deg_result.aging_uV_per_h)
                    bol_V.append(deg_result.bol_V)
                    uncertainty_uV2.append(deg_result.uncertainty_uV2)
                    time_coverage = max(cell.Urc_results.calh) - min(cell.Urc_results.calh)
                    # TODO: consider time-coverage treatment, aligned with Urc.py.
                    weights.append(1 / deg_result.uncertainty_uV2 * len(cell.Urc_results) * (time_coverage ** 2))

                    if plot:
                        x = cell.Urc_results.calh
                        y = cell.Urc_results.Urc
                        se = cell.Urc_results.Urc_se
                        x_all = np.append(x_all, x)
                        errorbar_plot = plt.errorbar(x, y,
                                                    yerr=se,
                                                    linestyle="", marker=".")
                        # Get the color of the errorbar plot
                        color = errorbar_plot.lines[0].get_color()
                        plt.plot(x, x * deg_result.aging_uV_per_h / 1e6 + deg_result.bol_V, linestyle="--", color=color,
                                label=rf"{cell.name}: {round(deg_result.aging_uV_per_h, 2)} (+-{round(np.sqrt(deg_result.uncertainty_uV2), 2)}) uV/calh")
                except Exception:
                    pass

            # Combine results using weighted average. Same codes as in Urc.py where Urc1 and Urc2 results are combined.
            # Numpy.average (https://numpy.org/doc/stable/reference/generated/numpy.average.html):
            # avg = sum(a * weights) / sum(weights)
            aging_uV_per_h_combined = np.average(aging_uV_per_h, weights=weights)
            bol_V_combined = np.average(bol_V, weights=weights)

            # Lennard: "the average has two contributions to the net uncertainty:
            # 1. the spread of individual observations --> uncertainty_spread
            # 2. the averaged uncertainty of the individual observations" --> uncertainty_average

            # Weighted average of the uncertainties of Urc1 and Urc2
            weights_normalized = np.array(weights) / sum(weights)
            uncertainty_average = sum((weights_normalized**2) * uncertainty_uV2)

            # Differences of degradation rates between Urc1 and Urc2
            uncertainty_spread = sum(weights_normalized * ((np.array(aging_uV_per_h) - aging_uV_per_h_combined) ** 2))
            uncertainty_combined_uV2 = uncertainty_average + uncertainty_spread

            self.degradation_results[i] = degradation_result(aging_uV_per_h_combined,
                                                              bol_V_combined,
                                                              uncertainty_combined_uV2)

            if plot:
                plt.plot(x_all, aging_uV_per_h_combined / 1e6 * x_all + bol_V_combined, color="k", linestyle="--",
                        label=rf"Overall: {round(aging_uV_per_h_combined, 2)} (+-{round(np.sqrt(uncertainty_combined_uV2), 2)}) uV/calh")
                plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                plt.xlabel("Calendar hours since test start [h]")
                plt.ylabel("U$_{rc}$ [V]")
                if len(Urc_multi_electrolyzers_list) == 1:
                    plt.title(f"{name} @ {i}A/cm2\n(Only 1 voltage --> uncertainty underestimated!)")
                else:
                    plt.title(f"{name} @ {i}A/cm2")
                plt.grid()
                plt.savefig(f"plots/{name}_{i}A_cm2.png", bbox_inches='tight')
                plt.close()  # If I don't close the figure, plots of multiple load levels will pile up in one figure.
                logger.info("Saved plot: plots/%s_%sA_cm2.png", name, i)
        return


if __name__ == '__main__':
    from Urc import Urc
    import pickle

    # data1 = example_data(1)
    # electrolyzer1 = Urc(data1, name="Example 1")
    # electrolyzer1.plot_Urc1_Urc2_before_offset()
    #
    # data2 = example_data(2)
    # electrolyzer2 = Urc(data2, name="Example 2")
    # electrolyzer2.plot_Urc1_Urc2_before_offset()
    #
    # multi_cells([electrolyzer1, electrolyzer2])

    file_path = os.path.dirname(__file__)
    file_path = os.path.join(file_path, '../tests/Linz_M12_min')
    data = pd.read_parquet(file_path)
    data["currentDensity"] = data.I / 5000
    results_multi_cell = []

    # all_cells = [f"U_cell_{i}" for i in range(1, 51)]
    # data["U_ave"] = data[all_cells].mean(axis=1)
    # data_cell = data[["currentDensity", "T_stack", "U_ave"]].copy()
    # data_cell.rename(columns={"T_stack": "temperature", "U_ave": "voltage"},
    #                      inplace=True)
    # results_single_cell = Urc(data_cell, name=f"Linz_M12_U_ave", configDict={'plotSummary':True})
    # results_single_cell.plot_Urc1_Urc2_separately()
    # results_multi_cell.append(results_single_cell)

    for cell in [1, 25]:  # ~1.5 min for each cell with ~8000h operation.
        # Organized data into the required format.
        data_cell = data[["currentDensity", "T_stack", f"U_cell_{cell}"]].copy()
        data_cell.rename(columns={"T_stack": "temperature", f"U_cell_{cell}": "voltage"},
                         inplace=True)
        # Calculated Urc and save plots (optional, controlled by 'plotSummary').
        results_single_cell = Urc(data_cell, name=f"Cell_{cell}", configDict={'plotSummary': True})
        # Save results
        with open(f"Cell_{cell}.pickle", 'wb') as f:
            pickle.dump(results_single_cell, f)
        # Append the Urc results to a list.
        results_multi_cell.append(results_single_cell)
    # Calculate degradation rate using Urc results of multiple cells.
    deg_rate_multi_electrolyzers(results_multi_cell, name="Degradation rate of Linz_M12")
    # TODO: Add README example and packaging notes for this workflow.
