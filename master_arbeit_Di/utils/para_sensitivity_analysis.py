"""
Parameter Sensitivity Analysis Module (OAT - One-at-a-Time)

Provides a class to run, evaluate, and visualize parameter sensitivity
for the Urc Adaptive model.

Two sensitivity metrics are computed:
  - CV% (Coefficient of Variation): measures raw output variability
  - Elasticity: measures output response normalized by parameter change
"""

import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from tqdm.notebook import tqdm
from typing import Dict, List, Optional, Any

from degradation_toolbox.Urc.Urc import Urc
from utils.stability_evaluation import evaluate_urc_stability


# ── Default evaluation metrics used throughout ──────────────────────────────
DEFAULT_METRICS = [
    'Data Points (n)',
    'RMSE (mV)',
    'Mono (Rank) [0-1]',
    'Slope Sigma (uV/h)',
    'Outlier (%)',
]

# ── Plot colour palette (by reference current) ─────────────────────────────
IREF_COLORS = {
    '0.6': '#636EFA',
    '1.0': '#EF553B',
    '1.5': '#00CC96',
}


class ParameterSensitivityAnalysis:
    """
    One-at-a-Time (OAT) parameter sensitivity analysis for the Urc Adaptive model.

    Typical workflow
    ----------------
    >>> psa = ParameterSensitivityAnalysis(data, name, COMMON_CONFIG, ADAPTIVE_DEFAULT, PARAM_GRIDS, IREF_LIST)
    >>> psa.run_all_experiments()       # run experiments
    >>> psa.calculate_sensitivity()     # compute CV & Elasticity
    >>> psa.plot_all_param_results()    # per-parameter line plots
    >>> psa.plot_heatmap()              # combined heatmap
    >>> psa.plot_ranking()              # bar-chart ranking
    >>> psa.save_results()              # export CSVs
    """

    def __init__(
        self,
        data: pd.DataFrame,
        name: str,
        common_config: dict,
        adaptive_default: dict,
        param_grids: Dict[str, list],
        iref_list: List[float],
        metrics_cols: Optional[List[str]] = None,
    ):
        """
        Parameters
        ----------
        data : pd.DataFrame
            Pre-processed measurement data.
        name : str
            Dataset / stack identifier (e.g. 'G6M2').
        common_config : dict
            Model configuration shared by all runs (Iref, Tref, filters, …).
        adaptive_default : dict
            Default hyper-parameters for the Adaptive method.
            Also serves as the **baseline** for Elasticity calculation.
        param_grids : dict
            ``{param_name: [value1, value2, ...]}`` – search space per parameter.
        iref_list : list of float
            Reference current densities to evaluate.
        metrics_cols : list of str, optional
            Which metrics to include in sensitivity indices.
            Defaults to ``DEFAULT_METRICS``.
        """
        self.data = data
        self.name = name
        self.common_config = common_config
        self.adaptive_default = adaptive_default
        self.param_grids = param_grids
        self.iref_list = iref_list
        self.metrics_cols = metrics_cols or DEFAULT_METRICS

        # ── Results containers ──
        self.experiment_results: Dict[str, pd.DataFrame] = {}
        self.baseline_metrics: Dict[float, dict] = {}
        self.sensitivity_cv: Optional[pd.DataFrame] = None
        self.sensitivity_elasticity: Optional[pd.DataFrame] = None
        self.sensitivity_df: Optional[pd.DataFrame] = None  # combined

    # ================================================================
    #  1.  Train Baseline
    # ================================================================
    def train_baseline(self) -> Dict[float, dict]:
        """Train the *baseline* (expanding-window) model and extract metrics."""
        print("Training Baseline model …")
        cfg = self.common_config.copy()
        cfg["method"] = "baseline"
        urc_bl = Urc(data=self.data, name=self.name, **cfg)

        self.baseline_metrics = {}
        for i_ref in self.iref_list:
            m = evaluate_urc_stability(urc_bl, i_ref)
            if m:
                self.baseline_metrics[i_ref] = m

        print("  Baseline metrics collected for", list(self.baseline_metrics.keys()))
        return self.baseline_metrics

    # ================================================================
    #  2.  Run Experiments
    # ================================================================
    def run_experiment(self, param_name: str) -> pd.DataFrame:
        """
        Run the OAT experiment for a single parameter.

        Returns a DataFrame with one row per (param_value, i_ref) combination.
        """
        param_values = self.param_grids[param_name]
        results = []

        print(f"\n{'=' * 60}")
        print(f"Testing parameter: {param_name}")
        print(f"Values to test:    {param_values}")
        print(f"{'=' * 60}")

        for value in tqdm(param_values, desc=f"Testing {param_name}"):
            cfg = self.common_config.copy()
            cfg.update(self.adaptive_default.copy())
            cfg[param_name] = value

            try:
                urc_model = Urc(data=self.data, name=self.name, **cfg)
                for i_ref in self.iref_list:
                    m = evaluate_urc_stability(urc_model, i_ref)
                    if m:
                        m['param_name'] = param_name
                        m['param_value'] = value
                        m['i_ref'] = i_ref
                        results.append(m)
            except Exception as e:
                print(f"  ⚠️  Error with {param_name}={value}: {e}")

        df = pd.DataFrame(results)
        self.experiment_results[param_name] = df
        print(f"  → Collected {len(df)} rows")
        return df

    def run_all_experiments(self) -> Dict[str, pd.DataFrame]:
        """Run OAT experiments for **all** parameters in ``param_grids``."""
        if not self.baseline_metrics:
            self.train_baseline()

        for p in self.param_grids:
            self.run_experiment(p)
        return self.experiment_results

    # ================================================================
    #  3.  Sensitivity Indices
    # ================================================================
    def _calc_cv(self) -> pd.DataFrame:
        """
        CV% = std(metric) / mean(metric) × 100

        Measures how much the metric **varies** across the tested parameter
        values.  Does NOT account for the magnitude of parameter change.
        """
        rows = []
        for param_name, df in self.experiment_results.items():
            for metric in self.metrics_cols:
                if metric not in df.columns:
                    continue
                grouped = df.groupby('param_value')[metric].mean()
                mean_val = grouped.mean()
                cv = (grouped.std() / mean_val * 100) if mean_val != 0 else 0.0
                rows.append({
                    'Parameter': param_name,
                    'Metric': metric,
                    'CV (%)': round(cv, 2),
                    'Min': round(grouped.min(), 3),
                    'Max': round(grouped.max(), 3),
                    'Range': round(grouped.max() - grouped.min(), 3),
                })
        return pd.DataFrame(rows)

    def _calc_elasticity(self) -> pd.DataFrame:
        """
        Elasticity = mean | (ΔMetric / Metric_base) / (ΔParam / Param_base) |

        Measures how a **1 % change in the parameter** translates into
        a percentage change in the metric.  Accounts for the magnitude of
        parameter change, enabling fair cross-parameter comparison.
        """
        rows = []
        for param_name, df in self.experiment_results.items():
            base_param = self.adaptive_default.get(param_name)
            if base_param is None or base_param == 0:
                continue

            for metric in self.metrics_cols:
                if metric not in df.columns:
                    continue

                # Average metric per param value (across all Iref)
                grouped = df.groupby('param_value')[metric].mean()

                # Baseline metric value (at default param value)
                if base_param in grouped.index:
                    base_metric = grouped.loc[base_param]
                else:
                    # Fallback: use the mean as baseline
                    base_metric = grouped.mean()

                if base_metric == 0:
                    rows.append({
                        'Parameter': param_name,
                        'Metric': metric,
                        'Elasticity (%)': 0.0,
                    })
                    continue

                elasticities = []
                for val, met in grouped.items():
                    if val == base_param:
                        continue
                    dp = (val - base_param) / base_param        # relative param change
                    dm = (met - base_metric) / base_metric      # relative metric change
                    if dp != 0:
                        elasticities.append(abs(dm / dp) * 100)

                avg_e = float(np.mean(elasticities)) if elasticities else 0.0
                rows.append({
                    'Parameter': param_name,
                    'Metric': metric,
                    'Elasticity (%)': round(avg_e, 2),
                })
        return pd.DataFrame(rows)

    def calculate_sensitivity(self) -> pd.DataFrame:
        """
        Compute **both** CV and Elasticity indices and merge them into
        a single DataFrame stored in ``self.sensitivity_df``.
        """
        self.sensitivity_cv = self._calc_cv()
        self.sensitivity_elasticity = self._calc_elasticity()

        self.sensitivity_df = self.sensitivity_cv.merge(
            self.sensitivity_elasticity,
            on=['Parameter', 'Metric'],
            how='left',
        )
        self.sensitivity_df['Elasticity (%)'] = self.sensitivity_df['Elasticity (%)'].fillna(0)

        return self.sensitivity_df

    # ================================================================
    #  4.  Plotting Helpers
    # ================================================================
    def plot_param_results(
        self,
        param_name: str,
        title_suffix: str = "(Dashed = Baseline)",
    ) -> go.Figure:
        """Line-chart grid (5 metrics) for a single parameter."""
        df = self.experiment_results.get(param_name)
        if df is None or df.empty:
            print(f"No results for {param_name}")
            return go.Figure()

        metrics_to_plot = [
            ('RMSE (mV)',            'RMSE',               'lower'),
            ('Slope Sigma (uV/h)',   'Slope Uncertainty',   'lower'),
            ('Outlier (%)',          'Outlier %',           'lower'),
            ('Mono (Rank) [0-1]',   'Monotonicity',        'higher'),
            ('Data Points (n)',     'Data Points',         'higher'),
        ]

        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[m[1] for m in metrics_to_plot] + [''],
            vertical_spacing=0.15,
            horizontal_spacing=0.08,
        )

        for idx, (col, label, better) in enumerate(metrics_to_plot):
            row, c = idx // 3 + 1, idx % 3 + 1
            for i_ref in self.iref_list:
                df_i = df[df['i_ref'] == i_ref]
                if df_i.empty or col not in df_i.columns:
                    continue
                clr = IREF_COLORS.get(str(i_ref), '#888')
                fig.add_trace(
                    go.Scatter(
                        x=df_i['param_value'], y=df_i[col],
                        mode='lines+markers',
                        name=f'{i_ref} A/cm²',
                        line=dict(color=clr),
                        legendgroup=str(i_ref),
                        showlegend=(idx == 0),
                    ),
                    row=row, col=c,
                )
                # Baseline reference line
                bl = self.baseline_metrics.get(i_ref, {})
                if col in bl:
                    fig.add_hline(
                        y=bl[col], line_dash='dash',
                        line_color=clr, opacity=0.5,
                        row=row, col=c,
                    )
            arrow = '↑' if better == 'higher' else '↓'
            fig.update_yaxes(title_text=f"{label} ({arrow} better)", row=row, col=c)

        fig.update_layout(
            height=600,
            title_text=f"Sensitivity Analysis: {param_name} {title_suffix}",
            template='plotly_white',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        )
        for c in [1, 2, 3]:
            fig.update_xaxes(title_text=param_name, row=2, col=c)

        fig.show()
        return fig

    def plot_all_param_results(self) -> Dict[str, go.Figure]:
        """Plot line-chart grids for every tested parameter."""
        figs = {}
        for p in self.experiment_results:
            figs[p] = self.plot_param_results(p)
        return figs

    # ── Heatmap ─────────────────────────────────────────────────────
    def plot_heatmap(self, method: str = 'both') -> go.Figure:
        """
        Plot a sensitivity heatmap.

        Parameters
        ----------
        method : str
            ``'cv'``  – CV% only,
            ``'elasticity'`` – Elasticity only,
            ``'both'`` – side-by-side subplots (default).
        """
        if self.sensitivity_df is None:
            self.calculate_sensitivity()

        if method == 'both':
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=[
                    'CV% (Coefficient of Variation)',
                    'Elasticity (% metric change per 1% param change)',
                ],
                horizontal_spacing=0.12,
            )
            pivot_cv = self.sensitivity_df.pivot(
                index='Parameter', columns='Metric', values='CV (%)')
            pivot_el = self.sensitivity_df.pivot(
                index='Parameter', columns='Metric', values='Elasticity (%)')

            fig.add_trace(
                go.Heatmap(
                    z=pivot_cv.values, x=pivot_cv.columns, y=pivot_cv.index,
                    colorscale='RdYlBu_r', text=np.round(pivot_cv.values, 2),
                    texttemplate='%{text}', colorbar=dict(title='CV%', x=0.45),
                ),
                row=1, col=1,
            )
            fig.add_trace(
                go.Heatmap(
                    z=pivot_el.values, x=pivot_el.columns, y=pivot_el.index,
                    colorscale='RdYlBu_r', text=np.round(pivot_el.values, 2),
                    texttemplate='%{text}', colorbar=dict(title='Elast%', x=1.0),
                ),
                row=1, col=2,
            )
            fig.update_layout(
                height=420, width=1200,
                title_text=f"Parameter Sensitivity Heatmap – {self.name}",
                template='plotly_white',
            )
        else:
            value_col = 'CV (%)' if method == 'cv' else 'Elasticity (%)'
            title_label = 'CV% (Coefficient of Variation)' if method == 'cv' \
                else 'Elasticity (% metric change per 1% param change)'
            pivot = self.sensitivity_df.pivot(
                index='Parameter', columns='Metric', values=value_col)

            fig = px.imshow(
                pivot,
                labels=dict(x='Metric', y='Parameter', color=value_col),
                title=f"Parameter Sensitivity Heatmap: {title_label} – {self.name}",
                color_continuous_scale='RdYlBu_r',
                aspect='auto',
                text_auto='.2f',
            )
            fig.update_layout(height=420, width=800, template='plotly_white')

        fig.show()
        return fig

    # ── Ranking Bar Chart ───────────────────────────────────────────
    def plot_ranking(self, method: str = 'both') -> go.Figure:
        """
        Bar chart of average sensitivity per parameter.

        Parameters
        ----------
        method : ``'cv'`` | ``'elasticity'`` | ``'both'`` (default)
        """
        if self.sensitivity_df is None:
            self.calculate_sensitivity()

        if method == 'both':
            avg_cv = (self.sensitivity_df.groupby('Parameter')['CV (%)']
                      .mean().sort_values(ascending=False))
            avg_el = (self.sensitivity_df.groupby('Parameter')['Elasticity (%)']
                      .mean().sort_values(ascending=False))

            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=['Mean CV (%)', 'Mean Elasticity (%)'],
                horizontal_spacing=0.15,
            )
            fig.add_trace(
                go.Bar(
                    x=avg_cv.index, y=avg_cv.values,
                    text=[f'{v:.2f}%' for v in avg_cv.values],
                    textposition='auto',
                    marker=dict(color=avg_cv.values, colorscale='RdYlBu_r'),
                ),
                row=1, col=1,
            )
            fig.add_trace(
                go.Bar(
                    x=avg_el.index, y=avg_el.values,
                    text=[f'{v:.2f}%' for v in avg_el.values],
                    textposition='auto',
                    marker=dict(color=avg_el.values, colorscale='RdYlBu_r'),
                ),
                row=1, col=2,
            )
            fig.update_layout(
                height=420, width=1000,
                title_text=f"Parameter Sensitivity Ranking – {self.name}",
                template='plotly_white', showlegend=False,
            )
        else:
            value_col = 'CV (%)' if method == 'cv' else 'Elasticity (%)'
            avg = (self.sensitivity_df.groupby('Parameter')[value_col]
                   .mean().sort_values(ascending=False))

            fig = go.Figure(go.Bar(
                x=avg.index, y=avg.values,
                text=[f'{v:.2f}%' for v in avg.values],
                textposition='auto',
                marker=dict(color=avg.values, colorscale='RdYlBu_r', showscale=True,
                            colorbar=dict(title=value_col)),
            ))
            title_label = 'CV%' if method == 'cv' else 'Elasticity'
            fig.update_layout(
                title_text=f"Parameter Sensitivity Ranking ({title_label}) – {self.name}",
                xaxis_title='Parameter', yaxis_title=f'Mean {value_col}',
                template='plotly_white', height=420, width=800,
            )

        fig.show()
        return fig

    # ================================================================
    #  5.  Print Summary
    # ================================================================
    def print_summary(self):
        """Print a concise text summary of both CV and Elasticity rankings."""
        if self.sensitivity_df is None:
            self.calculate_sensitivity()

        for col, label in [('CV (%)', 'CV%'), ('Elasticity (%)', 'Elasticity')]:
            avg = (self.sensitivity_df.groupby('Parameter')[col]
                   .mean().sort_values(ascending=False))
            print(f"\n{'=' * 60}")
            print(f"  Ranking by {label} (higher = more sensitive)")
            print(f"{'=' * 60}")
            for i, (p, v) in enumerate(avg.items(), 1):
                print(f"    {i}. {p:35s} {v:8.2f}%")

        print(f"\n💡 CV%  = raw output variability (ignores param range).")
        print(f"   Elasticity = output change per 1% param change (normalised).")

    # ================================================================
    #  6.  Save / Export
    # ================================================================
    def save_results(self, output_dir: str = '.') -> None:
        """Save experiment data and sensitivity indices to CSV."""
        if self.sensitivity_df is None:
            self.calculate_sensitivity()

        # All raw experiment rows
        all_exp = pd.concat(self.experiment_results.values(), ignore_index=True)
        exp_path = f"{output_dir}/sensitivity_analysis_{self.name}.csv"
        all_exp.to_csv(exp_path, index=False)
        print(f"✅ Experiment results  → {exp_path}")

        # Sensitivity indices
        idx_path = f"{output_dir}/sensitivity_index_{self.name}.csv"
        self.sensitivity_df.to_csv(idx_path, index=False)
        print(f"✅ Sensitivity indices → {idx_path}")
