"""
Visualization and reporting functions for VITAL-CHD Bayesian analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import logging
from pathlib import Path

from config import FIGURES_DIR, TABLES_DIR
from inference import InferenceResults

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")


def setup_plot_style():
    """Set up consistent plotting style."""
    plt.rcParams.update({
        'figure.figsize': (12, 8),
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 11,
        'font.size': 11,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'figure.facecolor': 'white'
    })


def plot_hr_comparison(
    inference_results: Dict[str, InferenceResults],
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 10)
) -> None:
    """
    Create a comprehensive hazard ratio comparison plot.
    
    Args:
        inference_results: Dictionary of inference results
        output_path: Path to save the figure
        figsize: Figure size
    """
    if output_path is None:
        output_path = FIGURES_DIR / "fig_HR_by_prior.png"
    
    logger.info(f"Creating HR comparison plot with {len(inference_results)} priors")
    
    setup_plot_style()
    
    # Extract HR summaries
    hr_data = []
    for prior_name, result in inference_results.items():
        # Find HR variable with model prefix
        var_names = list(result.trace.posterior.data_vars)
        hr_var = [v for v in var_names if v.endswith('::hr_intervention')][0]
        hr_samples = result.trace.posterior[hr_var].values.flatten()
        hr_data.append({
            'prior_name': prior_name,
            'hr_mean': np.mean(hr_samples),
            'hr_median': np.median(hr_samples),
            'hr_q025': np.percentile(hr_samples, 2.5),
            'hr_q975': np.percentile(hr_samples, 97.5),
            'hr_q25': np.percentile(hr_samples, 25),
            'hr_q75': np.percentile(hr_samples, 75),
            'samples': hr_samples,
            'prior_type': 'LLM' if prior_name.startswith('llm_') else 'Existing'
        })
    
    df = pd.DataFrame(hr_data)
    
    # Sort by median HR
    df = df.sort_values('hr_median').reset_index(drop=True)
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # 1. Forest plot of HR estimates
    ax1 = fig.add_subplot(gs[0, :])
    
    colors = ['#1f77b4' if pt == 'Existing' else '#ff7f0e' for pt in df['prior_type']]
    y_positions = range(len(df))
    
    # Plot points and confidence intervals
    ax1.errorbar(
        df['hr_median'], y_positions,
        xerr=[df['hr_median'] - df['hr_q025'], df['hr_q975'] - df['hr_median']],
        fmt='o', capsize=4, capthick=2, markersize=8,
        color='black', alpha=0.7
    )
    
    # Color-code by prior type
    for i, (_, row) in enumerate(df.iterrows()):
        ax1.scatter(row['hr_median'], i, c=colors[i], s=80, zorder=5)
    
    # Add vertical line at HR = 1
    ax1.axvline(x=1, color='red', linestyle='--', alpha=0.7, linewidth=2)
    
    # Formatting
    ax1.set_yticks(y_positions)
    ax1.set_yticklabels(df['prior_name'], fontsize=10)
    ax1.set_xlabel('Hazard Ratio (95% CrI)', fontsize=12)
    ax1.set_title('Posterior Hazard Ratio Estimates by Prior', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', 
                   markersize=10, label='Existing Prior'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e', 
                   markersize=10, label='LLM Prior')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    # 2. Violin plot of posterior distributions
    ax2 = fig.add_subplot(gs[1, 0])
    
    # Prepare data for violin plot
    violin_data = []
    violin_labels = []
    for _, row in df.iterrows():
        # Sample subset for plotting (too many points can be slow)
        samples = row['samples']
        if len(samples) > 1000:
            samples = np.random.choice(samples, 1000, replace=False)
        violin_data.append(samples)
        violin_labels.append(row['prior_name'])
    
    parts = ax2.violinplot(violin_data, positions=range(len(violin_data)), 
                          showmeans=True, showmedians=True)
    
    # Color violins by prior type
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    ax2.axhline(y=1, color='red', linestyle='--', alpha=0.7)
    ax2.set_xticks(range(len(violin_labels)))
    ax2.set_xticklabels(violin_labels, rotation=45, ha='right', fontsize=9)
    ax2.set_ylabel('Hazard Ratio', fontsize=12)
    ax2.set_title('Posterior Distributions', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 3. P(HR < 1) comparison
    ax3 = fig.add_subplot(gs[1, 1])
    
    prob_hr_less_1 = [np.mean(row['samples'] < 1) for _, row in df.iterrows()]
    
    bars = ax3.bar(range(len(df)), prob_hr_less_1, color=colors, alpha=0.7)
    ax3.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
    ax3.set_xticks(range(len(df)))
    ax3.set_xticklabels(df['prior_name'], rotation=45, ha='right', fontsize=9)
    ax3.set_ylabel('P(HR < 1)', fontsize=12)
    ax3.set_title('Probability of Protective Effect', fontsize=12, fontweight='bold')
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, (bar, prob) in enumerate(zip(bars, prob_hr_less_1)):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{prob:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('VITAL-CHD Bayesian Analysis: Hazard Ratio Comparison by Prior', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved HR comparison plot to {output_path}")
    plt.close()


def plot_loo_comparison(
    predictive_summary: pd.DataFrame,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> None:
    """
    Create LOO comparison plot.
    
    Args:
        predictive_summary: DataFrame with predictive summary
        output_path: Path to save the figure
        figsize: Figure size
    """
    if output_path is None:
        output_path = FIGURES_DIR / "fig_LOO_by_prior.png"
    
    logger.info(f"Creating LOO comparison plot with {len(predictive_summary)} models")
    
    setup_plot_style()
    
    # Sort by LOO elpd
    df = predictive_summary.sort_values('loo_elpd', ascending=True).reset_index(drop=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Colors by prior type
    colors = ['#1f77b4' if pt == 'existing' else '#ff7f0e' for pt in df['prior_type']]
    
    # 1. LOO elpd with error bars
    y_positions = range(len(df))
    
    ax1.errorbar(
        df['loo_elpd'], y_positions,
        xerr=df['loo_se'],
        fmt='o', capsize=4, capthick=2, markersize=8,
        color='black', alpha=0.7
    )
    
    # Color-code points
    for i, (_, row) in enumerate(df.iterrows()):
        ax1.scatter(row['loo_elpd'], i, c=colors[i], s=80, zorder=5)
    
    ax1.set_yticks(y_positions)
    ax1.set_yticklabels(df['prior_name'], fontsize=10)
    ax1.set_xlabel('LOO elpd', fontsize=12)
    ax1.set_title('Leave-One-Out Expected Log Predictive Density', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add best model indicator
    best_idx = df['loo_elpd'].idxmax()
    ax1.annotate('Best', xy=(df.loc[best_idx, 'loo_elpd'], best_idx),
                xytext=(10, 0), textcoords='offset points',
                ha='left', va='center', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    # 2. ΔLOO from reference
    ax2.errorbar(
        df['loo_diff'], y_positions,
        xerr=df['loo_diff_se'],
        fmt='o', capsize=4, capthick=2, markersize=8,
        color='black', alpha=0.7
    )
    
    # Color-code points
    for i, (_, row) in enumerate(df.iterrows()):
        ax2.scatter(row['loo_diff'], i, c=colors[i], s=80, zorder=5)
    
    # Add vertical line at 0
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    ax2.set_yticks(y_positions)
    ax2.set_yticklabels(df['prior_name'], fontsize=10)
    ax2.set_xlabel('ΔLOO (vs Reference)', fontsize=12)
    ax2.set_title('LOO Difference from Reference Model', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', 
                   markersize=10, label='Existing Prior'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e', 
                   markersize=10, label='LLM Prior')
    ]
    ax1.legend(handles=legend_elements, loc='best')
    
    plt.suptitle('Model Comparison: Predictive Performance (LOO)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved LOO comparison plot to {output_path}")
    plt.close()


def plot_prior_posterior_comparison(
    inference_results: Dict[str, InferenceResults],
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 10)
) -> None:
    """
    Plot prior vs posterior distributions for log-HR.
    
    Args:
        inference_results: Dictionary of inference results
        output_path: Path to save the figure
        figsize: Figure size
    """
    if output_path is None:
        output_path = FIGURES_DIR / "fig_prior_posterior_comparison.png"
    
    logger.info(f"Creating prior-posterior comparison plot")
    
    setup_plot_style()
    
    n_priors = len(inference_results)
    n_cols = 3
    n_rows = int(np.ceil(n_priors / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    axes = axes.flatten()
    
    for i, (prior_name, result) in enumerate(inference_results.items()):
        ax = axes[i]
        
        # Get posterior samples (find variable with model prefix)
        var_names = list(result.trace.posterior.data_vars)
        log_hr_var = [v for v in var_names if v.endswith('::log_hr_intervention')][0]
        posterior_samples = result.trace.posterior[log_hr_var].values.flatten()
        
        # Plot posterior distribution
        ax.hist(posterior_samples, bins=30, density=True, alpha=0.7, 
               label='Posterior', color='steelblue')
        
        # Plot prior distribution
        prior_mu = result.model.log_hr_prior.mu
        prior_sigma = result.model.log_hr_prior.sigma
        
        x_range = np.linspace(
            min(posterior_samples.min(), prior_mu - 3*prior_sigma),
            max(posterior_samples.max(), prior_mu + 3*prior_sigma),
            200
        )
        
        prior_density = (1 / (prior_sigma * np.sqrt(2 * np.pi))) * \
                       np.exp(-0.5 * ((x_range - prior_mu) / prior_sigma) ** 2)
        
        ax.plot(x_range, prior_density, 'r-', linewidth=2, label='Prior')
        
        # Add vertical lines for means
        ax.axvline(np.mean(posterior_samples), color='steelblue', 
                  linestyle='--', alpha=0.8, label='Post. Mean')
        ax.axvline(prior_mu, color='red', linestyle='--', alpha=0.8, label='Prior Mean')
        
        ax.set_title(f'{prior_name}', fontsize=11)
        ax.set_xlabel('log-HR', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_priors, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Prior vs Posterior Distributions (log-HR)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved prior-posterior comparison to {output_path}")
    plt.close()


def create_summary_report(
    inference_summary: pd.DataFrame,
    predictive_summary: pd.DataFrame,
    output_path: Optional[Path] = None
) -> None:
    """
    Create a comprehensive summary report as a formatted text file.
    
    Args:
        inference_summary: Inference results summary
        predictive_summary: Predictive results summary
        output_path: Path to save the report
    """
    if output_path is None:
        output_path = TABLES_DIR / "summary_report.txt"
    
    logger.info("Creating summary report")
    
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("VITAL-CHD BAYESIAN RE-ANALYSIS SUMMARY REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    # Dataset summary
    report_lines.append("DATASET SUMMARY")
    report_lines.append("-" * 20)
    report_lines.append(f"Number of models compared: {len(inference_summary)}")
    
    existing_priors = inference_summary[~inference_summary['prior_name'].str.startswith('llm_')]
    llm_priors = inference_summary[inference_summary['prior_name'].str.startswith('llm_')]
    
    report_lines.append(f"Existing priors: {len(existing_priors)}")
    report_lines.append(f"LLM priors: {len(llm_priors)}")
    report_lines.append("")
    
    # Inference results summary
    report_lines.append("INFERENCE RESULTS")
    report_lines.append("-" * 20)
    
    # Best HR estimate (lowest median)
    best_hr = inference_summary.loc[inference_summary['hr_median'].idxmin()]
    report_lines.append(f"Most protective effect: {best_hr['prior_name']}")
    report_lines.append(f"  HR = {best_hr['hr_median']:.3f} (95% CrI: {best_hr['hr_q025']:.3f}-{best_hr['hr_q975']:.3f})")
    report_lines.append(f"  P(HR < 1) = {best_hr['prob_hr_less_than_1']:.3f}")
    report_lines.append("")
    
    # Highest P(HR < 1)
    best_prob = inference_summary.loc[inference_summary['prob_hr_less_than_1'].idxmax()]
    if best_prob['prior_name'] != best_hr['prior_name']:
        report_lines.append(f"Highest P(HR < 1): {best_prob['prior_name']}")
        report_lines.append(f"  P(HR < 1) = {best_prob['prob_hr_less_than_1']:.3f}")
        report_lines.append("")
    
    # Convergence summary
    max_rhat = inference_summary['max_rhat'].max()
    min_ess = inference_summary['min_ess_bulk'].min()
    n_divergent = inference_summary['n_divergent'].sum()
    
    report_lines.append("CONVERGENCE DIAGNOSTICS")
    report_lines.append("-" * 25)
    report_lines.append(f"Maximum R-hat across all models: {max_rhat:.4f}")
    report_lines.append(f"Minimum ESS across all models: {min_ess:.0f}")
    report_lines.append(f"Total divergent transitions: {n_divergent}")
    
    if max_rhat <= 1.01:
        report_lines.append("✓ All models show good convergence (R-hat ≤ 1.01)")
    else:
        report_lines.append("⚠ Some models show convergence issues (R-hat > 1.01)")
    report_lines.append("")
    
    # Predictive performance
    report_lines.append("PREDICTIVE PERFORMANCE")
    report_lines.append("-" * 24)
    
    best_loo = predictive_summary.loc[predictive_summary['loo_elpd'].idxmax()]
    report_lines.append(f"Best predictive performance: {best_loo['prior_name']}")
    report_lines.append(f"  LOO elpd = {best_loo['loo_elpd']:.2f} ± {best_loo['loo_se']:.2f}")
    report_lines.append("")
    
    # LLM vs existing comparison
    if len(llm_priors) > 0:
        report_lines.append("LLM VS EXISTING PRIORS COMPARISON")
        report_lines.append("-" * 35)
        
        existing_mean_loo = existing_priors['loo_elpd'].mean()
        llm_mean_loo = llm_priors['loo_elpd'].mean()
        
        report_lines.append(f"Average LOO elpd - Existing: {existing_mean_loo:.2f}")
        report_lines.append(f"Average LOO elpd - LLM: {llm_mean_loo:.2f}")
        
        if llm_mean_loo > existing_mean_loo:
            report_lines.append("✓ LLM priors show better average predictive performance")
        else:
            report_lines.append("✗ Existing priors show better average predictive performance")
        
        # Count how many LLM priors are in top half
        n_total = len(predictive_summary)
        llm_in_top_half = len(llm_priors[llm_priors['loo_rank'] <= n_total/2])
        report_lines.append(f"LLM priors in top half of rankings: {llm_in_top_half}/{len(llm_priors)}")
        report_lines.append("")
    
    # Detailed results table
    report_lines.append("DETAILED RESULTS")
    report_lines.append("-" * 16)
    report_lines.append("")
    
    # Merge inference and predictive results
    merged = inference_summary.merge(
        predictive_summary[['prior_name', 'loo_elpd', 'loo_rank', 'loo_diff']], 
        on='prior_name'
    )
    
    merged = merged.sort_values('loo_rank')
    
    # Create formatted table
    report_lines.append("Rank | Prior Name              | HR (95% CrI)        | P(HR<1) | LOO elpd | ΔLOO")
    report_lines.append("-" * 85)
    
    for _, row in merged.iterrows():
        rank = f"{row['loo_rank']:2.0f}"
        name = f"{row['prior_name'][:22]:22s}"
        hr_ci = f"{row['hr_median']:.3f} ({row['hr_q025']:.3f}-{row['hr_q975']:.3f})"
        prob = f"{row['prob_hr_less_than_1']:.3f}"
        loo = f"{row['loo_elpd']:8.1f}"
        delta_loo = f"{row['loo_diff']:6.1f}"
        
        line = f"{rank:>4} | {name} | {hr_ci:>18} | {prob:>7} | {loo} | {delta_loo}"
        report_lines.append(line)
    
    report_lines.append("")
    report_lines.append("=" * 80)
    
    # Save report
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write('\n'.join(report_lines))
    
    logger.info(f"Saved summary report to {output_path}")


def generate_predictive_plots(
    predictive_summary: pd.DataFrame,
    output_dir: Optional[Path] = None
) -> None:
    """
    Generate plots for predictive evaluation only.
    
    Args:
        predictive_summary: Predictive summary DataFrame
        output_dir: Directory to save plots (default: FIGURES_DIR)
    """
    if output_dir is None:
        output_dir = FIGURES_DIR
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating predictive plots in {output_dir}")
    
    # LOO comparison plot
    plot_loo_comparison(
        predictive_summary, 
        output_dir / "fig_LOO_by_prior.png"
    )
    
    logger.info("Predictive plots generated successfully")


def generate_all_plots(
    inference_results: Dict[str, InferenceResults],
    predictive_summary: pd.DataFrame,
    output_dir: Optional[Path] = None
) -> None:
    """
    Generate all standard plots for the analysis.
    
    Args:
        inference_results: Dictionary of inference results
        predictive_summary: Predictive summary DataFrame
        output_dir: Directory to save plots (default: FIGURES_DIR)
    """
    if output_dir is None:
        output_dir = FIGURES_DIR
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Generating all plots in {output_dir}")
    
    # 1. HR comparison plot
    plot_hr_comparison(
        inference_results, 
        output_dir / "fig_HR_by_prior.png"
    )
    
    # 2. LOO comparison plot
    plot_loo_comparison(
        predictive_summary, 
        output_dir / "fig_LOO_by_prior.png"
    )
    
    # 3. Prior-posterior comparison
    plot_prior_posterior_comparison(
        inference_results, 
        output_dir / "fig_prior_posterior_comparison.png"
    )
    
    logger.info("All plots generated successfully")


if __name__ == "__main__":
    # Test plotting functions with dummy data
    try:
        import numpy as np
        from config import SEED
        
        logger.info("Testing plotting functions with dummy data")
        
        # Create dummy predictive summary
        dummy_summary = pd.DataFrame({
            'prior_name': ['noninformative', 'primary_informed', 'llm_test'],
            'prior_type': ['existing', 'existing', 'llm'],
            'loo_elpd': [-1500, -1490, -1485],
            'loo_se': [50, 48, 52],
            'loo_diff': [-10, 0, 5],
            'loo_diff_se': [5, 0, 3],
            'loo_rank': [3, 2, 1]
        })
        
        # Test LOO plot
        plot_loo_comparison(dummy_summary, figsize=(10, 6))
        
        logger.info("Plotting tests completed successfully")
        
    except Exception as e:
        print(f"Error testing plotting functions: {e}")
