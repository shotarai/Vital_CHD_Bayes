"""
Analysis and visualization for holdout-based small-n sensitivity results.

This module provides functions to analyze and visualize the results of the
holdout-based small-n sensitivity evaluation.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional
from pathlib import Path
import logging

from config import TABLES_DIR, FIGURES_DIR

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")


def create_learning_curves_plot(results_df: pd.DataFrame, 
                              output_dir: Optional[Path] = None) -> None:
    """
    Create learning curves showing C-index and IBS vs training fraction.
    
    Args:
        results_df: Results DataFrame from holdout sensitivity analysis
        output_dir: Directory to save plots (default: FIGURES_DIR)
    """
    if output_dir is None:
        output_dir = FIGURES_DIR
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Creating learning curves")
    
    # Set up the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Color and style mapping
    existing_priors = results_df[~results_df['prior_name'].str.startswith('llm_')]['prior_name'].unique()
    llm_priors = results_df[results_df['prior_name'].str.startswith('llm_')]['prior_name'].unique()
    
    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(existing_priors) + len(llm_priors)))
    
    # Plot C-index
    for i, prior_name in enumerate(existing_priors):
        prior_data = results_df[results_df['prior_name'] == prior_name].sort_values('train_fraction')
        ax1.plot(prior_data['train_fraction'], prior_data['c_index'], 
                'o-', color=colors[i], label=prior_name, linewidth=2, markersize=6)
    
    for i, prior_name in enumerate(llm_priors):
        prior_data = results_df[results_df['prior_name'] == prior_name].sort_values('train_fraction')
        ax1.plot(prior_data['train_fraction'], prior_data['c_index'], 
                's--', color=colors[len(existing_priors) + i], label=prior_name, 
                linewidth=2, markersize=8, alpha=0.8)
    
    ax1.set_xlabel('Training Data Fraction', fontsize=12)
    ax1.set_ylabel('C-index', fontsize=12)
    ax1.set_title('C-index vs Training Data Size', fontsize=14)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.4, 0.9)
    
    # Plot IBS
    for i, prior_name in enumerate(existing_priors):
        prior_data = results_df[results_df['prior_name'] == prior_name].sort_values('train_fraction')
        ax2.plot(prior_data['train_fraction'], prior_data['ibs'], 
                'o-', color=colors[i], label=prior_name, linewidth=2, markersize=6)
    
    for i, prior_name in enumerate(llm_priors):
        prior_data = results_df[results_df['prior_name'] == prior_name].sort_values('train_fraction')
        ax2.plot(prior_data['train_fraction'], prior_data['ibs'], 
                's--', color=colors[len(existing_priors) + i], label=prior_name, 
                linewidth=2, markersize=8, alpha=0.8)
    
    ax2.set_xlabel('Training Data Fraction', fontsize=12)
    ax2.set_ylabel('IBS (Integrated Brier Score)', fontsize=12)
    ax2.set_title('IBS vs Training Data Size', fontsize=14)
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_path = output_dir / "holdout_learning_curves.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Learning curves saved to {output_path}")
    
    plt.show()


def create_performance_heatmap(results_df: pd.DataFrame,
                              output_dir: Optional[Path] = None) -> None:
    """
    Create heatmaps showing performance across priors and training fractions.
    
    Args:
        results_df: Results DataFrame
        output_dir: Directory to save plots (default: FIGURES_DIR)
    """
    if output_dir is None:
        output_dir = FIGURES_DIR
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Creating performance heatmaps")
    
    # Create pivot tables
    c_index_pivot = results_df.pivot(
        index='prior_name', columns='train_fraction', values='c_index'
    )
    
    ibs_pivot = results_df.pivot(
        index='prior_name', columns='train_fraction', values='ibs'
    )
    
    # Set up the plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # C-index heatmap
    sns.heatmap(
        c_index_pivot, annot=True, fmt='.3f', cmap='RdYlBu_r',
        ax=ax1, cbar_kws={'label': 'C-index'},
        vmin=0.5, vmax=0.8
    )
    ax1.set_title('C-index Across Training Fractions', fontsize=14)
    ax1.set_xlabel('Training Data Fraction', fontsize=12)
    ax1.set_ylabel('Prior', fontsize=12)
    
    # IBS heatmap
    sns.heatmap(
        ibs_pivot, annot=True, fmt='.3f', cmap='RdYlBu',
        ax=ax2, cbar_kws={'label': 'IBS'}
    )
    ax2.set_title('IBS Across Training Fractions', fontsize=14)
    ax2.set_xlabel('Training Data Fraction', fontsize=12)
    ax2.set_ylabel('Prior', fontsize=12)
    
    plt.tight_layout()
    
    # Save plot
    output_path = output_dir / "holdout_performance_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Performance heatmap saved to {output_path}")
    
    plt.show()


def analyze_small_n_robustness(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze robustness of priors in small-n settings.
    
    Args:
        results_df: Results DataFrame
        
    Returns:
        Robustness analysis DataFrame
    """
    logger.info("Analyzing small-n robustness")
    
    robustness_results = []
    
    for prior_name in results_df['prior_name'].unique():
        prior_data = results_df[results_df['prior_name'] == prior_name].copy()
        prior_data = prior_data.sort_values('train_fraction')
        
        # Calculate metrics
        c_index_values = prior_data['c_index'].dropna()
        ibs_values = prior_data['ibs'].dropna()
        
        if len(c_index_values) == 0 or len(ibs_values) == 0:
            continue
        
        # Performance at different fractions
        perf_100 = prior_data[prior_data['train_fraction'] == 1.0]
        perf_20 = prior_data[prior_data['train_fraction'] == 0.2]
        
        c_index_100 = perf_100['c_index'].iloc[0] if len(perf_100) > 0 else np.nan
        c_index_20 = perf_20['c_index'].iloc[0] if len(perf_20) > 0 else np.nan
        c_index_degradation = c_index_100 - c_index_20 if not (np.isnan(c_index_100) or np.isnan(c_index_20)) else np.nan
        
        ibs_100 = perf_100['ibs'].iloc[0] if len(perf_100) > 0 else np.nan
        ibs_20 = perf_20['ibs'].iloc[0] if len(perf_20) > 0 else np.nan
        ibs_degradation = ibs_20 - ibs_100 if not (np.isnan(ibs_100) or np.isnan(ibs_20)) else np.nan  # Higher is worse
        
        # Average performance and stability
        avg_c_index = c_index_values.mean()
        avg_ibs = ibs_values.mean()
        c_index_stability = c_index_values.std()
        ibs_stability = ibs_values.std()
        
        # Prior type
        prior_type = 'LLM' if prior_name.startswith('llm_') else 'Existing'
        
        # Convergence rate
        convergence_rate = prior_data['convergence_ok'].mean()
        avg_mcmc_time = prior_data['mcmc_time'].mean()
        
        robustness_results.append({
            'prior_name': prior_name,
            'prior_type': prior_type,
            'avg_c_index': avg_c_index,
            'avg_ibs': avg_ibs,
            'c_index_stability': c_index_stability,
            'ibs_stability': ibs_stability,
            'c_index_100': c_index_100,
            'c_index_20': c_index_20,
            'c_index_degradation': c_index_degradation,
            'ibs_100': ibs_100,
            'ibs_20': ibs_20,
            'ibs_degradation': ibs_degradation,
            'convergence_rate': convergence_rate,
            'avg_mcmc_time': avg_mcmc_time
        })
    
    robustness_df = pd.DataFrame(robustness_results)
    
    # Calculate robustness score
    # Higher is better: high avg C-index, low avg IBS, low degradation, high stability
    robustness_df['robustness_score'] = (
        robustness_df['avg_c_index'] * 2 +  # Higher C-index is better
        (1 / (robustness_df['avg_ibs'] + 0.01)) * 0.5 +  # Lower IBS is better
        (1 / (robustness_df['c_index_stability'] + 0.01)) * 0.5 +  # More stable is better
        (1 / (robustness_df['ibs_stability'] + 0.01)) * 0.5 +  # More stable is better
        (1 / (abs(robustness_df['c_index_degradation']) + 0.01)) * 0.5  # Less degradation is better
    )
    
    # Rank by robustness score
    robustness_df = robustness_df.sort_values('robustness_score', ascending=False)
    
    logger.info(f"Robustness analysis completed for {len(robustness_df)} priors")
    
    return robustness_df


def create_summary_table(results_df: pd.DataFrame, 
                        output_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Create formatted summary table for reporting.
    
    Args:
        results_df: Results DataFrame
        output_dir: Directory to save table (default: TABLES_DIR)
        
    Returns:
        Summary table DataFrame
    """
    if output_dir is None:
        output_dir = TABLES_DIR
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Creating summary table")
    
    # Create pivot table for better formatting
    summary_data = []
    
    for prior_name in results_df['prior_name'].unique():
        prior_data = results_df[results_df['prior_name'] == prior_name]
        prior_type = 'LLM' if prior_name.startswith('llm_') else 'Existing'
        
        row = {'Prior': prior_name, 'Type': prior_type}
        
        for fraction in sorted(results_df['train_fraction'].unique()):
            frac_data = prior_data[prior_data['train_fraction'] == fraction]
            
            if len(frac_data) > 0:
                c_index = frac_data['c_index'].iloc[0]
                ibs = frac_data['ibs'].iloc[0]
                convergence = frac_data['convergence_ok'].iloc[0]
                
                # Format performance metrics
                c_index_str = f"{c_index:.3f}" if not np.isnan(c_index) else "N/A"
                ibs_str = f"{ibs:.3f}" if not np.isnan(ibs) else "N/A"
                conv_str = "✓" if convergence else "✗"
                
                row[f'C-index_{int(fraction*100)}%'] = c_index_str
                row[f'IBS_{int(fraction*100)}%'] = ibs_str
                row[f'Conv_{int(fraction*100)}%'] = conv_str
            else:
                row[f'C-index_{int(fraction*100)}%'] = "N/A"
                row[f'IBS_{int(fraction*100)}%'] = "N/A"
                row[f'Conv_{int(fraction*100)}%'] = "N/A"
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    # Save summary table
    output_path = output_dir / "holdout_sensitivity_summary.csv"
    summary_df.to_csv(output_path, index=False)
    logger.info(f"Summary table saved to {output_path}")
    
    return summary_df


def generate_holdout_sensitivity_report(results_path: Optional[Path] = None,
                                       output_dir: Optional[Path] = None) -> None:
    """
    Generate complete holdout sensitivity analysis report.
    
    Args:
        results_path: Path to results CSV (default: auto-detect)
        output_dir: Directory for outputs (default: TABLES_DIR)
    """
    if results_path is None:
        results_path = TABLES_DIR / "holdout_sensitivity_results.csv"
    
    if output_dir is None:
        output_dir = TABLES_DIR
    
    logger.info("Generating holdout sensitivity analysis report")
    
    # Load results
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    results_df = pd.read_csv(results_path)
    logger.info(f"Loaded {len(results_df)} results")
    
    # Create visualizations
    create_learning_curves_plot(results_df)
    create_performance_heatmap(results_df)
    
    # Create summary table
    summary_df = create_summary_table(results_df, output_dir)
    
    # Analyze robustness
    robustness_df = analyze_small_n_robustness(results_df)
    
    # Save robustness analysis
    robustness_path = output_dir / "holdout_robustness_analysis.csv"
    robustness_df.to_csv(robustness_path, index=False)
    logger.info(f"Robustness analysis saved to {robustness_path}")
    
    # Print report
    print("\n" + "="*70)
    print("HOLDOUT-BASED SMALL-N SENSITIVITY ANALYSIS REPORT")
    print("="*70)
    
    print(f"\nDataset Information:")
    test_size = results_df['n_test_total'].iloc[0] if len(results_df) > 0 else 0
    train_size_100 = results_df[results_df['train_fraction'] == 1.0]['n_train_total'].iloc[0] if len(results_df[results_df['train_fraction'] == 1.0]) > 0 else 0
    total_size = test_size + train_size_100
    
    print(f"  Total observations: {total_size}")
    print(f"  Training set (100%): {train_size_100} obs")
    print(f"  Test set: {test_size} obs ({test_size/total_size:.1%})")
    
    print(f"\nExperiment Design:")
    train_fractions = sorted(results_df['train_fraction'].unique())
    print(f"  Training fractions: {[f'{f:.0%}' for f in train_fractions]}")
    print(f"  Priors evaluated: {len(results_df['prior_name'].unique())}")
    print(f"  Total evaluations: {len(results_df)}")
    
    print(f"\nConvergence Summary:")
    convergence_rate = results_df['convergence_ok'].mean()
    print(f"  Overall convergence rate: {convergence_rate:.1%}")
    
    failed_runs = results_df[results_df['c_index'].isna()]
    if len(failed_runs) > 0:
        print(f"  Failed evaluations: {len(failed_runs)}")
    
    print(f"\nTOP 5 MOST ROBUST PRIORS:")
    for i, (_, row) in enumerate(robustness_df.head(5).iterrows()):
        print(f"{i+1}. {row['prior_name']} ({row['prior_type']})")
        print(f"   Avg C-index: {row['avg_c_index']:.3f}")
        print(f"   Avg IBS: {row['avg_ibs']:.3f}")
        print(f"   C-index degradation (100% → 20%): {row['c_index_degradation']:.3f}")
        print(f"   Robustness score: {row['robustness_score']:.2f}")
        print()
    
    print(f"Performance at Small-n (20% training data):")
    small_n_results = results_df[results_df['train_fraction'] == 0.2].sort_values('c_index', ascending=False)
    for _, row in small_n_results.head(3).iterrows():
        print(f"  {row['prior_name']}: C-index={row['c_index']:.3f}, IBS={row['ibs']:.3f}")
    
    print(f"\nDetailed results available in:")
    print(f"  Summary: {output_dir / 'holdout_sensitivity_summary.csv'}")
    print(f"  Robustness: {robustness_path}")
    print(f"  Figures: {FIGURES_DIR}")
    print("="*70)


if __name__ == "__main__":
    # Generate report from existing results
    generate_holdout_sensitivity_report()
