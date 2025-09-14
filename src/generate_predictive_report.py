"""
Simple predictive report generator that uses existing predictive results.

This script reads the saved predictive_summary.csv and generates reports and plots.
"""

import sys
import pandas as pd
from pathlib import Path

# Add src to path to enable imports
sys.path.insert(0, str(Path(__file__).parent))

from config import TABLES_DIR, FIGURES_DIR
from reporting import generate_predictive_plots


def main():
    """Generate predictive evaluation report from saved results."""
    
    print("=" * 60)
    print("VITAL-CHD PREDICTIVE EVALUATION REPORT")
    print("=" * 60)
    
    # Load existing predictive results
    predictive_path = TABLES_DIR / "predictive_summary.csv"
    if not predictive_path.exists():
        print(f"ERROR: Predictive results not found at {predictive_path}")
        print("Please run the full experiment first.")
        return
    
    # Load results
    df = pd.read_csv(predictive_path)
    
    print(f"Loaded predictive results for {len(df)} models")
    print()
    
    # Generate basic summary
    print("PREDICTIVE PERFORMANCE RANKING:")
    print("-" * 50)
    print("Rank | Model               | LOO elpd    | ΔLOO    | WAIC elpd")
    print("-" * 65)
    
    for i, (_, row) in enumerate(df.iterrows(), 1):
        model = f"{row['prior_name'][:18]:18s}"
        loo_elpd = f"{row['loo_elpd']:8.1f} ± {row['loo_se']:4.1f}"
        loo_diff = f"{row['loo_diff']:7.1f}"
        waic_elpd = f"{row['waic_elpd']:8.1f}" if 'waic_elpd' in row and not pd.isna(row['waic_elpd']) else "   N/A"
        
        print(f"{i:4d} | {model} | {loo_elpd} | {loo_diff} | {waic_elpd}")
    
    # Best model
    best_model = df.iloc[0]  # Already sorted by rank
    print(f"\nBEST PREDICTIVE PERFORMANCE:")
    print(f"  Model: {best_model['prior_name']}")
    print(f"  LOO elpd: {best_model['loo_elpd']:.1f} ± {best_model['loo_se']:.1f}")
    
    # LLM vs existing comparison
    existing = df[df['prior_type'] == 'existing']
    llm = df[df['prior_type'] == 'llm']
    
    if len(llm) > 0 and len(existing) > 0:
        print(f"\nLLM VS EXISTING COMPARISON:")
        print(f"  LLM models: {len(llm)}")
        print(f"  Existing models: {len(existing)}")
        print(f"  LLM average LOO: {llm['loo_elpd'].mean():.1f}")
        print(f"  Existing average LOO: {existing['loo_elpd'].mean():.1f}")
        
        # Count LLMs in top half
        n_total = len(df)
        llm_in_top_half = len(llm[llm['loo_rank'] <= n_total/2])
        print(f"  LLM in top half: {llm_in_top_half}/{len(llm)}")
        
        if llm['loo_elpd'].mean() > existing['loo_elpd'].mean():
            print("  → LLM priors show better average predictive performance")
        else:
            print("  → Existing priors show better average predictive performance")
    
    # Model differences
    print(f"\nMODEL DIFFERENCES (vs Reference):")
    significant_improvements = df[df['loo_diff'] > 2 * df['loo_diff_se']]
    if len(significant_improvements) > 0:
        print("  Models with significant improvement (ΔLOO > 2SE):")
        for _, row in significant_improvements.iterrows():
            print(f"    {row['prior_name']}: ΔLOO = {row['loo_diff']:.1f} ± {row['loo_diff_se']:.1f}")
    else:
        print("  No models show significant improvement over reference")
    
    print()
    print("=" * 60)
    print("INFERENCE + PREDICTIVE COMBINED SUMMARY:")
    print("=" * 60)
    
    # Load inference results for combined summary
    inference_path = TABLES_DIR / "inference_summary.csv"
    if inference_path.exists():
        inf_df = pd.read_csv(inference_path)
        
        # Merge results
        combined = inf_df.merge(df[['prior_name', 'loo_elpd', 'loo_rank', 'loo_diff']], 
                               on='prior_name')
        combined = combined.sort_values('loo_rank')
        
        print("Combined Ranking (by predictive performance):")
        print("-" * 80)
        print("Rank | Model               | HR (95% CrI)        | P(HR<1) | LOO elpd | ΔLOO")
        print("-" * 85)
        
        for _, row in combined.iterrows():
            rank = f"{row['loo_rank']:2.0f}"
            name = f"{row['prior_name'][:18]:18s}"
            hr_ci = f"{row['hr_median']:.3f} ({row['hr_q025']:.3f}-{row['hr_q975']:.3f})"
            prob = f"{row['prob_hr_less_than_1']:.3f}"
            loo = f"{row['loo_elpd']:8.1f}"
            delta_loo = f"{row['loo_diff']:6.1f}"
            
            line = f"{rank:>4} | {name} | {hr_ci:>18} | {prob:>7} | {loo} | {delta_loo}"
            print(line)
    
    print()
    print("=" * 60)
    print("KEY FINDINGS:")
    print("=" * 60)
    
    # Key insights
    if len(llm) > 0:
        llm_avg_rank = llm['loo_rank'].mean()
        existing_avg_rank = existing['loo_rank'].mean()
        
        print(f"1. LLM priors average rank: {llm_avg_rank:.1f}")
        print(f"   Existing priors average rank: {existing_avg_rank:.1f}")
        
        if llm_avg_rank < existing_avg_rank:
            print("   → LLM priors perform better in predictive ranking")
        else:
            print("   → Existing priors perform better in predictive ranking")
    
    # Best overall model
    if 'combined' in locals():
        best_combined = combined.iloc[0]
        print(f"\n2. Best overall model: {best_combined['prior_name']}")
        print(f"   - Most protective: HR = {best_combined['hr_median']:.3f}")
        print(f"   - Best predictive: LOO rank = {best_combined['loo_rank']:.0f}")
        print(f"   - High confidence: P(HR<1) = {best_combined['prob_hr_less_than_1']:.3f}")
    
    # Generate plots if possible
    try:
        generate_predictive_plots(df)
        print(f"\n3. Plots generated in: {FIGURES_DIR}")
    except Exception as e:
        print(f"\n3. Could not generate plots: {e}")
    
    print("\n" + "=" * 60)
    print("Report completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
