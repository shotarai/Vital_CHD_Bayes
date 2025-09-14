"""
Predictive-only script for VITAL-CHD Bayesian analysis.

This script runs only the predictive evaluation part:
1. Load previously saved inference results
2. Compute predictive performance metrics (PSIS-LOO, WAIC)
3. Generate comparison tables and figures
"""

import sys
import logging
from pathlib import Path
from typing import Optional, Dict
import time
import pandas as pd

# Add src to path to enable imports
sys.path.insert(0, str(Path(__file__).parent))

from config import SEED, TABLES_DIR, FIGURES_DIR
from inference import InferenceResults, load_inference_results
from predictive import compute_predictive_performance, save_predictive_results
from reporting import generate_predictive_plots

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(TABLES_DIR / 'predictive_experiment.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class PredictiveOnlyRunner:
    """Predictive-only experiment runner class."""
    
    def __init__(self):
        """Initialize predictive runner."""
        self.results = {}
        
        logger.info("=" * 60)
        logger.info("VITAL-CHD BAYESIAN ANALYSIS (PREDICTIVE ONLY)")
        logger.info("=" * 60)
        
    def step_1_load_inference_results(self):
        """Step 1: Load previously saved inference results."""
        logger.info("STEP 1: Loading inference results")
        logger.info("-" * 40)
        
        try:
            # Check if inference results exist
            inference_summary_path = TABLES_DIR / "inference_summary.csv"
            if not inference_summary_path.exists():
                raise FileNotFoundError(
                    f"Inference results not found at {inference_summary_path}. "
                    "Please run inference first using: rye run python -m src.run_inference_only"
                )
            
            # Load inference summary
            inference_df = load_inference_results()
            
            logger.info(f"Inference results loaded successfully:")
            logger.info(f"  Models found: {len(inference_df)}")
            
            # Try to load full traces
            from inference import load_inference_traces
            traces = load_inference_traces()
            
            if traces:
                logger.info(f"  Full traces available: {len(traces)}")
                
                # Reconstruct InferenceResults objects from traces
                inference_results = self._reconstruct_from_traces(traces, inference_df)
                self.results['inference'] = inference_results
                self.results['has_traces'] = True
                
                logger.info(f"Successfully loaded {len(inference_results)} inference results with traces")
            else:
                logger.warning("No saved traces found. Limited predictive evaluation possible.")
                self.results['has_traces'] = False
                
        except Exception as e:
            logger.error(f"Failed to load inference results: {e}")
            raise
            
    def _reconstruct_from_traces(self, traces, inference_df) -> Dict[str, InferenceResults]:
        """
        Reconstruct InferenceResults objects from saved traces.
        
        Args:
            traces: Dictionary of loaded traces
            inference_df: DataFrame with inference summaries
            
        Returns:
            Dict[str, InferenceResults]: Reconstructed inference results
        """
        logger.info("Reconstructing inference results from saved traces")
        
        # We need to also load the model and prior information
        # For simplicity, we'll create minimal objects that work with predictive evaluation
        from priors import get_all_priors
        from model_weibull_ph import create_weibull_ph_model
        from data_io import get_processed_data
        
        # Get data and priors
        X, time, event = get_processed_data()
        all_priors = get_all_priors()
        
        inference_results = {}
        
        for prior_name, trace in traces.items():
            if prior_name in all_priors:
                try:
                    # Recreate the model (needed for log-likelihood computation)
                    model = create_weibull_ph_model(
                        X=X,
                        time=time,
                        event=event,
                        log_hr_prior=all_priors[prior_name],
                        model_name=f"weibull_ph_{prior_name}"
                    )
                    
                    # Create InferenceResults object
                    result = InferenceResults(prior_name, trace, model)
                    inference_results[prior_name] = result
                    
                    logger.info(f"Reconstructed inference result for {prior_name}")
                    
                except Exception as e:
                    logger.warning(f"Could not reconstruct {prior_name}: {e}")
                    continue
            else:
                logger.warning(f"Prior {prior_name} not found in current prior set")
        
        return inference_results
            
    def step_2_compute_predictive_performance(self):
        """Step 2: Compute predictive performance metrics."""
        logger.info("STEP 2: Computing predictive performance")
        logger.info("-" * 40)
        
        # Check if we have inference results with traces
        if not self.results.get('has_traces', False):
            logger.warning("No saved traces found. Cannot compute PSIS-LOO and WAIC.")
            logger.warning("These metrics require the full posterior samples.")
            logger.warning("Consider running the full experiment or ensure traces are saved.")
            return
        
        if not self.results.get('inference'):
            logger.error("No inference results available for predictive evaluation")
            raise RuntimeError("Inference results not available for predictive evaluation")
        
        try:
            predictive_results = compute_predictive_performance(
                self.results['inference']
            )
            
            self.results['predictive'] = predictive_results
            
            # Save predictive results
            save_predictive_results(predictive_results)
            
            logger.info(f"Predictive evaluation completed:")
            logger.info(f"  Models evaluated: {len(predictive_results)}")
            
            # Log best performing model
            if predictive_results:
                import numpy as np
                import pandas as pd
                
                best_model = max(
                    predictive_results.items(),
                    key=lambda x: x[1].loo_lpd if x[1].loo_lpd and not pd.isna(x[1].loo_lpd) else -np.inf
                )
                logger.info(f"  Best LOO performance: {best_model[0]} "
                           f"(elpd = {best_model[1].loo_lpd:.2f})")
            
        except Exception as e:
            logger.error(f"Failed to compute predictive performance: {e}")
            raise
            
    def step_3_generate_predictive_reports(self):
        """Step 3: Generate predictive performance reports and figures."""
        logger.info("STEP 3: Generating predictive reports and figures")
        logger.info("-" * 40)
        
        try:
            from predictive import load_predictive_results
            
            # Load results (in case they were saved/loaded)
            predictive_summary = load_predictive_results()
            
            # Generate predictive plots
            generate_predictive_plots(predictive_summary)
            
            # Print predictive summary
            self._print_predictive_summary(predictive_summary)
            
            logger.info("Predictive reports and figures generated:")
            logger.info(f"  Tables saved to: {TABLES_DIR}")
            logger.info(f"  Figures saved to: {FIGURES_DIR}")
            
        except Exception as e:
            logger.error(f"Failed to generate predictive reports: {e}")
            raise
            
    def _print_predictive_summary(self, predictive_summary):
        """Print a summary of predictive results."""
        try:
            print("\n" + "="*60)
            print("PREDICTIVE EVALUATION SUMMARY")
            print("="*60)
            
            print(f"Models evaluated: {len(predictive_summary)}")
            
            if len(predictive_summary) > 0:
                # Sort by LOO elpd (descending - higher is better)
                predictive_summary_sorted = predictive_summary.sort_values('loo_elpd', ascending=False)
                
                print("\nMODEL RANKING BY PREDICTIVE PERFORMANCE:")
                print("-" * 50)
                print("Rank | Model               | LOO elpd    | ΔLOO    | WAIC elpd")
                print("-" * 65)
                
                for i, (_, row) in enumerate(predictive_summary_sorted.iterrows(), 1):
                    model = f"{row['prior_name'][:18]:18s}"
                    loo_elpd = f"{row['loo_elpd']:8.1f} ± {row['loo_se']:4.1f}"
                    loo_diff = f"{row['loo_diff']:7.1f}"
                    waic_elpd = f"{row['waic_elpd']:8.1f}" if 'waic_elpd' in row and not pd.isna(row['waic_elpd']) else "   N/A"
                    
                    print(f"{i:4d} | {model} | {loo_elpd} | {loo_diff} | {waic_elpd}")
                
                # Best model
                best_model = predictive_summary_sorted.iloc[0]
                print(f"\nBEST PREDICTIVE PERFORMANCE:")
                print(f"  Model: {best_model['prior_name']}")
                print(f"  LOO elpd: {best_model['loo_elpd']:.1f} ± {best_model['loo_se']:.1f}")
                
                # LLM vs existing comparison (if both exist)
                existing = predictive_summary[predictive_summary['prior_type'] == 'existing']
                llm = predictive_summary[predictive_summary['prior_type'] == 'llm']
                
                if len(llm) > 0 and len(existing) > 0:
                    print(f"\nLLM VS EXISTING COMPARISON:")
                    print(f"  LLM models: {len(llm)}")
                    print(f"  Existing models: {len(existing)}")
                    print(f"  LLM average LOO: {llm['loo_elpd'].mean():.1f}")
                    print(f"  Existing average LOO: {existing['loo_elpd'].mean():.1f}")
                    
                    # Count LLMs in top half
                    n_total = len(predictive_summary)
                    llm_in_top_half = len(llm[llm['loo_rank'] <= n_total/2])
                    print(f"  LLM in top half: {llm_in_top_half}/{len(llm)}")
                    
                    if llm['loo_elpd'].mean() > existing['loo_elpd'].mean():
                        print("  → LLM priors show better average predictive performance")
                    else:
                        print("  → Existing priors show better average predictive performance")
            
            print("="*60)
            
        except Exception as e:
            logger.warning(f"Could not generate predictive summary: {e}")
            
    def run_predictive_experiment(self):
        """Run the predictive-only experiment pipeline."""
        start_time = time.time()
        
        try:
            self.step_1_load_inference_results()
            
            # Check if we can do full predictive evaluation
            if self.results.get('has_traces', False):
                logger.info("Full traces available. Running complete predictive evaluation.")
                self.step_2_compute_predictive_performance()
                self.step_3_generate_predictive_reports()
            else:
                logger.warning("=" * 60)
                logger.warning("LIMITED PREDICTIVE EVALUATION")
                logger.warning("=" * 60)
                logger.warning("Saved traces not found. Cannot compute PSIS-LOO and WAIC.")
                logger.warning("Generating basic reports from inference summaries only.")
                logger.warning("=" * 60)
                
                # Generate basic report from inference summaries
                self._generate_basic_report()
            
            total_time = time.time() - start_time
            
            logger.info("=" * 60)
            logger.info("PREDICTIVE EXPERIMENT COMPLETED")
            logger.info("=" * 60)
            logger.info(f"Total time: {total_time:.1f} seconds")
            
        except Exception as e:
            logger.error(f"Predictive experiment failed: {e}")
            # Try to generate basic report anyway
            try:
                self._generate_basic_report()
            except:
                pass
            raise
            
    def _generate_basic_report(self):
        """Generate a basic report using only inference summaries."""
        try:
            inference_df = load_inference_results()
            
            print("\n" + "="*60)
            print("BASIC INFERENCE SUMMARY (NO PREDICTIVE METRICS)")
            print("="*60)
            
            print(f"Models completed: {len(inference_df)}")
            
            if len(inference_df) > 0:
                # Sort by HR median
                inference_df_sorted = inference_df.sort_values('hr_median')
                
                print("\nMODEL RANKING BY HR (Most protective first):")
                print("-" * 40)
                for i, (_, row) in enumerate(inference_df_sorted.iterrows(), 1):
                    print(f"{i:2d}. {row['prior_name']:20s} "
                          f"HR={row['hr_median']:.3f} "
                          f"(95% CrI: {row['hr_q025']:.3f}-{row['hr_q975']:.3f}) "
                          f"P(HR<1)={row['prob_hr_less_than_1']:.3f}")
                
                # LLM vs existing comparison
                existing = inference_df[~inference_df['prior_name'].str.startswith('llm_')]
                llm = inference_df[inference_df['prior_name'].str.startswith('llm_')]
                
                if len(llm) > 0 and len(existing) > 0:
                    print(f"\nLLM VS EXISTING COMPARISON:")
                    print(f"  LLM models: {len(llm)}")
                    print(f"  Existing models: {len(existing)}")
                    print(f"  LLM average HR: {llm['hr_median'].mean():.3f}")
                    print(f"  Existing average HR: {existing['hr_median'].mean():.3f}")
                    print(f"  LLM average P(HR<1): {llm['prob_hr_less_than_1'].mean():.3f}")
                    print(f"  Existing average P(HR<1): {existing['prob_hr_less_than_1'].mean():.3f}")
                
                # Save basic comparison
                basic_comparison_path = TABLES_DIR / "basic_model_comparison.csv"
                inference_df_sorted.to_csv(basic_comparison_path, index=False)
                print(f"\nBasic comparison saved to: {basic_comparison_path}")
            
            print("="*60)
            print("NOTE: For full predictive evaluation (PSIS-LOO, WAIC):")
            print("  Implement trace saving/loading or run full experiment")
            print("="*60)
            
        except Exception as e:
            logger.warning(f"Could not generate basic report: {e}")


def main():
    """Main function to run the predictive-only experiment."""
    
    # Ensure directories exist
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create and run experiment
    experiment = PredictiveOnlyRunner()
    experiment.run_predictive_experiment()


if __name__ == "__main__":
    main()
