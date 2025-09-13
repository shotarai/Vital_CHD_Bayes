"""
Main experiment orchestration for VITAL-CHD Bayesian re-analysis.

This script runs the complete experiment pipeline:
1. Load and preprocess data
2. Get all prior distributions (existing + LLM)
3. Run Bayesian inference for each prior
4. Compute predictive performance metrics
5. Generate summary tables and figures
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Optional
import time

# Add src to path to enable imports
sys.path.insert(0, str(Path(__file__).parent))

from config import SEED, TABLES_DIR, FIGURES_DIR
from io import get_processed_data
from priors import get_all_priors, save_priors_summary
from inference import run_inference_all_priors, save_inference_results
from predictive import compute_predictive_performance, save_predictive_results
from reporting import generate_all_plots, create_summary_report

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(TABLES_DIR / 'experiment.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Main experiment runner class."""
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize experiment runner.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed or SEED
        self.results = {}
        
        logger.info("=" * 60)
        logger.info("VITAL-CHD BAYESIAN RE-ANALYSIS EXPERIMENT")
        logger.info("=" * 60)
        logger.info(f"Random seed: {self.random_seed}")
        
    def step_1_load_data(self):
        """Step 1: Load and preprocess VITAL data."""
        logger.info("STEP 1: Loading and preprocessing data")
        logger.info("-" * 40)
        
        try:
            X, time, event = get_processed_data()
            
            self.results['data'] = {
                'X': X,
                'time': time,
                'event': event,
                'n_obs': len(X),
                'n_covariates': X.shape[1],
                'n_events': int(event.sum()),
                'event_rate': float(event.mean())
            }
            
            logger.info(f"Data loaded successfully:")
            logger.info(f"  Observations: {self.results['data']['n_obs']}")
            logger.info(f"  Covariates: {self.results['data']['n_covariates']}")
            logger.info(f"  Events: {self.results['data']['n_events']} "
                       f"({self.results['data']['event_rate']:.1%})")
            logger.info(f"  Mean follow-up: {time.mean():.2f} years")
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            raise
            
    def step_2_get_priors(self):
        """Step 2: Get all prior distributions."""
        logger.info("STEP 2: Getting prior distributions")
        logger.info("-" * 40)
        
        try:
            priors = get_all_priors()
            self.results['priors'] = priors
            
            # Save prior summary
            save_priors_summary(priors, TABLES_DIR / "priors_summary.csv")
            
            existing_count = sum(1 for p in priors.values() if p.source == "existing")
            llm_count = sum(1 for p in priors.values() if p.source == "llm")
            
            logger.info(f"Prior distributions obtained:")
            logger.info(f"  Existing priors: {existing_count}")
            logger.info(f"  LLM priors: {llm_count}")
            logger.info(f"  Total: {len(priors)}")
            
            # Log prior details
            for name, prior in priors.items():
                logger.info(f"  {name}: N({prior.mu:.4f}, {prior.sigma:.4f}^2) [{prior.source}]")
                
        except Exception as e:
            logger.error(f"Failed to get priors: {e}")
            raise
            
    def step_3_run_inference(self):
        """Step 3: Run Bayesian inference for all priors."""
        logger.info("STEP 3: Running Bayesian inference")
        logger.info("-" * 40)
        
        try:
            start_time = time.time()
            
            inference_results = run_inference_all_priors(
                X=self.results['data']['X'],
                time=self.results['data']['time'],
                event=self.results['data']['event'],
                priors=self.results['priors'],
                random_seed=self.random_seed
            )
            
            self.results['inference'] = inference_results
            
            # Save inference results
            save_inference_results(inference_results)
            
            elapsed_time = time.time() - start_time
            
            logger.info(f"Inference completed:")
            logger.info(f"  Successful models: {len(inference_results)}")
            logger.info(f"  Time elapsed: {elapsed_time/60:.1f} minutes")
            
            # Log convergence summary
            convergence_issues = []
            for name, result in inference_results.items():
                if result.hr_summary['max_rhat'] > 1.01:
                    convergence_issues.append(name)
            
            if convergence_issues:
                logger.warning(f"Convergence issues detected in: {convergence_issues}")
            else:
                logger.info("All models converged successfully (R-hat ≤ 1.01)")
                
        except Exception as e:
            logger.error(f"Failed to run inference: {e}")
            raise
            
    def step_4_compute_predictive_performance(self):
        """Step 4: Compute predictive performance metrics."""
        logger.info("STEP 4: Computing predictive performance")
        logger.info("-" * 40)
        
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
            
    def step_5_generate_reports(self):
        """Step 5: Generate summary tables and figures."""
        logger.info("STEP 5: Generating reports and figures")
        logger.info("-" * 40)
        
        try:
            from inference import load_inference_results
            from predictive import load_predictive_results
            
            # Load results (in case they were saved/loaded)
            inference_summary = load_inference_results()
            predictive_summary = load_predictive_results()
            
            # Generate all plots
            generate_all_plots(
                self.results['inference'],
                predictive_summary
            )
            
            # Create comprehensive summary report
            create_summary_report(
                inference_summary,
                predictive_summary
            )
            
            logger.info("Reports and figures generated:")
            logger.info(f"  Tables saved to: {TABLES_DIR}")
            logger.info(f"  Figures saved to: {FIGURES_DIR}")
            
        except Exception as e:
            logger.error(f"Failed to generate reports: {e}")
            raise
            
    def run_complete_experiment(self):
        """Run the complete experiment pipeline."""
        start_time = time.time()
        
        try:
            self.step_1_load_data()
            self.step_2_get_priors()
            self.step_3_run_inference()
            self.step_4_compute_predictive_performance()
            self.step_5_generate_reports()
            
            total_time = time.time() - start_time
            
            logger.info("=" * 60)
            logger.info("EXPERIMENT COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            logger.info(f"Total time: {total_time/60:.1f} minutes")
            logger.info(f"Results saved to: {TABLES_DIR.parent}")
            
            # Print final summary
            self._print_final_summary()
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            raise
            
    def _print_final_summary(self):
        """Print a final summary of key results."""
        try:
            from inference import load_inference_results
            from predictive import load_predictive_results
            
            inference_df = load_inference_results()
            predictive_df = load_predictive_results()
            
            print("\n" + "="*60)
            print("EXPERIMENT SUMMARY")
            print("="*60)
            
            # Data summary
            print(f"Dataset: {self.results['data']['n_obs']} observations, "
                  f"{self.results['data']['n_events']} events "
                  f"({self.results['data']['event_rate']:.1%})")
            
            # Model summary
            print(f"Models compared: {len(inference_df)}")
            existing = inference_df[~inference_df['prior_name'].str.startswith('llm_')]
            llm = inference_df[inference_df['prior_name'].str.startswith('llm_')]
            print(f"  Existing priors: {len(existing)}")
            print(f"  LLM priors: {len(llm)}")
            
            # Best results
            print("\nKEY FINDINGS:")
            
            # Most protective effect
            best_hr_idx = inference_df['hr_median'].idxmin()
            best_hr = inference_df.loc[best_hr_idx]
            print(f"Most protective effect: {best_hr['prior_name']}")
            print(f"  HR = {best_hr['hr_median']:.3f} "
                  f"(95% CrI: {best_hr['hr_q025']:.3f}-{best_hr['hr_q975']:.3f})")
            print(f"  P(HR < 1) = {best_hr['prob_hr_less_than_1']:.3f}")
            
            # Best predictive performance
            best_loo_idx = predictive_df['loo_elpd'].idxmax()
            best_loo = predictive_df.loc[best_loo_idx]
            print(f"Best predictive performance: {best_loo['prior_name']}")
            print(f"  LOO elpd = {best_loo['loo_elpd']:.1f} ± {best_loo['loo_se']:.1f}")
            
            # LLM performance
            if len(llm) > 0:
                llm_predictive = predictive_df[predictive_df['prior_name'].str.startswith('llm_')]
                existing_predictive = predictive_df[~predictive_df['prior_name'].str.startswith('llm_')]
                
                llm_mean_loo = llm_predictive['loo_elpd'].mean()
                existing_mean_loo = existing_predictive['loo_elpd'].mean()
                
                print(f"\nLLM vs Existing Performance:")
                print(f"  LLM average LOO: {llm_mean_loo:.1f}")
                print(f"  Existing average LOO: {existing_mean_loo:.1f}")
                
                if llm_mean_loo > existing_mean_loo:
                    print("  → LLM priors show better predictive performance")
                else:
                    print("  → Existing priors show better predictive performance")
            
            print(f"\nDetailed results available in: {TABLES_DIR.parent}")
            print("="*60)
            
        except Exception as e:
            logger.warning(f"Could not generate final summary: {e}")


def main():
    """Main function to run the experiment."""
    
    # Ensure directories exist
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create and run experiment
    experiment = ExperimentRunner(random_seed=SEED)
    experiment.run_complete_experiment()


if __name__ == "__main__":
    main()
