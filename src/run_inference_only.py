"""
Inference-only script for VITAL-CHD Bayesian analysis.

This script runs only the Bayesian inference (MCMC sampling) part:
1. Load and preprocess data
2. Get all prior distributions (existing + LLM)
3. Run Bayesian inference for each prior
4. Save inference results
"""

import sys
import logging
from pathlib import Path
from typing import Optional
import time

# Add src to path to enable imports
sys.path.insert(0, str(Path(__file__).parent))

from config import SEED, TABLES_DIR, FIGURES_DIR
from data_io import get_processed_data
from priors import get_all_priors, save_priors_summary
from inference import run_inference_all_priors, save_inference_results

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(TABLES_DIR / 'inference_experiment.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class InferenceOnlyRunner:
    """Inference-only experiment runner class."""
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize inference runner.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed or SEED
        self.results = {}
        
        logger.info("=" * 60)
        logger.info("VITAL-CHD BAYESIAN INFERENCE (INFERENCE ONLY)")
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
            
            # Save inference results (with traces for predictive evaluation)
            save_inference_results(inference_results, save_traces=True)
            
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
            
    def run_inference_experiment(self):
        """Run the inference-only experiment pipeline."""
        start_time = time.time()
        
        try:
            self.step_1_load_data()
            self.step_2_get_priors()
            self.step_3_run_inference()
            
            total_time = time.time() - start_time
            
            logger.info("=" * 60)
            logger.info("INFERENCE EXPERIMENT COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            logger.info(f"Total time: {total_time/60:.1f} minutes")
            logger.info(f"Results saved to: {TABLES_DIR}")
            
            # Print inference summary
            self._print_inference_summary()
            
        except Exception as e:
            logger.error(f"Inference experiment failed: {e}")
            raise
            
    def _print_inference_summary(self):
        """Print a summary of inference results."""
        try:
            from inference import load_inference_results
            
            inference_df = load_inference_results()
            
            print("\n" + "="*60)
            print("INFERENCE SUMMARY")
            print("="*60)
            
            # Data summary
            print(f"Dataset: {self.results['data']['n_obs']} observations, "
                  f"{self.results['data']['n_events']} events "
                  f"({self.results['data']['event_rate']:.1%})")
            
            # Model summary
            print(f"Models fitted: {len(inference_df)}")
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
            
            # Highest probability of benefit
            best_prob_idx = inference_df['prob_hr_less_than_1'].idxmax()
            best_prob = inference_df.loc[best_prob_idx]
            print(f"Highest P(HR < 1): {best_prob['prior_name']}")
            print(f"  P(HR < 1) = {best_prob['prob_hr_less_than_1']:.3f}")
            print(f"  HR = {best_prob['hr_median']:.3f} "
                  f"(95% CrI: {best_prob['hr_q025']:.3f}-{best_prob['hr_q975']:.3f})")
            
            # LLM vs existing comparison
            if len(llm) > 0 and len(existing) > 0:
                llm_mean_hr = llm['hr_median'].mean()
                existing_mean_hr = existing['hr_median'].mean()
                llm_mean_prob = llm['prob_hr_less_than_1'].mean()
                existing_mean_prob = existing['prob_hr_less_than_1'].mean()
                
                print(f"\nLLM vs Existing Comparison:")
                print(f"  LLM average HR: {llm_mean_hr:.3f}")
                print(f"  Existing average HR: {existing_mean_hr:.3f}")
                print(f"  LLM average P(HR < 1): {llm_mean_prob:.3f}")
                print(f"  Existing average P(HR < 1): {existing_mean_prob:.3f}")
                
                if llm_mean_prob > existing_mean_prob:
                    print("  → LLM priors show higher probability of benefit")
                else:
                    print("  → Existing priors show higher probability of benefit")
            
            print(f"\nDetailed results available in: {TABLES_DIR}")
            print("="*60)
            print("\nNOTE: To run predictive evaluation, use:")
            print("  rye run python -m src.run_predictive_only")
            print("="*60)
            
        except Exception as e:
            logger.warning(f"Could not generate inference summary: {e}")


def main():
    """Main function to run the inference-only experiment."""
    
    # Ensure directories exist
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create and run experiment
    experiment = InferenceOnlyRunner(random_seed=SEED)
    experiment.run_inference_experiment()


if __name__ == "__main__":
    main()
