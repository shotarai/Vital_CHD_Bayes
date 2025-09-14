"""
GPT-4 prior-only inference script for VITAL-CHD Bayesian analysis.

This script runs inference only for GPT-4 generated prior distribution:
1. Load and preprocess data
2. Get GPT-4 generated prior distribution
3. Run Bayesian inference for GPT-4 prior only
4. Save results and generate summary
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
from priors import query_openai_gpt4_prior, PriorDistribution
from inference import run_inference_single_prior, save_inference_results
from reporting import plot_hr_comparison

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(TABLES_DIR / 'gpt5_inference_experiment.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class GPT4InferenceRunner:
    """GPT-4 prior-only inference runner class."""
    
    def __init__(self, random_seed: Optional[int] = None):
        """
        Initialize GPT-4 inference runner.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed or SEED
        self.results = {}
        
        logger.info("=" * 60)
        logger.info("VITAL-CHD BAYESIAN INFERENCE (GPT-4 PRIOR ONLY)")
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
            
    def step_2_get_gpt4_prior(self):
        """Step 2: Get GPT-4 generated prior distribution."""
        logger.info("STEP 2: Getting GPT-4 prior distribution")
        logger.info("-" * 40)
        
        try:
            # Query GPT-4 for prior
            result = query_openai_gpt4_prior(temperature=0.0)
            
            if result is None:
                raise RuntimeError("Failed to obtain prior from GPT-4")
            
            mu, sigma = result
            
            # Create PriorDistribution object
            gpt4_prior = PriorDistribution(
                name="llm_gpt_4",
                mu=mu,
                sigma=sigma,
                description="LLM-generated prior from OpenAI GPT-4",
                source="llm"
            )
            
            self.results['prior'] = gpt4_prior
            
            logger.info(f"GPT-4 prior obtained successfully:")
            logger.info(f"  Name: {gpt4_prior.name}")
            logger.info(f"  Distribution: N({gpt4_prior.mu:.4f}, {gpt4_prior.sigma:.4f}^2)")
            logger.info(f"  Description: {gpt4_prior.description}")
                
        except Exception as e:
            logger.error(f"Failed to get GPT-4 prior: {e}")
            raise
            
    def step_3_run_inference(self):
        """Step 3: Run Bayesian inference for GPT-4 prior."""
        logger.info("STEP 3: Running Bayesian inference for GPT-4 prior")
        logger.info("-" * 40)
        
        try:
            start_time = time.time()
            
            # Run inference for GPT-4 prior
            inference_result = run_inference_single_prior(
                X=self.results['data']['X'],
                time=self.results['data']['time'],
                event=self.results['data']['event'],
                prior=self.results['prior'],
                random_seed=self.random_seed
            )
            
            self.results['inference'] = {
                self.results['prior'].name: inference_result
            }
            
            # Save inference results
            save_inference_results(self.results['inference'], save_traces=True)
            
            elapsed_time = time.time() - start_time
            
            logger.info(f"Inference completed:")
            logger.info(f"  Model: {self.results['prior'].name}")
            logger.info(f"  Time elapsed: {elapsed_time/60:.1f} minutes")
            
            # Log convergence summary
            hr_summary = inference_result.hr_summary
            if hr_summary['max_rhat'] > 1.01:
                logger.warning(f"Convergence warning: max R-hat = {hr_summary['max_rhat']:.4f} > 1.01")
            else:
                logger.info(f"Good convergence: max R-hat = {hr_summary['max_rhat']:.4f} ≤ 1.01")
            
            if hr_summary['n_divergent'] > 0:
                logger.warning(f"Divergent transitions: {hr_summary['n_divergent']}")
            else:
                logger.info("No divergent transitions")
                
        except Exception as e:
            logger.error(f"Failed to run inference: {e}")
            raise
            
    def step_4_generate_report(self):
        """Step 4: Generate summary report for GPT-4 results."""
        logger.info("STEP 4: Generating GPT-4 inference report")
        logger.info("-" * 40)
        
        try:
            inference_result = self.results['inference'][self.results['prior'].name]
            hr_summary = inference_result.hr_summary
            
            # Save individual summary
            gpt4_summary_path = TABLES_DIR / "gpt4_inference_summary.csv"
            
            import pandas as pd
            summary_data = {
                'prior_name': [hr_summary['prior_name']],
                'hr_mean': [hr_summary['hr_mean']],
                'hr_median': [hr_summary['hr_median']],
                'hr_q025': [hr_summary['hr_q025']],
                'hr_q975': [hr_summary['hr_q975']],
                'prob_hr_less_than_1': [hr_summary['prob_hr_less_than_1']],
                'prob_hr_less_than_095': [hr_summary['prob_hr_less_than_095']],
                'prob_hr_less_than_090': [hr_summary['prob_hr_less_than_090']],
                'log_hr_mean': [hr_summary['log_hr_mean']],
                'log_hr_median': [hr_summary['log_hr_median']],
                'log_hr_q025': [hr_summary['log_hr_q025']],
                'log_hr_q975': [hr_summary['log_hr_q975']],
                'rhat_log_hr': [hr_summary['rhat_log_hr']],
                'ess_bulk_log_hr': [hr_summary['ess_bulk_log_hr']],
                'ess_tail_log_hr': [hr_summary['ess_tail_log_hr']],
                'max_rhat': [hr_summary['max_rhat']],
                'min_ess_bulk': [hr_summary['min_ess_bulk']],
                'n_divergent': [hr_summary['n_divergent']]
            }
            
            df = pd.DataFrame(summary_data)
            df.to_csv(gpt4_summary_path, index=False)
            logger.info(f"Saved GPT-4 summary to {gpt4_summary_path}")
            
            logger.info("GPT-4 inference report generated successfully")
            
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")
            raise
            
    def run_gpt4_inference_experiment(self):
        """Run the GPT-4 inference-only experiment pipeline."""
        start_time = time.time()
        
        try:
            self.step_1_load_data()
            self.step_2_get_gpt4_prior()
            self.step_3_run_inference()
            self.step_4_generate_report()
            
            total_time = time.time() - start_time
            
            logger.info("=" * 60)
            logger.info("GPT-4 INFERENCE EXPERIMENT COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            logger.info(f"Total time: {total_time/60:.1f} minutes")
            logger.info(f"Results saved to: {TABLES_DIR}")
            
            # Print GPT-4 summary
            self._print_gpt4_summary()
            
        except Exception as e:
            logger.error(f"GPT-4 inference experiment failed: {e}")
            raise
            
    def _print_gpt4_summary(self):
        """Print a summary of GPT-4 inference results."""
        try:
            inference_result = self.results['inference'][self.results['prior'].name]
            hr_summary = inference_result.hr_summary
            prior = self.results['prior']
            
            print("\n" + "="*60)
            print("GPT-4 INFERENCE SUMMARY")
            print("="*60)
            
            # Data summary
            print(f"Dataset: {self.results['data']['n_obs']} observations, "
                  f"{self.results['data']['n_events']} events "
                  f"({self.results['data']['event_rate']:.1%})")
            
            # Prior summary
            print(f"\nGPT-4 Generated Prior:")
            print(f"  Distribution: N({prior.mu:.4f}, {prior.sigma:.4f}^2)")
            print(f"  Description: {prior.description}")
            
            # Inference results
            print(f"\nInference Results:")
            print(f"  Hazard Ratio (HR):")
            print(f"    Mean: {hr_summary['hr_mean']:.4f}")
            print(f"    Median: {hr_summary['hr_median']:.4f}")
            print(f"    95% CrI: ({hr_summary['hr_q025']:.4f}, {hr_summary['hr_q975']:.4f})")
            print(f"  Probability of Benefit:")
            print(f"    P(HR < 1.0): {hr_summary['prob_hr_less_than_1']:.4f}")
            print(f"    P(HR < 0.95): {hr_summary['prob_hr_less_than_095']:.4f}")
            print(f"    P(HR < 0.90): {hr_summary['prob_hr_less_than_090']:.4f}")
            
            # Convergence diagnostics
            print(f"\nConvergence Diagnostics:")
            print(f"  Maximum R-hat: {hr_summary['max_rhat']:.4f}")
            print(f"  Minimum ESS (bulk): {hr_summary['min_ess_bulk']:.0f}")
            print(f"  Divergent transitions: {hr_summary['n_divergent']}")
            
            if hr_summary['max_rhat'] <= 1.01:
                print("  ✓ Good convergence (R-hat ≤ 1.01)")
            else:
                print("  ⚠ Potential convergence issues (R-hat > 1.01)")
            
            # Clinical interpretation
            print(f"\nClinical Interpretation:")
            if hr_summary['hr_median'] < 1.0:
                reduction = (1 - hr_summary['hr_median']) * 100
                print(f"  GPT-4 suggests {reduction:.1f}% risk reduction")
                if hr_summary['prob_hr_less_than_1'] > 0.95:
                    print(f"  High confidence in protective effect (P > 0.95)")
                elif hr_summary['prob_hr_less_than_1'] > 0.90:
                    print(f"  Moderate confidence in protective effect (P > 0.90)")
                else:
                    print(f"  Low confidence in protective effect (P < 0.90)")
            else:
                print(f"  GPT-4 suggests potential harm or no benefit")
            
            print(f"\nDetailed results available in: {TABLES_DIR}")
            print("="*60)
            
        except Exception as e:
            logger.warning(f"Could not generate GPT-4 summary: {e}")


def main():
    """Main function to run the GPT-4 inference-only experiment."""
    
    # Ensure directories exist
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create and run experiment
    experiment = GPT4InferenceRunner(random_seed=SEED)
    experiment.run_gpt4_inference_experiment()


if __name__ == "__main__":
    main()
