"""
Predictive performance evaluation using PSIS-LOO and WAIC.
"""

import numpy as np
import pandas as pd
import arviz as az
from typing import Dict, Tuple, Optional
import logging
from pathlib import Path

from config import TABLES_DIR
from inference import InferenceResults
from model_weibull_ph import compute_log_likelihood

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PredictiveResults:
    """Class to store predictive evaluation results."""
    
    def __init__(self, prior_name: str, inference_result: InferenceResults):
        self.prior_name = prior_name
        self.inference_result = inference_result
        self.log_likelihood = None
        self.loo_result = None
        self.waic_result = None
        self.loo_lpd = None
        self.waic_lpd = None
        self._compute_predictive_metrics()
    
    def _compute_predictive_metrics(self):
        """Compute PSIS-LOO and WAIC metrics."""
        logger.info(f"Computing predictive metrics for {self.prior_name}")
        
        # Compute pointwise log-likelihood
        self.log_likelihood = compute_log_likelihood(
            self.inference_result.model, 
            self.inference_result.trace
        )
        
        # Add log-likelihood to trace for ArviZ
        log_lik_da = az.xarray.DataArray(
            self.log_likelihood,
            dims=["chain", "draw", "obs"],
            coords={
                "chain": range(self.log_likelihood.shape[0] // 
                              len(self.inference_result.trace.posterior.chain)),
                "draw": range(len(self.inference_result.trace.posterior.draw)),
                "obs": range(self.log_likelihood.shape[1])
            }
        )
        
        # Reshape if needed for ArviZ format
        n_chains = len(self.inference_result.trace.posterior.chain)
        n_draws = len(self.inference_result.trace.posterior.draw)
        log_lik_reshaped = self.log_likelihood.reshape(n_chains, n_draws, -1)
        
        # Create proper DataArray
        log_lik_da = az.xarray.DataArray(
            log_lik_reshaped,
            dims=["chain", "draw", "obs"],
            coords={
                "chain": self.inference_result.trace.posterior.chain.values,
                "draw": self.inference_result.trace.posterior.draw.values,
                "obs": range(log_lik_reshaped.shape[2])
            }
        )
        
        # Add to trace
        if hasattr(self.inference_result.trace, 'log_likelihood'):
            self.inference_result.trace.log_likelihood = log_lik_da.to_dataset(name='log_likelihood')
        else:
            self.inference_result.trace = self.inference_result.trace.assign(
                log_likelihood={"log_likelihood": log_lik_da}
            )
        
        # Compute LOO
        try:
            self.loo_result = az.loo(self.inference_result.trace)
            self.loo_lpd = self.loo_result.elpd_loo
            logger.info(f"LOO computed successfully: elpd_loo = {self.loo_lpd:.2f}")
        except Exception as e:
            logger.error(f"Failed to compute LOO for {self.prior_name}: {e}")
            self.loo_result = None
            self.loo_lpd = np.nan
        
        # Compute WAIC
        try:
            self.waic_result = az.waic(self.inference_result.trace)
            self.waic_lpd = self.waic_result.elpd_waic
            logger.info(f"WAIC computed successfully: elpd_waic = {self.waic_lpd:.2f}")
        except Exception as e:
            logger.error(f"Failed to compute WAIC for {self.prior_name}: {e}")
            self.waic_result = None
            self.waic_lpd = np.nan
    
    def get_summary(self) -> Dict:
        """Get summary of predictive metrics."""
        summary = {
            'prior_name': self.prior_name,
            'loo_elpd': self.loo_lpd if self.loo_result else np.nan,
            'loo_se': self.loo_result.se if self.loo_result else np.nan,
            'loo_p_eff': self.loo_result.p_loo if self.loo_result else np.nan,
            'waic_elpd': self.waic_lpd if self.waic_result else np.nan,
            'waic_se': self.waic_result.se if self.waic_result else np.nan,
            'waic_p_eff': self.waic_result.p_waic if self.waic_result else np.nan,
        }
        
        # Add warning flags for problematic observations
        if self.loo_result:
            summary['loo_n_high_pareto_k'] = np.sum(
                self.loo_result.pareto_k > 0.7
            ) if hasattr(self.loo_result, 'pareto_k') else 0
        else:
            summary['loo_n_high_pareto_k'] = np.nan
            
        return summary


def compute_predictive_performance(
    inference_results: Dict[str, InferenceResults]
) -> Dict[str, PredictiveResults]:
    """
    Compute predictive performance metrics for all inference results.
    
    Args:
        inference_results: Dictionary of inference results
        
    Returns:
        Dict[str, PredictiveResults]: Predictive results for each prior
    """
    logger.info(f"Computing predictive performance for {len(inference_results)} models")
    
    predictive_results = {}
    
    for prior_name, inference_result in inference_results.items():
        logger.info(f"Processing {prior_name}")
        
        try:
            predictive_results[prior_name] = PredictiveResults(
                prior_name, inference_result
            )
            logger.info(f"Successfully computed predictive metrics for {prior_name}")
        except Exception as e:
            logger.error(f"Failed to compute predictive metrics for {prior_name}: {e}")
            continue
    
    logger.info(f"Completed predictive evaluation for {len(predictive_results)} models")
    return predictive_results


def compare_models_loo(
    predictive_results: Dict[str, PredictiveResults],
    reference_model: str = "primary_informed"
) -> pd.DataFrame:
    """
    Compare models using LOO with a reference model.
    
    Args:
        predictive_results: Dictionary of predictive results
        reference_model: Name of reference model for comparison
        
    Returns:
        pd.DataFrame: Model comparison table
    """
    logger.info(f"Comparing models using LOO with reference: {reference_model}")
    
    # Handle empty results
    if not predictive_results:
        logger.warning("No predictive results to compare")
        return pd.DataFrame(columns=[
            'prior_name', 'loo_elpd', 'loo_se', 'loo_diff', 'loo_diff_se',
            'loo_p_eff', 'loo_n_high_pareto_k', 'waic_elpd', 'waic_se', 'waic_p_eff'
        ])
    
    if reference_model not in predictive_results:
        logger.warning(f"Reference model {reference_model} not found. Using first available model.")
        reference_model = list(predictive_results.keys())[0]
    
    # Get reference LOO value
    ref_loo = predictive_results[reference_model].loo_lpd
    if np.isnan(ref_loo):
        logger.error(f"Reference model {reference_model} has invalid LOO value")
        ref_loo = 0
    
    comparison_data = []
    
    for prior_name, result in predictive_results.items():
        summary = result.get_summary()
        
        # Compute difference from reference
        loo_diff = summary['loo_elpd'] - ref_loo
        
        comparison_data.append({
            'prior_name': prior_name,
            'loo_elpd': summary['loo_elpd'],
            'loo_se': summary['loo_se'],
            'loo_diff': loo_diff,
            'loo_diff_se': summary['loo_se'],  # Approximate - proper calculation would need covariance
            'loo_p_eff': summary['loo_p_eff'],
            'loo_n_high_pareto_k': summary['loo_n_high_pareto_k'],
            'waic_elpd': summary['waic_elpd'],
            'waic_se': summary['waic_se'],
            'waic_p_eff': summary['waic_p_eff']
        })
    
    df = pd.DataFrame(comparison_data)
    
    # Sort by LOO elpd (descending - higher is better)
    df = df.sort_values('loo_elpd', ascending=False).reset_index(drop=True)
    
    # Add ranking
    df['loo_rank'] = range(1, len(df) + 1)
    
    logger.info(f"Model comparison completed with {len(df)} models")
    return df


def create_predictive_summary_table(
    predictive_results: Dict[str, PredictiveResults],
    reference_model: str = "primary_informed"
) -> pd.DataFrame:
    """
    Create a comprehensive summary table of predictive performance.
    
    Args:
        predictive_results: Dictionary of predictive results
        reference_model: Name of reference model
        
    Returns:
        pd.DataFrame: Summary table
    """
    logger.info("Creating predictive summary table")
    
    # Get model comparison
    comparison_df = compare_models_loo(predictive_results, reference_model)
    
    # Add prior type for better organization
    comparison_df['prior_type'] = comparison_df['prior_name'].apply(
        lambda x: 'llm' if x.startswith('llm_') else 'existing'
    )
    
    # Reorder columns (only use columns that exist)
    column_order = [
        'loo_rank', 'prior_name', 'prior_type',
        'loo_elpd', 'loo_se', 'loo_diff', 'loo_diff_se',
        'loo_p_eff', 'loo_n_high_pareto_k',
        'waic_elpd', 'waic_se', 'waic_p_eff'
    ]
    
    # Only select columns that exist in the DataFrame
    existing_columns = [col for col in column_order if col in comparison_df.columns]
    if existing_columns:
        comparison_df = comparison_df[existing_columns]
    
    logger.info(f"Summary table created with {len(comparison_df)} models")
    return comparison_df


def save_predictive_results(
    predictive_results: Dict[str, PredictiveResults],
    reference_model: str = "primary_informed",
    output_dir: Optional[Path] = None
) -> None:
    """
    Save predictive evaluation results to files.
    
    Args:
        predictive_results: Dictionary of predictive results
        reference_model: Reference model for comparison
        output_dir: Directory to save results (default: TABLES_DIR)
    """
    if output_dir is None:
        output_dir = TABLES_DIR
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving predictive results to {output_dir}")
    
    # Create and save summary table
    summary_df = create_predictive_summary_table(predictive_results, reference_model)
    summary_path = output_dir / "predictive_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved predictive summary to {summary_path}")
    
    # Save individual detailed results
    for prior_name, result in predictive_results.items():
        if result.loo_result:
            loo_path = output_dir / f"loo_detailed_{prior_name}.csv"
            loo_df = pd.DataFrame({
                'obs': range(len(result.loo_result.pareto_k)) if hasattr(result.loo_result, 'pareto_k') else [],
                'pareto_k': result.loo_result.pareto_k if hasattr(result.loo_result, 'pareto_k') else []
            })
            loo_df.to_csv(loo_path, index=False)
    
    logger.info("All predictive results saved successfully")


def load_predictive_results(
    results_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Load previously saved predictive results.
    
    Args:
        results_dir: Directory containing results (default: TABLES_DIR)
        
    Returns:
        pd.DataFrame: Predictive summary table
    """
    if results_dir is None:
        results_dir = TABLES_DIR
    
    results_path = Path(results_dir) / "predictive_summary.csv"
    
    if not results_path.exists():
        raise FileNotFoundError(f"Predictive results not found at {results_path}")
    
    logger.info(f"Loading predictive results from {results_path}")
    df = pd.read_csv(results_path)
    
    return df


if __name__ == "__main__":
    # Test predictive evaluation with dummy results
    try:
        from priors import get_existing_priors
        from inference import run_inference_all_priors
        from data_io import get_processed_data
        
        # Get data
        try:
            X, time, event = get_processed_data()
            logger.info("Using real VITAL data for testing")
        except:
            logger.info("Creating dummy data for testing")
            import numpy as np
            from config import SEED
            np.random.seed(SEED)
            n_obs = 100
            X = np.random.randn(n_obs, 2)
            time = np.random.exponential(2, n_obs)
            event = np.random.binomial(1, 0.3, n_obs)
        
        # Get subset of priors for testing
        all_priors = get_existing_priors()
        test_priors = {
            'noninformative': all_priors['noninformative'],
            'primary_informed': all_priors['primary_informed']
        }
        
        # Run inference (short run for testing)
        print("Running inference for testing...")
        inference_results = run_inference_all_priors(X, time, event, test_priors)
        
        # Compute predictive metrics
        print("Computing predictive metrics...")
        predictive_results = compute_predictive_performance(inference_results)
        
        # Create summary
        summary_df = create_predictive_summary_table(predictive_results)
        print("Predictive summary:")
        print(summary_df[['prior_name', 'loo_elpd', 'loo_diff', 'waic_elpd']])
        
    except Exception as e:
        print(f"Error testing predictive evaluation: {e}")
