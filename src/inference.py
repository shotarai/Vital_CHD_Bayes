"""
Bayesian inference for VITAL-CHD analysis: MCMC execution and posterior summaries.
"""

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from typing import Dict, Tuple, Any, Optional
import logging
from pathlib import Path

from config import MAX_RHAT, SEED, TABLES_DIR
from priors import PriorDistribution
from model_weibull_ph import create_weibull_ph_model, sample_weibull_ph_model

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class InferenceResults:
    """Class to store and summarize inference results."""
    
    def __init__(self, prior_name: str, trace: az.InferenceData, model: pm.Model):
        self.prior_name = prior_name
        self.trace = trace
        self.model = model
        self.summary = None
        self.hr_summary = None
        self._compute_summaries()
    
    def _compute_summaries(self):
        """Compute posterior summaries."""
        logger.info(f"Computing summaries for {self.prior_name}")
        
        try:
            # Standard parameter summary
            self.summary = az.summary(self.trace)
            
            # Log available variables for debugging
            available_vars = list(self.trace.posterior.data_vars)
            logger.debug(f"Available variables: {available_vars}")
            
            # Hazard ratio specific summary
            hr_var_name = None
            log_hr_var_name = None
            
            # Find the correct variable names (they have model prefix)
            for var in available_vars:
                if 'hr_intervention' in var and 'log_hr' not in var:
                    hr_var_name = var
                elif 'log_hr_intervention' in var:
                    log_hr_var_name = var
            
            # If not found, try more flexible matching
            if hr_var_name is None or log_hr_var_name is None:
                for var in available_vars:
                    if 'hr_intervention' in var:
                        if 'log' in var and log_hr_var_name is None:
                            log_hr_var_name = var
                        elif 'log' not in var and hr_var_name is None:
                            hr_var_name = var
            
            if hr_var_name is None or log_hr_var_name is None:
                raise ValueError(f"Could not find hr_intervention variables. Available: {available_vars}")
            
            logger.info(f"Using variables: hr={hr_var_name}, log_hr={log_hr_var_name}")
            
            hr_samples = self.trace.posterior[hr_var_name].values.flatten()
            log_hr_samples = self.trace.posterior[log_hr_var_name].values.flatten()
            
        except Exception as e:
            logger.error(f"Error computing summaries for {self.prior_name}: {e}")
            # Create default summary with NaN values
            self.hr_summary = {
                'prior_name': self.prior_name,
                'hr_mean': np.nan,
                'hr_median': np.nan,
                'hr_q025': np.nan,
                'hr_q975': np.nan,
                'prob_hr_less_than_1': np.nan,
                'prob_hr_less_than_095': np.nan,
                'prob_hr_less_than_090': np.nan,
                'log_hr_mean': np.nan,
                'log_hr_median': np.nan,
                'log_hr_q025': np.nan,
                'log_hr_q975': np.nan,
                'rhat_log_hr': np.nan,
                'ess_bulk_log_hr': np.nan,
                'ess_tail_log_hr': np.nan,
                'max_rhat': np.nan,
                'min_ess_bulk': np.nan,
                'n_divergent': 0
            }
            return
        
        self.hr_summary = {
            'prior_name': self.prior_name,
            'hr_mean': np.mean(hr_samples),
            'hr_median': np.median(hr_samples),
            'hr_std': np.std(hr_samples),
            'hr_q025': np.percentile(hr_samples, 2.5),
            'hr_q975': np.percentile(hr_samples, 97.5),
            'log_hr_mean': np.mean(log_hr_samples),
            'log_hr_median': np.median(log_hr_samples),
            'log_hr_std': np.std(log_hr_samples),
            'log_hr_q025': np.percentile(log_hr_samples, 2.5),
            'log_hr_q975': np.percentile(log_hr_samples, 97.5),
            'prob_hr_less_than_1': np.mean(hr_samples < 1.0),
            'prob_hr_less_than_095': np.mean(hr_samples < 0.95),
            'prob_hr_less_than_090': np.mean(hr_samples < 0.90),
            'rhat_log_hr': self.summary.loc[log_hr_var_name, 'r_hat'] if log_hr_var_name in self.summary.index else 1.0,
            'ess_bulk_log_hr': self.summary.loc[log_hr_var_name, 'ess_bulk'] if log_hr_var_name in self.summary.index else 1000,
            'ess_tail_log_hr': self.summary.loc[log_hr_var_name, 'ess_tail'] if log_hr_var_name in self.summary.index else 1000,
            'n_divergent': getattr(self.trace.sample_stats, 'diverging', np.array([])).sum(),
            'max_rhat': self.summary['r_hat'].max() if 'r_hat' in self.summary.columns else 1.0,
            'min_ess_bulk': self.summary['ess_bulk'].min() if 'ess_bulk' in self.summary.columns else 1000
        }
        
        logger.info(f"HR posterior mean: {self.hr_summary['hr_mean']:.4f}")
        logger.info(f"P(HR < 1): {self.hr_summary['prob_hr_less_than_1']:.4f}")
        logger.info(f"R-hat (log-HR): {self.hr_summary['rhat_log_hr']:.4f}")


def run_inference_single_prior(
    X: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    prior: PriorDistribution,
    intervention_idx: int = 0,
    random_seed: Optional[int] = None
) -> InferenceResults:
    """
    Run Bayesian inference for a single prior distribution.
    
    Args:
        X: Covariate matrix
        time: Event/censoring times
        event: Event indicators
        prior: Prior distribution for intervention log-HR
        intervention_idx: Index of intervention variable in X
        random_seed: Random seed for reproducibility
        
    Returns:
        InferenceResults: Results object with trace and summaries
    """
    if random_seed is None:
        random_seed = SEED
    
    logger.info(f"Running inference for prior: {prior.name}")
    
    try:
        # Create model
        model = create_weibull_ph_model(
            X=X,
            time=time,
            event=event,
            log_hr_prior=prior,
            intervention_idx=intervention_idx,
            model_name=f"weibull_ph_{prior.name}"
        )
        
        # Sample from model
        trace = sample_weibull_ph_model(model, random_seed=random_seed)
        
        # Create results object
        results = InferenceResults(prior.name, trace, model)
        
        # Check convergence
        if not np.isnan(results.hr_summary['max_rhat']) and results.hr_summary['max_rhat'] > MAX_RHAT:
            logger.warning(f"Convergence warning for {prior.name}: "
                          f"max R-hat = {results.hr_summary['max_rhat']:.4f} > {MAX_RHAT}")
        
        if results.hr_summary['n_divergent'] > 0:
            logger.warning(f"Divergent transitions for {prior.name}: "
                          f"{results.hr_summary['n_divergent']}")
        
        return results
        
    except Exception as e:
        logger.error(f"Failed to run inference for {prior.name}: {e}")
        raise


def run_inference_all_priors(
    X: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    priors: Dict[str, PriorDistribution],
    intervention_idx: int = 0,
    random_seed: Optional[int] = None
) -> Dict[str, InferenceResults]:
    """
    Run Bayesian inference for all prior distributions.
    
    Args:
        X: Covariate matrix
        time: Event/censoring times
        event: Event indicators
        priors: Dictionary of prior distributions
        intervention_idx: Index of intervention variable in X
        random_seed: Random seed for reproducibility
        
    Returns:
        Dict[str, InferenceResults]: Results for each prior
    """
    if random_seed is None:
        random_seed = SEED
    
    logger.info(f"Running inference for {len(priors)} priors")
    
    results = {}
    
    for i, (prior_name, prior) in enumerate(priors.items()):
        logger.info(f"Progress: {i+1}/{len(priors)} - {prior_name}")
        
        # Use different seed for each prior to avoid identical results
        prior_seed = random_seed + i
        
        try:
            results[prior_name] = run_inference_single_prior(
                X=X,
                time=time,
                event=event,
                prior=prior,
                intervention_idx=intervention_idx,
                random_seed=prior_seed
            )
            logger.info(f"Successfully completed inference for {prior_name}")
            
        except Exception as e:
            logger.error(f"Failed to run inference for {prior_name}: {e}")
            continue
    
    logger.info(f"Completed inference for {len(results)}/{len(priors)} priors")
    return results


def create_inference_summary_table(results: Dict[str, InferenceResults]) -> pd.DataFrame:
    """
    Create a summary table of inference results across all priors.
    
    Args:
        results: Dictionary of inference results
        
    Returns:
        pd.DataFrame: Summary table
    """
    logger.info("Creating inference summary table")
    
    # Handle empty results
    if not results:
        logger.warning("No inference results to summarize")
        # Create empty DataFrame with expected columns
        column_order = [
            'prior_name',
            'hr_mean', 'hr_median', 'hr_q025', 'hr_q975',
            'prob_hr_less_than_1', 'prob_hr_less_than_095', 'prob_hr_less_than_090',
            'log_hr_mean', 'log_hr_median', 'log_hr_q025', 'log_hr_q975',
            'rhat_log_hr', 'ess_bulk_log_hr', 'ess_tail_log_hr',
            'max_rhat', 'min_ess_bulk', 'n_divergent'
        ]
        return pd.DataFrame(columns=column_order)
    
    summary_data = []
    
    for prior_name, result in results.items():
        summary_data.append(result.hr_summary)
    
    df = pd.DataFrame(summary_data)
    
    # Reorder columns for better readability
    column_order = [
        'prior_name',
        'hr_mean', 'hr_median', 'hr_q025', 'hr_q975',
        'prob_hr_less_than_1', 'prob_hr_less_than_095', 'prob_hr_less_than_090',
        'log_hr_mean', 'log_hr_median', 'log_hr_q025', 'log_hr_q975',
        'rhat_log_hr', 'ess_bulk_log_hr', 'ess_tail_log_hr',
        'max_rhat', 'min_ess_bulk', 'n_divergent'
    ]
    
    # Only reorder columns that exist in the DataFrame
    existing_columns = [col for col in column_order if col in df.columns]
    if existing_columns:
        df = df[existing_columns]
    
    # Sort by prior type (existing vs LLM) and then by name
    if 'prior_name' in df.columns:
        df['prior_type'] = df['prior_name'].apply(lambda x: 'llm' if x.startswith('llm_') else 'existing')
        df = df.sort_values(['prior_type', 'prior_name']).reset_index(drop=True)
        df = df.drop('prior_type', axis=1)
    
    logger.info(f"Summary table created with {len(df)} rows")
    return df


def save_inference_results(
    results: Dict[str, InferenceResults],
    output_dir: Optional[Path] = None
) -> None:
    """
    Save inference results to files.
    
    Args:
        results: Dictionary of inference results
        output_dir: Directory to save results (default: TABLES_DIR)
    """
    if output_dir is None:
        output_dir = TABLES_DIR
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving inference results to {output_dir}")
    
    # Create and save summary table
    summary_df = create_inference_summary_table(results)
    summary_path = output_dir / "inference_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"Saved inference summary to {summary_path}")
    
    # Save individual trace summaries
    for prior_name, result in results.items():
        trace_summary_path = output_dir / f"trace_summary_{prior_name}.csv"
        result.summary.to_csv(trace_summary_path)
        logger.info(f"Saved trace summary for {prior_name} to {trace_summary_path}")
    
    logger.info("All inference results saved successfully")


def load_inference_results(
    results_dir: Optional[Path] = None
) -> pd.DataFrame:
    """
    Load previously saved inference results.
    
    Args:
        results_dir: Directory containing results (default: TABLES_DIR)
        
    Returns:
        pd.DataFrame: Inference summary table
    """
    if results_dir is None:
        results_dir = TABLES_DIR
    
    results_path = Path(results_dir) / "inference_summary.csv"
    
    if not results_path.exists():
        raise FileNotFoundError(f"Inference results not found at {results_path}")
    
    logger.info(f"Loading inference results from {results_path}")
    df = pd.read_csv(results_path)
    
    return df


if __name__ == "__main__":
    # Test inference with dummy data
    try:
        from priors import get_existing_priors
        from data_io import get_processed_data
        
        # Get real data (if available) or create dummy data
        try:
            X, time, event = get_processed_data()
            logger.info("Using real VITAL data for testing")
        except:
            logger.info("Creating dummy data for testing")
            np.random.seed(SEED)
            n_obs = 100
            X = np.random.randn(n_obs, 2)
            time = np.random.exponential(2, n_obs)
            event = np.random.binomial(1, 0.3, n_obs)
        
        # Get a subset of priors for testing
        all_priors = get_existing_priors()
        test_priors = {
            'noninformative': all_priors['noninformative'],
            'primary_informed': all_priors['primary_informed']
        }
        
        # Run inference
        results = run_inference_all_priors(X, time, event, test_priors)
        
        # Create summary
        summary_df = create_inference_summary_table(results)
        print("Inference summary:")
        print(summary_df[['prior_name', 'hr_mean', 'hr_q025', 'hr_q975', 'prob_hr_less_than_1']])
        
    except Exception as e:
        print(f"Error testing inference: {e}")
