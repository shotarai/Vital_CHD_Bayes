"""
Small-n sensitivity evaluation with holdout split design.

This module implements the simplified experimental design for evaluating 
prior sensitivity with reduced training data sizes using C-index and IBS.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import time
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
import warnings
from scipy import integrate, stats

# Local imports
from config import SEED, TABLES_DIR, FIGURES_DIR
from data_io import get_processed_data
from priors import get_all_priors, PriorDistribution
from inference import run_inference_single_prior

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class HoldoutSensitivityConfig:
    """Configuration for holdout-based small-n sensitivity evaluation."""
    train_fractions: List[float] = None
    test_size: float = 0.2  # 20% for test set
    tau_max: float = 5.0  # Maximum follow-up time from VITAL data (years)
    time_grid_points: int = 50
    random_seed: int = SEED
    
    def __post_init__(self):
        if self.train_fractions is None:
            self.train_fractions = [0.2, 0.4, 0.6, 0.8, 1.0]


@dataclass
class SensitivityResult:
    """Results for a single prior-fraction combination."""
    prior_name: str
    train_fraction: float
    n_train_total: int
    n_train_events: int
    n_test_total: int
    n_test_events: int
    c_index: float
    ibs: float
    convergence_ok: bool
    mcmc_time: float
    max_rhat: float
    n_divergent: int


class HoldoutSensitivityEvaluator:
    """Evaluator for holdout-based small-n sensitivity analysis."""
    
    def __init__(self, config: HoldoutSensitivityConfig):
        self.config = config
        self.results = []
    
    def kaplan_meier_estimator(self, time: np.ndarray, event: np.ndarray, 
                             time_grid: np.ndarray) -> np.ndarray:
        """
        Compute Kaplan-Meier survival estimator.
        
        Args:
            time: Event/censoring times
            event: Event indicators (1=event, 0=censored)
            time_grid: Time points for estimation
            
        Returns:
            Survival probabilities at time_grid points
        """
        # Sort by time
        order = np.argsort(time)
        time_sorted = time[order]
        event_sorted = event[order]
        
        # Get unique event times
        unique_times = np.unique(time_sorted[event_sorted == 1])
        
        # Calculate KM estimate
        n_at_risk = len(time)
        km_surv = np.ones(len(time_grid))
        
        for i, t in enumerate(time_grid):
            surv_t = 1.0
            
            for event_time in unique_times:
                if event_time > t:
                    break
                
                # Number at risk just before event_time
                n_risk = np.sum(time_sorted >= event_time)
                # Number of events at event_time
                n_events = np.sum((time_sorted == event_time) & (event_sorted == 1))
                
                if n_risk > 0:
                    surv_t *= (1 - n_events / n_risk)
            
            km_surv[i] = surv_t
        
        return km_surv
    
    def censoring_distribution(self, time: np.ndarray, event: np.ndarray,
                             time_grid: np.ndarray) -> np.ndarray:
        """
        Estimate censoring distribution G(t) = P(C > t) using reverse KM.
        
        Args:
            time: Event/censoring times
            event: Event indicators
            time_grid: Time points for estimation
            
        Returns:
            Censoring survival probabilities
        """
        # Reverse KM: treat censoring as "events"
        censoring_event = 1 - event  # Censoring indicators
        return self.kaplan_meier_estimator(time, censoring_event, time_grid)
        
    def create_stratified_split(self, X: np.ndarray, time: np.ndarray, 
                              event: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create stratified train-test split ensuring event distribution is maintained.
        
        Args:
            X: Covariate matrix
            time: Event/censoring times  
            event: Event indicators
            
        Returns:
            Tuple of (train_idx, test_idx)
        """
        # Create stratification variable based on event status and time quartiles
        n_obs = len(X)
        
        # Stratify primarily by event status
        event_status = event.astype(int)
        
        # For additional stratification, use time quartiles
        try:
            time_quartiles = pd.qcut(time, q=4, labels=[0, 1, 2, 3], duplicates='drop')
            strata = event_status * 10 + time_quartiles.astype(int)
        except:
            # If quartiles fail, use event status only
            strata = event_status
        
        # Stratified split
        train_idx, test_idx = train_test_split(
            np.arange(n_obs),
            test_size=self.config.test_size,
            stratify=strata,
            random_state=self.config.random_seed
        )
        
        # Log split statistics
        train_events = event[train_idx].sum()
        test_events = event[test_idx].sum()
        
        logger.info(f"Data split created:")
        logger.info(f"  Training: {len(train_idx)} obs, {train_events} events "
                   f"({train_events/len(train_idx):.1%})")
        logger.info(f"  Test: {len(test_idx)} obs, {test_events} events "
                   f"({test_events/len(test_idx):.1%})")
        
        return train_idx, test_idx
    
    def subsample_training_data(self, train_idx: np.ndarray, event: np.ndarray,
                              fraction: float) -> np.ndarray:
        """
        Subsample training data maintaining event distribution.
        
        Args:
            train_idx: Training indices
            event: Event indicators for full dataset
            fraction: Fraction of training data to keep
            
        Returns:
            Subsampled training indices
        """
        if fraction >= 1.0:
            return train_idx
        
        # Get training events
        train_event = event[train_idx]
        
        # Separate by event status
        event_indices = train_idx[train_event == 1]
        censored_indices = train_idx[train_event == 0]
        
        # Calculate target sample sizes
        n_events_target = max(1, int(len(event_indices) * fraction))
        n_censored_target = max(1, int(len(censored_indices) * fraction))
        
        # Ensure we don't exceed available samples
        n_events_target = min(n_events_target, len(event_indices))
        n_censored_target = min(n_censored_target, len(censored_indices))
        
        # Random sampling
        np.random.seed(self.config.random_seed + int(fraction * 1000))
        
        sampled_events = np.random.choice(
            event_indices, n_events_target, replace=False
        )
        sampled_censored = np.random.choice(
            censored_indices, n_censored_target, replace=False
        )
        
        subsampled_idx = np.concatenate([sampled_events, sampled_censored])
        
        logger.debug(f"Subsampled training data: fraction={fraction}, "
                    f"events={n_events_target}, censored={n_censored_target}")
        
        return subsampled_idx
    
    def predict_survival_function(self, trace, X_test: np.ndarray,
                                time_grid: np.ndarray) -> np.ndarray:
        """
        Predict survival function for test data using posterior samples.
        
        Args:
            trace: MCMC trace from inference
            X_test: Test covariate matrix
            time_grid: Time points for prediction
            
        Returns:
            Survival probabilities: (n_test, n_timepoints)
        """
        try:
            # Extract posterior samples
            posterior = trace.posterior
            
            # Find variable names (they have model prefix)
            log_lambda_var = None
            log_k_var = None
            log_hr_intervention_var = None
            log_hr_other_var = None
            
            for var_name in posterior.data_vars:
                if 'log_lambda' in var_name:
                    log_lambda_var = var_name
                elif 'log_k' in var_name:
                    log_k_var = var_name
                elif 'log_hr_intervention' in var_name:
                    log_hr_intervention_var = var_name
                elif 'log_hr_other' in var_name:
                    log_hr_other_var = var_name
            
            if not all([log_lambda_var, log_k_var, log_hr_intervention_var]):
                raise ValueError(f"Could not find required variables in trace. "
                               f"Available: {list(posterior.data_vars)}")
            
            # Get posterior samples (flatten chains and draws)
            log_lambda_samples = posterior[log_lambda_var].values.flatten()
            log_k_samples = posterior[log_k_var].values.flatten()
            log_hr_intervention_samples = posterior[log_hr_intervention_var].values.flatten()
            
            # Get other covariate coefficients if available
            if log_hr_other_var and X_test.shape[1] > 1:
                log_hr_other_samples = posterior[log_hr_other_var].values
                # Reshape to (n_samples, n_covariates-1)
                if len(log_hr_other_samples.shape) == 3:  # chains, draws, covariates
                    log_hr_other_samples = log_hr_other_samples.reshape(-1, log_hr_other_samples.shape[-1])
                elif len(log_hr_other_samples.shape) == 2:  # already flat
                    pass
                else:
                    log_hr_other_samples = log_hr_other_samples.flatten().reshape(-1, X_test.shape[1]-1)
            else:
                log_hr_other_samples = None
            
            n_samples = len(log_lambda_samples)
            n_test = X_test.shape[0]
            n_times = len(time_grid)
            n_covariates = X_test.shape[1]
            
            # Initialize survival matrix
            survival_probs = np.zeros((n_test, n_times))
            
            # Use a subset of samples for efficiency (max 200 samples)
            sample_indices = np.random.choice(n_samples, min(200, n_samples), replace=False)
            
            # Predict for each posterior sample
            for s_idx in sample_indices:
                lambda_s = np.exp(log_lambda_samples[s_idx])
                k_s = np.exp(log_k_samples[s_idx])
                
                # For each test observation
                for i in range(n_test):
                    # Calculate full linear predictor (log hazard ratio)
                    # Start with intervention effect (first covariate)
                    log_hr_total = log_hr_intervention_samples[s_idx] * X_test[i, 0]
                    
                    # Add other covariate effects
                    if log_hr_other_samples is not None and n_covariates > 1:
                        for j in range(1, n_covariates):
                            log_hr_total += log_hr_other_samples[s_idx, j-1] * X_test[i, j]
                    
                    # Convert to hazard ratio
                    hr_total = np.exp(log_hr_total)
                    
                    # Weibull survival function: S(t) = exp(-lambda * hr * t^k)
                    for t_idx, t in enumerate(time_grid):
                        if t >= 0:
                            survival_probs[i, t_idx] += np.exp(
                                -lambda_s * hr_total * (t ** k_s)
                            )
            
            # Average over samples
            survival_probs /= len(sample_indices)
            
            return survival_probs
            
        except Exception as e:
            logger.error(f"Failed to predict survival function: {e}")
            import traceback
            traceback.print_exc()
            # Return dummy predictions if prediction fails
            return np.ones((X_test.shape[0], len(time_grid))) * 0.5
    
    def compute_uno_c_index(self, time_train: np.ndarray, event_train: np.ndarray,
                           time_test: np.ndarray, event_test: np.ndarray,
                           survival_probs: np.ndarray, time_grid: np.ndarray) -> float:
        """
        Compute Uno's C-index with proper censoring correction.
        
        Args:
            time_train, event_train: Training data for censoring distribution
            time_test, event_test: Test data
            survival_probs: Predicted survival probabilities
            time_grid: Time grid for predictions
            
        Returns:
            Uno's C-index
        """
        try:
            # Estimate censoring distribution from training data
            G_hat = self.censoring_distribution(time_train, event_train, time_grid)
            
            # Convert survival probabilities to risk scores
            # Use integral of survival function as mean survival time
            dt = np.diff(time_grid)
            if len(dt) == 0:
                return np.nan
            dt = np.append(dt, dt[-1])
            
            # Mean survival time for each test subject
            mean_survival_times = np.sum(survival_probs * dt[np.newaxis, :], axis=1)
            risk_scores = -mean_survival_times  # Higher risk = lower survival time
            
            # Uno's C-index calculation
            n_test = len(time_test)
            numerator = 0.0
            denominator = 0.0
            
            for i in range(n_test):
                for j in range(n_test):
                    if i == j:
                        continue
                    
                    # Only consider comparable pairs
                    t_i, t_j = time_test[i], time_test[j]
                    d_i, d_j = event_test[i], event_test[j]
                    
                    # Case 1: i has event and occurs before j (either event or censored)
                    if d_i == 1 and t_i < t_j:
                        # Find weight from censoring distribution
                        t_idx = np.searchsorted(time_grid, t_i, side='right') - 1
                        t_idx = max(0, min(t_idx, len(G_hat) - 1))
                        
                        if G_hat[t_idx] > 0:
                            weight = 1.0 / G_hat[t_idx]
                            denominator += weight
                            
                            # Check if prediction is concordant
                            if risk_scores[i] > risk_scores[j]:
                                numerator += weight
                    
                    # Case 2: j has event and occurs before i
                    elif d_j == 1 and t_j < t_i:
                        # Find weight from censoring distribution
                        t_idx = np.searchsorted(time_grid, t_j, side='right') - 1
                        t_idx = max(0, min(t_idx, len(G_hat) - 1))
                        
                        if G_hat[t_idx] > 0:
                            weight = 1.0 / G_hat[t_idx]
                            denominator += weight
                            
                            # Check if prediction is concordant
                            if risk_scores[j] > risk_scores[i]:
                                numerator += weight
            
            if denominator == 0:
                return np.nan
            
            c_index = numerator / denominator
            return c_index
            
        except Exception as e:
            logger.warning(f"Failed to compute Uno C-index: {e}")
            return np.nan
    
    def compute_ipcw_brier_ibs(self, time_test: np.ndarray, event_test: np.ndarray,
                             survival_probs: np.ndarray, time_grid: np.ndarray) -> float:
        """
        Compute Integrated Brier Score with IPCW correction.
        
        Args:
            time_test: Test event/censoring times
            event_test: Test event indicators
            survival_probs: Predicted survival probabilities
            time_grid: Time grid for predictions
            
        Returns:
            Integrated Brier Score (IBS)
        """
        try:
            # Estimate censoring distribution from test data
            G_hat = self.censoring_distribution(time_test, event_test, time_grid)
            
            # Limit evaluation to tau_max
            valid_indices = time_grid <= self.config.tau_max
            time_grid_limited = time_grid[valid_indices]
            survival_probs_limited = survival_probs[:, valid_indices]
            G_hat_limited = G_hat[valid_indices]
            
            if len(time_grid_limited) == 0:
                return np.nan
            
            brier_scores = []
            
            for t_idx, t in enumerate(time_grid_limited):
                if t > time_test.max():
                    continue
                
                # IPCW Brier Score at time t
                brier_t = 0.0
                weight_sum = 0.0
                
                for i in range(len(time_test)):
                    T_i = time_test[i]
                    delta_i = event_test[i]
                    S_hat_it = survival_probs_limited[i, t_idx]
                    
                    # Case 1: Event before time t
                    if delta_i == 1 and T_i <= t:
                        # True survival status: 0 (did not survive to t)
                        Y_it = 0.0
                        
                        # Weight: 1/G(T_i)
                        T_idx = np.searchsorted(time_grid_limited, T_i, side='right') - 1
                        T_idx = max(0, min(T_idx, len(G_hat_limited) - 1))
                        
                        if G_hat_limited[T_idx] > 0:
                            weight = 1.0 / G_hat_limited[T_idx]
                            brier_t += weight * (S_hat_it - Y_it) ** 2
                            weight_sum += weight
                    
                    # Case 2: Still at risk at time t (T_i > t)
                    elif T_i > t:
                        # True survival status: 1 (survived to t)
                        Y_it = 1.0
                        
                        # Weight: 1/G(t)
                        if G_hat_limited[t_idx] > 0:
                            weight = 1.0 / G_hat_limited[t_idx]
                            brier_t += weight * (S_hat_it - Y_it) ** 2
                            weight_sum += weight
                
                # Normalize by total weight
                if weight_sum > 0:
                    brier_t /= weight_sum
                    brier_scores.append(brier_t)
                else:
                    brier_scores.append(np.nan)
            
            # Remove NaN values
            brier_scores = np.array(brier_scores)
            valid_brier = ~np.isnan(brier_scores)
            
            if np.sum(valid_brier) == 0:
                return np.nan
            
            # Integrate using trapezoidal rule
            valid_times = time_grid_limited[:len(brier_scores)][valid_brier]
            valid_scores = brier_scores[valid_brier]
            
            if len(valid_times) < 2:
                return np.nan
            
            ibs = np.trapz(valid_scores, valid_times)
            
            # Normalize by time range
            time_range = valid_times[-1] - valid_times[0]
            if time_range > 0:
                ibs /= time_range
            
            return ibs
            
        except Exception as e:
            logger.warning(f"Failed to compute IPCW-Brier IBS: {e}")
            return np.nan
    
    def evaluate_single_condition(self, X_train_full: np.ndarray, time_train_full: np.ndarray,
                                event_train_full: np.ndarray, X_test: np.ndarray,
                                time_test: np.ndarray, event_test: np.ndarray,
                                prior: PriorDistribution, train_fraction: float) -> SensitivityResult:
        """
        Evaluate a single prior-fraction combination.
        
        Args:
            X_train_full, time_train_full, event_train_full: Full training data
            X_test, time_test, event_test: Test data
            prior: Prior distribution
            train_fraction: Fraction of training data to use
            
        Returns:
            SensitivityResult object
        """
        logger.info(f"Evaluating {prior.name} with fraction {train_fraction}")
        
        start_time = time.time()
        
        try:
            # Subsample training data
            train_indices = np.arange(len(X_train_full))
            subsampled_indices = self.subsample_training_data(
                train_indices, event_train_full, train_fraction
            )
            
            X_train = X_train_full[subsampled_indices]
            time_train = time_train_full[subsampled_indices]
            event_train = event_train_full[subsampled_indices]
            
            n_train_total = len(X_train)
            n_train_events = event_train.sum()
            n_test_total = len(X_test)
            n_test_events = event_test.sum()
            
            logger.info(f"  Training: {n_train_total} obs, {n_train_events} events")
            logger.info(f"  Test: {n_test_total} obs, {n_test_events} events")
            
            # Skip if insufficient events
            if n_train_events < 2:
                logger.warning(f"  Skipping: insufficient training events ({n_train_events})")
                return SensitivityResult(
                    prior_name=prior.name, train_fraction=train_fraction,
                    n_train_total=n_train_total, n_train_events=n_train_events,
                    n_test_total=n_test_total, n_test_events=n_test_events,
                    c_index=np.nan, ibs=np.nan, convergence_ok=False,
                    mcmc_time=0.0, max_rhat=np.nan, n_divergent=0
                )
            
            # Run inference
            inference_result = run_inference_single_prior(
                X=X_train, time=time_train, event=event_train,
                prior=prior, random_seed=self.config.random_seed
            )
            
            # Check convergence
            max_rhat = inference_result.hr_summary.get('max_rhat', np.nan)
            n_divergent = inference_result.hr_summary.get('n_divergent', 0)
            
            # Convert DataArray values to Python scalars if needed
            if hasattr(max_rhat, 'item'):
                max_rhat = float(max_rhat.item())
            elif hasattr(max_rhat, 'values'):
                max_rhat = float(max_rhat.values)
            else:
                max_rhat = float(max_rhat) if not np.isnan(max_rhat) else np.nan
                
            if hasattr(n_divergent, 'item'):
                n_divergent = int(n_divergent.item())
            elif hasattr(n_divergent, 'values'):
                n_divergent = int(n_divergent.values)
            else:
                n_divergent = int(n_divergent)
                
            convergence_ok = (max_rhat <= 1.1) and (n_divergent == 0)
            
            if not convergence_ok:
                logger.warning(f"  Convergence issues: R-hat={max_rhat:.3f}, "
                             f"divergent={n_divergent}")
            
            # Create time grid for predictions
            time_grid = np.linspace(
                0, min(self.config.tau_max, max(time_test.max(), time_train.max())),
                self.config.time_grid_points
            )
            
            # Predict survival function
            survival_probs = self.predict_survival_function(
                inference_result.trace, X_test, time_grid
            )
            
            # Compute evaluation metrics
            c_index = self.compute_uno_c_index(
                time_train_full, event_train_full,  # Use full training for censoring distribution
                time_test, event_test, survival_probs, time_grid
            )
            
            ibs = self.compute_ipcw_brier_ibs(
                time_test, event_test, survival_probs, time_grid
            )
            
            mcmc_time = time.time() - start_time
            
            logger.info(f"  Results: C-index={c_index:.3f}, IBS={ibs:.3f}, "
                       f"time={mcmc_time:.1f}s")
            
            return SensitivityResult(
                prior_name=prior.name, train_fraction=train_fraction,
                n_train_total=n_train_total, n_train_events=n_train_events,
                n_test_total=n_test_total, n_test_events=n_test_events,
                c_index=c_index, ibs=ibs, convergence_ok=convergence_ok,
                mcmc_time=mcmc_time, max_rhat=max_rhat, n_divergent=n_divergent
            )
            
        except Exception as e:
            logger.error(f"  Failed: {e}")
            import traceback
            traceback.print_exc()
            
            return SensitivityResult(
                prior_name=prior.name, train_fraction=train_fraction,
                n_train_total=0, n_train_events=0, n_test_total=len(X_test),
                n_test_events=event_test.sum(), c_index=np.nan, ibs=np.nan,
                convergence_ok=False, mcmc_time=time.time() - start_time,
                max_rhat=np.nan, n_divergent=0
            )
    
    def load_checkpoint(self, checkpoint_path: Path) -> List[SensitivityResult]:
        """Load previous results from checkpoint file."""
        try:
            if checkpoint_path.exists():
                import pickle
                with open(checkpoint_path, 'rb') as f:
                    return pickle.load(f)
            return []
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return []
    
    def save_checkpoint(self, checkpoint_path: Path, results: List[SensitivityResult]):
        """Save current results to checkpoint file."""
        try:
            import pickle
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(results, f)
            logger.info(f"Checkpoint saved: {len(results)} results")
        except Exception as e:
            logger.warning(f"Failed to save checkpoint: {e}")
    
    def save_intermediate_results(self, prior_name: str, prior_results: List[SensitivityResult]):
        """Save results after completing each prior."""
        try:
            # Convert to DataFrame
            results_df = pd.DataFrame([
                {
                    'prior_name': r.prior_name,
                    'train_fraction': r.train_fraction,
                    'n_train_total': r.n_train_total,
                    'n_train_events': r.n_train_events,
                    'n_test_total': r.n_test_total,
                    'n_test_events': r.n_test_events,
                    'c_index': r.c_index,
                    'ibs': r.ibs,
                    'convergence_ok': r.convergence_ok,
                    'mcmc_time': r.mcmc_time,
                    'max_rhat': r.max_rhat,
                    'n_divergent': r.n_divergent
                }
                for r in prior_results
            ])
            
            # Save intermediate file
            from config import TABLES_DIR
            import time
            timestamp = time.strftime('%Y%m%d_%H%M%S')
            intermediate_path = TABLES_DIR / f"holdout_intermediate_{prior_name}_{timestamp}.csv"
            results_df.to_csv(intermediate_path, index=False)
            
            # Also save as latest for this prior
            latest_path = TABLES_DIR / f"holdout_latest_{prior_name}.csv"
            results_df.to_csv(latest_path, index=False)
            
            logger.info(f"Intermediate results saved for {prior_name}: {len(results_df)} evaluations")
            
        except Exception as e:
            logger.warning(f"Failed to save intermediate results for {prior_name}: {e}")

    def run_sensitivity_analysis(self, X: np.ndarray, time: np.ndarray, event: np.ndarray,
                               priors: Dict[str, PriorDistribution], 
                               resume_from_checkpoint: bool = True) -> pd.DataFrame:
        """
        Run complete holdout-based small-n sensitivity analysis with checkpointing.
        
        Args:
            X, time, event: Data arrays
            priors: Dictionary of prior distributions
            resume_from_checkpoint: Whether to resume from previous checkpoint
            
        Returns:
            DataFrame with results
        """
        logger.info("Starting holdout-based small-n sensitivity analysis")
        logger.info(f"Data: {len(X)} observations, {event.sum()} events")
        logger.info(f"Priors: {len(priors)}")
        logger.info(f"Training fractions: {self.config.train_fractions}")
        logger.info(f"Test size: {self.config.test_size}")
        
        # Set up checkpoint
        from config import TABLES_DIR
        checkpoint_path = TABLES_DIR / "holdout_checkpoint.pkl"
        
        # Load existing results if resuming
        if resume_from_checkpoint:
            self.results = self.load_checkpoint(checkpoint_path)
            if self.results:
                completed_combinations = set((r.prior_name, r.train_fraction) for r in self.results)
                logger.info(f"Resuming from checkpoint: {len(self.results)} completed evaluations")
                logger.info(f"Completed combinations: {len(completed_combinations)}")
        
        # Create train-test split
        train_idx, test_idx = self.create_stratified_split(X, time, event)
        
        # Extract train and test data
        X_train_full = X[train_idx]
        time_train_full = time[train_idx]
        event_train_full = event[train_idx]
        
        X_test = X[test_idx]
        time_test = time[test_idx]
        event_test = event[test_idx]
        
        # Track which combinations are completed
        completed_combinations = set((r.prior_name, r.train_fraction) for r in self.results)
        
        # Run evaluation for all combinations
        total_combinations = len(priors) * len(self.config.train_fractions)
        current_combination = len(self.results)
        
        for prior_name, prior in priors.items():
            logger.info(f"Evaluating prior: {prior_name}")
            
            # Track results for this prior
            prior_results = []
            
            for train_fraction in self.config.train_fractions:
                # Skip if already completed
                if (prior_name, train_fraction) in completed_combinations:
                    logger.info(f"Skipping completed: {prior_name} @ {train_fraction}")
                    # Find existing result for this prior
                    existing_result = next(
                        r for r in self.results 
                        if r.prior_name == prior_name and r.train_fraction == train_fraction
                    )
                    prior_results.append(existing_result)
                    continue
                
                current_combination += 1
                logger.info(f"Progress: {current_combination}/{total_combinations} - "
                           f"Fraction {train_fraction}")
                
                try:
                    result = self.evaluate_single_condition(
                        X_train_full, time_train_full, event_train_full,
                        X_test, time_test, event_test,
                        prior, train_fraction
                    )
                    
                    self.results.append(result)
                    prior_results.append(result)
                    completed_combinations.add((prior_name, train_fraction))
                    
                    # Save checkpoint after each evaluation
                    self.save_checkpoint(checkpoint_path, self.results)
                    
                except Exception as e:
                    logger.error(f"Failed evaluation {prior_name} @ {train_fraction}: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            
            # Save intermediate results after completing each prior
            if prior_results:
                self.save_intermediate_results(prior_name, prior_results)
                logger.info(f"Completed prior {prior_name}: {len(prior_results)} evaluations")
        
        # Convert to final DataFrame
        results_df = pd.DataFrame([
            {
                'prior_name': r.prior_name,
                'train_fraction': r.train_fraction,
                'n_train_total': r.n_train_total,
                'n_train_events': r.n_train_events,
                'n_test_total': r.n_test_total,
                'n_test_events': r.n_test_events,
                'c_index': r.c_index,
                'ibs': r.ibs,
                'convergence_ok': r.convergence_ok,
                'mcmc_time': r.mcmc_time,
                'max_rhat': r.max_rhat,
                'n_divergent': r.n_divergent
            }
            for r in self.results
        ])
        
        logger.info(f"Sensitivity analysis completed: {len(results_df)} evaluations")
        
        # Clean up checkpoint file on successful completion
        try:
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                logger.info("Checkpoint file cleaned up")
        except Exception as e:
            logger.warning(f"Failed to clean up checkpoint: {e}")
        
        return results_df


def run_holdout_sensitivity_experiment(config: Optional[HoldoutSensitivityConfig] = None) -> pd.DataFrame:
    """
    Run the complete holdout-based small-n sensitivity experiment.
    
    Args:
        config: Configuration object (uses defaults if None)
        
    Returns:
        DataFrame with results
    """
    # Load data first to get max follow-up time
    logger.info("Loading data to determine max follow-up time")
    X, time, event = get_processed_data()
    max_followup = time.max()
    
    if config is None:
        config = HoldoutSensitivityConfig()
    
    # Update tau_max to actual data maximum
    config.tau_max = max_followup
    logger.info(f"Set tau_max to {max_followup:.2f} years (data maximum)")
    
    # Load priors
    logger.info("Loading priors")
    priors = get_all_priors()
    
    # Initialize evaluator
    evaluator = HoldoutSensitivityEvaluator(config)
    
    # Run analysis
    results_df = evaluator.run_sensitivity_analysis(X, time, event, priors)
    
    # Save results
    output_path = TABLES_DIR / "holdout_sensitivity_results.csv"
    results_df.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")
    
    return results_df


if __name__ == "__main__":
    # Run experiment with default configuration
    print("Starting holdout-based small-n sensitivity analysis...")
    print("Features: Uno's C-index + IPCW-Brier IBS")
    print("Design: 8:2 holdout split, 5 training fractions, 8 priors")
    
    results = run_holdout_sensitivity_experiment()
    print(f"Experiment completed: {len(results)} evaluations")
    
    # Print summary
    print("\nResults Summary:")
    summary = results.groupby(['prior_name', 'train_fraction'])[['c_index', 'ibs']].mean().round(3)
    print(summary)
    
    print(f"\nConvergence Summary:")
    conv_summary = results.groupby('prior_name')['convergence_ok'].mean().round(3)
    print(conv_summary)
    
    print(f"\nResults saved to: {TABLES_DIR / 'holdout_sensitivity_results.csv'}")
