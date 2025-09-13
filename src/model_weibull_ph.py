"""
Weibull proportional hazards model implementation using PyMC.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from typing import Optional, Dict, Any
import logging

from .config import SEED, N_CHAINS, N_DRAWS, N_TUNE, TARGET_ACCEPT
from .priors import PriorDistribution

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_weibull_ph_model(
    X: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    log_hr_prior: PriorDistribution,
    intervention_idx: int = 0,
    model_name: Optional[str] = None
) -> pm.Model:
    """
    Create a Weibull proportional hazards model using PyMC.
    
    The model assumes:
    h_i(t) = λk(λt)^(k-1) * exp(x_i^T β)
    
    where:
    - k > 0: shape parameter
    - λ > 0: scale parameter  
    - β: covariate coefficients (log-hazard ratios)
    
    Args:
        X: Covariate matrix (n_obs x n_covariates)
        time: Event/censoring times
        event: Event indicators (0/1)
        log_hr_prior: Prior distribution for the intervention log-HR
        intervention_idx: Index of the intervention variable in X (default: 0)
        model_name: Optional model name for identification
        
    Returns:
        pm.Model: PyMC model ready for sampling
    """
    if model_name is None:
        model_name = f"weibull_ph_{log_hr_prior.name}"
    
    logger.info(f"Creating Weibull PH model: {model_name}")
    logger.info(f"Data: {X.shape[0]} observations, {X.shape[1]} covariates")
    logger.info(f"Prior for log-HR: N({log_hr_prior.mu:.4f}, {log_hr_prior.sigma:.4f}^2)")
    
    with pm.Model(name=model_name) as model:
        # Priors for Weibull parameters
        # Shape parameter k (log-normal prior to ensure k > 0)
        log_k = pm.Normal("log_k", mu=0, sigma=1)
        k = pm.Deterministic("k", pt.exp(log_k))
        
        # Scale parameter λ (log-normal prior to ensure λ > 0)
        log_lambda = pm.Normal("log_lambda", mu=0, sigma=1)
        lam = pm.Deterministic("lambda", pt.exp(log_lambda))
        
        # Covariate coefficients (log-hazard ratios)
        n_covariates = X.shape[1]
        
        # Special prior for intervention effect (main interest)
        log_hr_intervention = pm.Normal(
            "log_hr_intervention",
            mu=log_hr_prior.mu,
            sigma=log_hr_prior.sigma
        )
        
        # Weakly informative priors for other covariates
        if n_covariates > 1:
            log_hr_other = pm.Normal(
                "log_hr_other",
                mu=0,
                sigma=2.5,
                shape=n_covariates - 1
            )
            
            # Combine intervention and other coefficients
            if intervention_idx == 0:
                beta = pt.concatenate([
                    log_hr_intervention.reshape(1),
                    log_hr_other
                ])
            else:
                # Insert intervention effect at the correct position
                beta_list = []
                for i in range(n_covariates):
                    if i == intervention_idx:
                        beta_list.append(log_hr_intervention)
                    elif i < intervention_idx:
                        beta_list.append(log_hr_other[i])
                    else:
                        beta_list.append(log_hr_other[i-1])
                beta = pt.stack(beta_list)
        else:
            # Only intervention effect
            beta = log_hr_intervention.reshape(1)
        
        # Linear predictor
        eta = pt.dot(X, beta)
        
        # Hazard ratio (for interpretation)
        hr_intervention = pm.Deterministic("hr_intervention", pt.exp(log_hr_intervention))
        
        # Weibull survival function and hazard
        # S(t) = exp(-λt)^k * exp(η))
        # f(t) = h(t) * S(t) where h(t) = k*λ*(λt)^(k-1) * exp(η)
        
        # For computational stability, work in log space
        log_hazard = (
            pt.log(k) + 
            pt.log(lam) + 
            (k - 1) * pt.log(lam * time) + 
            eta
        )
        
        log_survival = -(lam * time) ** k * pt.exp(eta)
        
        # Log-likelihood for survival data
        # For observed events: log(h(t)) + log(S(t))
        # For censored: log(S(t))
        ll_event = log_hazard + log_survival  # For event = 1
        ll_censor = log_survival              # For event = 0
        
        # Total log-likelihood
        ll = pt.sum(event * ll_event + (1 - event) * ll_censor)
        
        # Add to model as potential
        pm.Potential("likelihood", ll)
        
        # Store data for later use
        model.X = X
        model.time = time
        model.event = event
        model.intervention_idx = intervention_idx
        model.log_hr_prior = log_hr_prior
    
    logger.info(f"Model created successfully with {len(model.free_RVs)} free variables")
    return model


def sample_weibull_ph_model(
    model: pm.Model,
    random_seed: Optional[int] = None
) -> pm.InferenceData:
    """
    Sample from the Weibull PH model using NUTS.
    
    Args:
        model: PyMC model to sample from
        random_seed: Random seed for reproducibility
        
    Returns:
        pm.InferenceData: Posterior samples and diagnostics
    """
    if random_seed is None:
        random_seed = SEED
    
    logger.info(f"Starting MCMC sampling with {N_CHAINS} chains, {N_DRAWS} draws")
    logger.info(f"Warmup: {N_TUNE} draws, Target accept: {TARGET_ACCEPT}")
    
    with model:
        # Initialize sampler
        step = pm.NUTS(target_accept=TARGET_ACCEPT)
        
        # Sample
        trace = pm.sample(
            draws=N_DRAWS,
            tune=N_TUNE,
            chains=N_CHAINS,
            step=step,
            random_seed=random_seed,
            return_inferencedata=True,
            compute_convergence_checks=True
        )
    
    logger.info("MCMC sampling completed")
    
    # Log basic diagnostics
    summary = pm.summary(trace)
    rhat_max = summary['r_hat'].max()
    ess_min = summary['ess_bulk'].min()
    
    logger.info(f"Max R-hat: {rhat_max:.4f}")
    logger.info(f"Min ESS (bulk): {ess_min:.0f}")
    
    if rhat_max > 1.01:
        logger.warning(f"Some R-hat values exceed 1.01 (max: {rhat_max:.4f})")
    else:
        logger.info("All R-hat values ≤ 1.01 (good convergence)")
    
    return trace


def compute_log_likelihood(model: pm.Model, trace: pm.InferenceData) -> np.ndarray:
    """
    Compute pointwise log-likelihood for model comparison.
    
    Args:
        model: PyMC model
        trace: MCMC trace
        
    Returns:
        np.ndarray: Log-likelihood matrix (n_samples x n_observations)
    """
    logger.info("Computing pointwise log-likelihood")
    
    with model:
        # Extract posterior samples
        posterior = trace.posterior
        
        # Get parameter samples
        k_samples = posterior['k'].values.flatten()
        lambda_samples = posterior['lambda'].values.flatten()
        log_hr_intervention_samples = posterior['log_hr_intervention'].values.flatten()
        
        if 'log_hr_other' in posterior:
            log_hr_other_samples = posterior['log_hr_other'].values
            n_other = log_hr_other_samples.shape[-1]
            log_hr_other_samples = log_hr_other_samples.reshape(-1, n_other)
        else:
            log_hr_other_samples = None
        
        n_samples = len(k_samples)
        n_obs = len(model.time)
        
        # Compute log-likelihood for each sample and observation
        log_lik = np.zeros((n_samples, n_obs))
        
        for i in range(n_samples):
            k_i = k_samples[i]
            lambda_i = lambda_samples[i]
            log_hr_int_i = log_hr_intervention_samples[i]
            
            # Reconstruct beta vector
            if log_hr_other_samples is not None:
                if model.intervention_idx == 0:
                    beta_i = np.concatenate([[log_hr_int_i], log_hr_other_samples[i]])
                else:
                    beta_i = np.zeros(model.X.shape[1])
                    beta_i[model.intervention_idx] = log_hr_int_i
                    other_idx = 0
                    for j in range(len(beta_i)):
                        if j != model.intervention_idx:
                            beta_i[j] = log_hr_other_samples[i, other_idx]
                            other_idx += 1
            else:
                beta_i = np.array([log_hr_int_i])
            
            # Linear predictor
            eta_i = model.X @ beta_i
            
            # Log-likelihood for each observation
            log_hazard_i = (
                np.log(k_i) + 
                np.log(lambda_i) + 
                (k_i - 1) * np.log(lambda_i * model.time) + 
                eta_i
            )
            
            log_survival_i = -((lambda_i * model.time) ** k_i) * np.exp(eta_i)
            
            ll_event_i = log_hazard_i + log_survival_i
            ll_censor_i = log_survival_i
            
            log_lik[i, :] = model.event * ll_event_i + (1 - model.event) * ll_censor_i
    
    logger.info(f"Computed log-likelihood: {log_lik.shape}")
    return log_lik


if __name__ == "__main__":
    # Test model creation with dummy data
    try:
        from .priors import get_existing_priors
        
        # Create dummy data
        np.random.seed(SEED)
        n_obs = 100
        X = np.random.randn(n_obs, 2)
        time = np.random.exponential(2, n_obs)
        event = np.random.binomial(1, 0.3, n_obs)
        
        # Get a test prior
        priors = get_existing_priors()
        test_prior = priors["noninformative"]
        
        # Create and test model
        model = create_weibull_ph_model(X, time, event, test_prior)
        print(f"Model created successfully: {model.name}")
        print(f"Free variables: {[var.name for var in model.free_RVs]}")
        
        # Test sampling (very short run for testing)
        with model:
            prior_pred = pm.sample_prior_predictive(samples=10, random_seed=SEED)
            print("Prior predictive sampling successful")
        
    except Exception as e:
        print(f"Error testing model: {e}")
