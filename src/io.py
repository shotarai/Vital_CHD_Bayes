"""
Data input/output and preprocessing for VITAL-CHD Bayesian analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

from .config import VITAL_DATA_PATH, COLUMNS, COVARIATES

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_vital_data() -> pd.DataFrame:
    """
    Load the VITAL trial dataset.
    
    Returns:
        pd.DataFrame: Raw VITAL dataset
        
    Raises:
        FileNotFoundError: If the data file is not found
    """
    if not VITAL_DATA_PATH.exists():
        raise FileNotFoundError(f"VITAL data file not found at {VITAL_DATA_PATH}")
    
    logger.info(f"Loading VITAL data from {VITAL_DATA_PATH}")
    df = pd.read_csv(VITAL_DATA_PATH)
    logger.info(f"Loaded {len(df)} observations")
    
    return df


def check_required_columns(df: pd.DataFrame) -> None:
    """
    Check if all required columns are present in the dataset.
    
    Args:
        df: Input dataframe
        
    Raises:
        KeyError: If required columns are missing
    """
    required_cols = [COLUMNS["event"], COLUMNS["time"]] + COVARIATES
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise KeyError(f"Missing required columns: {missing_cols}")
    
    logger.info("All required columns present in dataset")


def preprocess_survival_data(df: pd.DataFrame, complete_case: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess VITAL data for survival analysis.
    
    Args:
        df: Raw VITAL dataset
        complete_case: If True, use complete case analysis (default)
        
    Returns:
        Tuple containing:
        - X: Covariate matrix (n_obs x n_covariates)
        - time: Event/censoring times
        - event: Event indicators (0/1)
    """
    # Check required columns
    check_required_columns(df)
    
    # Extract columns
    event_col = COLUMNS["event"]
    time_col = COLUMNS["time"]
    
    # Create working dataframe with required columns
    work_df = df[[event_col, time_col] + COVARIATES].copy()
    
    # Handle missing values
    if complete_case:
        n_before = len(work_df)
        work_df = work_df.dropna()
        n_after = len(work_df)
        if n_before != n_after:
            logger.warning(f"Dropped {n_before - n_after} observations with missing data "
                         f"({n_after}/{n_before} = {100*n_after/n_before:.1f}% remaining)")
    else:
        # Simple imputation if requested (though spec suggests complete case as default)
        logger.info("Using simple imputation for missing values")
        # Impute numeric variables with median
        for col in COVARIATES:
            if work_df[col].dtype in ['float64', 'int64']:
                work_df[col] = work_df[col].fillna(work_df[col].median())
            else:
                work_df[col] = work_df[col].fillna(work_df[col].mode()[0])
    
    # Extract event and time
    event = work_df[event_col].values.astype(int)
    time = work_df[time_col].values.astype(float)
    
    # Validate survival data
    if np.any(time <= 0):
        logger.warning("Found non-positive survival times, filtering them out")
        valid_mask = time > 0
        work_df = work_df[valid_mask]
        event = event[valid_mask]
        time = time[valid_mask]
    
    # Prepare covariate matrix
    X = work_df[COVARIATES].values.astype(float)
    
    # Standardize sex variable if needed (ensure 0/1 coding)
    if "sex" in COVARIATES:
        sex_idx = COVARIATES.index("sex")
        unique_sex = np.unique(X[:, sex_idx])
        if not np.array_equal(sorted(unique_sex), [0, 1]) and not np.array_equal(sorted(unique_sex), [1, 2]):
            logger.warning(f"Sex variable has unexpected values: {unique_sex}")
        
        # Convert to 0/1 if it's 1/2
        if np.array_equal(sorted(unique_sex), [1, 2]):
            X[:, sex_idx] = X[:, sex_idx] - 1
            logger.info("Converted sex coding from 1/2 to 0/1")
    
    # Log final dataset characteristics
    logger.info(f"Final dataset: {len(work_df)} observations")
    logger.info(f"Events: {np.sum(event)} ({100*np.mean(event):.1f}%)")
    logger.info(f"Median follow-up: {np.median(time):.2f} years")
    logger.info(f"Covariates: {COVARIATES}")
    
    # Log covariate summaries
    for i, covar in enumerate(COVARIATES):
        logger.info(f"{covar}: mean={np.mean(X[:, i]):.2f}, std={np.std(X[:, i]):.2f}")
    
    return X, time, event


def get_processed_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convenience function to load and preprocess VITAL data in one step.
    
    Returns:
        Tuple containing:
        - X: Covariate matrix (n_obs x n_covariates)
        - time: Event/censoring times
        - event: Event indicators (0/1)
    """
    df = load_vital_data()
    return preprocess_survival_data(df)


if __name__ == "__main__":
    # Test the data loading and preprocessing
    try:
        X, time, event = get_processed_data()
        print(f"Successfully processed data: {X.shape[0]} observations, {X.shape[1]} covariates")
        print(f"Event rate: {np.mean(event):.3f}")
        print(f"Mean follow-up: {np.mean(time):.2f} years")
    except Exception as e:
        print(f"Error processing data: {e}")
