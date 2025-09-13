"""
Configuration settings for VITAL-CHD Bayesian re-analysis experiment.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

# Ensure directories exist
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Data file
VITAL_DATA_PATH = DATA_DIR / "VITAL_trial_NEJM_2022.csv"

# Random seed for reproducibility
SEED = 2025

# MCMC settings (single chain to avoid multiprocessing issues)
N_CHAINS = 3
N_DRAWS = 4000
N_TUNE = 2000  # warmup draws
TARGET_ACCEPT = 0.95  # Higher target accept for more stable sampling

# Convergence criteria
MAX_RHAT = 1.01

# API settings for LLM prior generation
API_KEY = os.getenv("API_KEY")
if not API_KEY:
    raise ValueError("API_KEY not found in environment variables. Please set it in .env file.")

# LLM models to use
LLM_MODELS = [
    "llama-3.3-70b-instruct",
    "medgemma-27b-it"
]

# Temperature for LLM generation (fixed for reproducibility)
LLM_TEMPERATURE = 0

# Column names in the VITAL dataset
COLUMNS = {
    "event": "totchd",           # Total CHD event indicator (0/1)
    "time": "chdyears",          # Time to CHD event or censoring (years)
    "age": "ageyr",              # Age in years
    "sex": "sex",                # Sex (binary)
    "intervention": "fishoilactive" # Fish oil (omega-3) intervention (1=treatment, 0=control)
}

# Covariates to include - intervention MUST be first for HR interpretation
COVARIATES = ["fishoilactive", "ageyr", "sex", "vitdactive"]

# Prior distribution specifications (existing 5 types)
EXISTING_PRIORS = {
    "noninformative": {
        "mu": 0.0,
        "sigma": 10.0,
        "description": "Minimally informative prior allowing data to dominate"
    },
    "primary_informed": {
        "mu": -0.072,
        "sigma": 0.037,
        "description": "Primary prevention CHD effect from meta-analysis"
    },
    "weakly": {
        "mu": -0.072,
        "sigma": 0.055,  # 1.5 * primary
        "description": "Same center as primary but wider uncertainty"
    },
    "strong": {
        "mu": -0.072,
        "sigma": 0.018,  # 0.5 * primary
        "description": "Same center as primary but narrower uncertainty"
    },
    "skeptical": {
        "mu": 0.0,
        "sigma": 0.121,
        "description": "Skeptical of substantial effect (effect size <5%)"
    }
}

# LLM prompt template for prior elicitation
LLM_PROMPT = """You are assisting a Bayesian survival analysis for the VITAL trial context: a primary-prevention population of U.S. adults without prior CVD, randomized to vitamin D3 (2000 IU/day), omega-3 fatty acids (1 g/day), both, or placebo, with a median intervention of ~5.3 years. 
Our current analysis focuses on the omega-3 fatty acids intervention effect on total coronary heart disease (CHD) as the primary endpoint, modeled with a Bayesian Weibull proportional hazards model. The coefficient of interest is the log-hazard ratio (log-HR) for the omega-3 intervention effect on total CHD.

Task:
Propose a Normal prior distribution for the log-HR (omega-3 intervention vs. control) reflecting realistic prior clinical knowledge for CHD in primary prevention within this factorial-trial setting.

Important:
- Output ONLY the prior parameters as JSON in this schema:
  {"mu": <float>, "sigma": <float>}
- Do not include explanations or units; just the numeric parameters."""
