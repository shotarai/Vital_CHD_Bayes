"""
Prior distribution specifications for VITAL-CHD Bayesian analysis.
"""

import json
import httpx
import logging
from typing import Dict, Tuple, Optional, Any
from dataclasses import dataclass
import orjson

from config import (
    EXISTING_PRIORS, 
    LLM_MODELS, 
    LLM_TEMPERATURE, 
    LLM_PROMPT, 
    TABLES_DIR,
    API_KEY
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PriorDistribution:
    """Class to represent a prior distribution for log-HR."""
    name: str
    mu: float
    sigma: float
    description: str
    source: str = "existing"  # "existing" or "llm"


def get_existing_priors() -> Dict[str, PriorDistribution]:
    """
    Get the existing 5 prior distributions as specified.
    
    Returns:
        Dict[str, PriorDistribution]: Dictionary of existing prior distributions
    """
    priors = {}
    
    for name, params in EXISTING_PRIORS.items():
        priors[name] = PriorDistribution(
            name=name,
            mu=params["mu"],
            sigma=params["sigma"],
            description=params["description"],
            source="existing"
        )
    
    logger.info(f"Loaded {len(priors)} existing prior distributions")
    return priors


def query_llm_prior(model_name: str, timeout: int = 30) -> Optional[Tuple[float, float]]:
    """
    Query an LLM to get a prior distribution for log-HR.
    
    Args:
        model_name: Name of the LLM model to query
        timeout: Request timeout in seconds
        
    Returns:
        Optional[Tuple[float, float]]: (mu, sigma) if successful, None if failed
    """
    if not API_KEY:
        logger.error("API_KEY not set. Cannot query LLM.")
        return None
    
    logger.info(f"Querying {model_name} for prior distribution")
    
    # Prepare the request payload
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": LLM_PROMPT
            }
        ],
        "temperature": LLM_TEMPERATURE,
        "max_tokens": 100,  # We only need a small JSON response
        "response_format": {"type": "json_object"}  # Request JSON format
    }
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    try:
        with httpx.Client(timeout=timeout) as client:
            response = client.post(
                "https://chat-ai.academiccloud.de/v1/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # Parse the JSON response
            try:
                prior_params = orjson.loads(content)
                mu = float(prior_params["mu"])
                sigma = float(prior_params["sigma"])
                
                logger.info(f"Successfully obtained prior from {model_name}: mu={mu:.4f}, sigma={sigma:.4f}")
                return mu, sigma
                
            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                logger.error(f"Failed to parse response from {model_name}: {e}")
                logger.error(f"Response content: {content}")
                return None
                
    except httpx.RequestError as e:
        logger.error(f"Request failed for {model_name}: {e}")
        return None
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error for {model_name}: {e.response.status_code} - {e.response.text}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error querying {model_name}: {e}")
        return None


def get_llm_priors() -> Dict[str, PriorDistribution]:
    """
    Get prior distributions from LLM models.
    
    Returns:
        Dict[str, PriorDistribution]: Dictionary of LLM-generated prior distributions
    """
    llm_priors = {}
    
    for model_name in LLM_MODELS:
        result = query_llm_prior(model_name)
        
        if result is not None:
            mu, sigma = result
            prior_name = f"llm_{model_name.replace('-', '_')}"
            
            llm_priors[prior_name] = PriorDistribution(
                name=prior_name,
                mu=mu,
                sigma=sigma,
                description=f"LLM-generated prior from {model_name}",
                source="llm"
            )
        else:
            logger.warning(f"Failed to obtain prior from {model_name}")
    
    logger.info(f"Successfully obtained {len(llm_priors)} LLM prior distributions")
    return llm_priors


def get_all_priors() -> Dict[str, PriorDistribution]:
    """
    Get all prior distributions (existing + LLM-generated).
    
    Returns:
        Dict[str, PriorDistribution]: Dictionary of all prior distributions
    """
    logger.info("Collecting all prior distributions")
    
    # Get existing priors
    existing_priors = get_existing_priors()
    
    # Get LLM priors
    llm_priors = get_llm_priors()
    
    # Combine all priors
    all_priors = {**existing_priors, **llm_priors}
    
    logger.info(f"Total prior distributions: {len(all_priors)}")
    
    # Log summary of all priors
    for name, prior in all_priors.items():
        logger.info(f"{name}: mu={prior.mu:.4f}, sigma={prior.sigma:.4f} ({prior.source})")
    
    return all_priors


def save_priors_summary(priors: Dict[str, PriorDistribution], output_path: str) -> None:
    """
    Save a summary of prior distributions to a file.
    
    Args:
        priors: Dictionary of prior distributions
        output_path: Path to save the summary
    """
    import pandas as pd
    
    data = []
    for name, prior in priors.items():
        data.append({
            "prior_name": name,
            "mu": prior.mu,
            "sigma": prior.sigma,
            "description": prior.description,
            "source": prior.source
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved prior summary to {output_path}")


if __name__ == "__main__":
    # Test the prior generation functions
    try:
        # Test existing priors
        existing = get_existing_priors()
        print(f"Existing priors: {len(existing)}")
        
        # Test LLM priors (this will fail without proper API setup)
        print("Testing LLM prior generation...")
        llm = get_llm_priors()
        print(f"LLM priors: {len(llm)}")
        
        # Get all priors
        all_priors = get_all_priors()
        print(f"Total priors: {len(all_priors)}")
        
        # Print summary
        for name, prior in all_priors.items():
            print(f"{name}: N({prior.mu:.4f}, {prior.sigma:.4f}^2) - {prior.description[:50]}...")
            
    except Exception as e:
        print(f"Error testing priors: {e}")
