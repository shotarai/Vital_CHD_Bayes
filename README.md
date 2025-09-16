# VITAL-CHD Bayesian Re-analysis

This project implements Bayesian survival analysis using the VITAL trial CHD (coronary heart disease) data. It compares **LLM-generated priors** with traditional priors, focusing on their stability and robustness under reduced data conditions.

## Project Overview

### Objectives
- Analyze **total CHD (`totchd`)** as the primary outcome of the **VITAL trial**  
- Apply a **Weibull proportional hazards (PH)** Bayesian survival model  
- Compare **LLM-generated priors** (Llama 3.3 70B / MedGemma 27B) with **five traditional priors**  
- Evaluate **inference** (HR, P(HR<1)) and **predictive robustness** (C-index, IBS) across reduced training sizes  

### Data
- **Input file**: `data/VITAL_trial_NEJM_2022.csv`
- **Primary outcome**:  
  - Event indicator: `totchd` (total CHD: 0/1)  
  - Follow-up time: `chdyears` (years to CHD event or censoring)  
- **Covariates**: `ageyr`, `sex` (same setting as Hamaya et al.)  

### Methods
- **Model**: Bayesian Weibull proportional hazards  
- **Implementation**: Python + PyMC (HMC/NUTS)  
- **Settings**: 3 chains, 4,000 draws (2,000 warmup), target_accept ≥ 0.9  
- **Evaluation**: Inference (HR, P(HR<1)) + predictive robustness under small-n  

## Environment Setup

### Requirements
- Python 3.11+
- [rye](https://rye-up.com/) package manager

### Installation

```bash
cd vital_chd_bayes
rye sync
````

### Environment Variables

Configure API key in `.env`:

```bash
API_KEY=your_actual_api_key_here
```

## Project Structure

```
vital_chd_bayes/
├─ data/
│  └─ VITAL_trial_NEJM_2022.csv
├─ src/
│  ├─ config.py
│  ├─ io.py
│  ├─ priors.py
│  ├─ model_weibull_ph.py
│  ├─ inference.py
│  ├─ predictive.py
│  ├─ reporting.py
│  └─ run_experiments.py
├─ results/
│  ├─ tables/
│  └─ figures/
├─ .env
├─ README.md
└─ pyproject.toml
```

## Execution

### Full Experiment

```bash
rye run python -m src.run_experiments
```

### Module Tests

```bash
rye run python -m src.io
rye run python -m src.priors
rye run python -m src.model_weibull_ph
```

## Prior Specifications

### Five Existing Priors (log-HR \~ Normal(μ, σ²))

| Name                 | μ      | σ     | Description                  |
| -------------------- | ------ | ----- | ---------------------------- |
| **Noninformative**   | 0.0    | 10.0  | Data-driven, flat prior      |
| **Primary informed** | -0.072 | 0.037 | Meta-analysis based          |
| **Weakly**           | -0.072 | 0.055 | 1.5 × wider uncertainty      |
| **Strong**           | -0.072 | 0.018 | 0.5 × tighter uncertainty    |
| **Skeptical**        | 0.0    | 0.121 | Effect < 5%, skeptical prior |

### LLM-generated Priors

* **Models**:

  * `llama-3.3-70b-instruct` (general-purpose)
  * `medgemma-27b-it` (medical domain)
* **Temperature**: 0 (deterministic output)
* **Format**: JSON `{"mu": <float>, "sigma": <float>}`

## Output Files

### Tables (`results/tables/`)

* `inference_summary.csv`: Inference results
* `predictive_summary.csv`: Predictive robustness results
* `priors_summary.csv`: Priors used
* `summary_report.txt`: Integrated report

### Figures (`results/figures/`)

* `fig_HR_by_prior.png`: HR comparison by prior
* `fig_learning_curves.png`: C-index and IBS vs. training size
* `fig_prior_posterior_comparison.png`: Prior vs posterior

## Evaluation Metrics

### Inference

* **HR**: Posterior hazard ratio (exp(log-HR))
* **95% CrI**: Credible interval
* **P(HR < 1)**: Probability of protective effect
* **Convergence**: R-hat ≤ 1.01, ESS

### Predictive Robustness (Small-n Sensitivity)

* **C-index (Uno’s C)**: Discrimination under censoring
* **IBS (Integrated Brier Score)**: Calibration across follow-up
* **Learning curves**: Performance at training fractions (20%, 40%, 60%, 80%, 100%)
* **Robustness criterion**: Priors that maintain stable C-index and IBS even with reduced n

## Evaluation Criteria

An LLM prior is considered **useful** if it demonstrates:

1. **Stable HR estimates** with narrower CrIs
2. **Higher probability of HR < 1** (inference strength)
3. **Less degradation** in C-index and IBS at reduced sample sizes

## Troubleshooting

### Common Issues

1. **Missing API key** → Configure `.env`
2. **Convergence warnings** → Adjust MCMC settings
3. **Memory errors** → Reduce draws or increase memory

### Dependency Issues

```bash
rye sync --force
rye add package_name
```

## Development Notes

### Code Quality

```bash
rye run black src/
rye run isort src/
rye run ruff check src/
```

### Logs

Execution logs saved in `results/tables/experiment.log`

## References

* Hamaya et al. Bayesian analysis of VITAL trial
* PyMC documentation
* ArviZ model comparison documentation
* Uno et al. (C-index under censoring)
* Graf et al., Gerds & Schumacher (IBS consistency)

## License

This project is for research purposes only.

## Contact

Please open an issue for questions or problems.
