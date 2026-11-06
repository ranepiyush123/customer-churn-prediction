"""Generate reproducible synthetic customer records for demonstration."""
import numpy as np
import pandas as pd

FEATURES = ["tenure_months", "monthly_charges", "support_tickets", "contract_type", "autopay"]

def make_dataset(n=1200, seed=42):
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 73, n)
    charges = rng.uniform(20, 120, n).round(2)
    tickets = rng.poisson(1.5, n)
    contract = rng.choice(["monthly", "annual"], n, p=[0.65, 0.35])
    autopay = rng.choice(["yes", "no"], n, p=[0.6, 0.4])
    logit = -1.1 - 0.035 * tenure + 0.022 * (charges - 65) + 0.36 * tickets + 1.0 * (contract == "monthly") + 0.5 * (autopay == "no")
    prob = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, prob)
    return pd.DataFrame({"tenure_months": tenure, "monthly_charges": charges, "support_tickets": tickets, "contract_type": contract, "autopay": autopay, "churn": churn})

if __name__ == "__main__":
    from pathlib import Path
    path = Path("data/synthetic_customers.csv")
    path.parent.mkdir(exist_ok=True)
    make_dataset().to_csv(path, index=False)
    print(f"Created {path}")
