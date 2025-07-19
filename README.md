<<<<<<< HEAD
# 🧮 Aave V2 Wallet Credit Scoring System

This project analyzes user transaction data from the Aave V2 DeFi protocol and assigns each wallet a *credit score between 0 and 1000*, based on responsible or risky behavior. It is a rule-based scoring system that does not use machine learning, but rather mimics financial risk modeling logic using transaction patterns.

---

## 🚀 Problem Statement

Given a raw transaction dataset from Aave V2, each wallet must be scored based on historical DeFi behavior. The system must:

- Efficiently process a large JSON file.
- Engineer meaningful wallet-level features.
- Assign a score (0–1000) per wallet using a rule-based algorithm.
- Output scores in CSV + chart.
- Generate a markdown report with results.

---

## ⚙ Technologies Used

- Python 3
- pandas, numpy, matplotlib, shap
- sklearn for regression & metrics

---

## 📦 Project Structure

```
aave_credit_score/
├── main.py
├── user_transactions.json
├── wallet_scores.csv
├── score_distribution.png
├── shap_summary.png
├── analysis.md
├── README.md
```

---

## 🧠 Methodology

Each wallet’s credit score is calculated from features like:

- Repayment ratio
- Borrow/repay behavior
- Liquidation count
- Wallet age & activity
- Protocol diversity

The algorithm uses business logic + minmax normalization.

---

## 📊 Scoring Rules

| Behavior Pattern                  | Score Impact |
|----------------------------------|--------------|
| High deposit frequency           | +200         |
| Regular repayments               | +150         |
| Full repayment of loans          | +200         |
| Long-term wallet usage           | +100         |
| High transaction volume          | +100         |
| Only borrows, no deposits        | -100         |
| No repayments at all             | -300         |
| Liquidation events               | -500 each    |

---

## 📈 Sample Results (from model)

- Total Wallets Scored: {total_wallets}
- Average Score: {avg:.1f}
- Median Score: {median:.1f}
- High Scorers (900+): {high_scorers:.2f}%
- Low Scorers (<100): {low_scorers:.2f}%
- MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}

---

## 📥 How to Run

1. Place your `user_transactions.json` file
2. Run: `python main.py`
3. Outputs will be saved as `.csv`, `.md`, and `.png`

---

## ✅ Notes

- No internet connection required
- Transparent, interpretable results
- Scalable to 100K+ transactions

=======
# defi-credit-scoring
A DeFi wallet credit scoring system using rule-based modeling
>>>>>>> bf57a7da7f2f3303b10e3bf686072898a74d5ffb
