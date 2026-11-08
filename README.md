# Online Shopper Purchase Propensity Model

Predicts which website sessions end in a purchase, and shows how a marketing team could use those predictions to focus on high-intent visitors.

## Hypothesis

If a marketing team can only follow up with or retarget a limited share of visitors, which sessions should it prioritize? This project builds a purchase propensity model to answer that.

## Data

[UCI Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset): about 12,000 e-commerce sessions with page-visit counts and durations, bounce and exit rates, page value, month, visitor type, traffic type, and a `Revenue` label showing whether the session ended in a purchase.

## Approach

1.  **Cleaning:** checked for missing values, removed duplicate rows, converted boolean fields.
2.  **Exploration:** compared conversion rates by visitor type and month.
3.  **Modeling:** trained logistic regression and random forest classifiers using a stratified 80/20 train/test split and 5-fold cross-validation. Class weights handle the imbalance, since only a minority of sessions convert.
4.  **Evaluation:** ROC AUC, precision-recall AUC, and a confusion matrix. Accuracy alone is misleading here because most sessions do not convert.
5.  **Marketing application:** a cumulative gains analysis showing how many buyers a team would reach by targeting only the highest-scoring sessions.

## Results

- Overall conversion rate: **15.6%** (12,205 sessions after removing 125 duplicates)
- Best model: **Random Forest**, test ROC AUC **0.93** (logistic regression: 0.91)
- At a 0.5 threshold, the Random Forest identified **74%** of buyers with **64%** precision
- Top drivers of purchase: **PageValues, ExitRates, time on product pages, product pages viewed, BounceRates**
- Targeting the top 20% of sessions by score reaches **77%** of buyers, a **3.9x** lift over random targeting

## Takeaways

- Sessions with high page value, low exit rates, and more time on product pages are the strongest purchase signals, so they are the best candidates to follow up.
- New visitors converted at 24.9% versus 14.1% for returning visitors, and November converted best, so acquisition spend and promotions pay off most in the fall.
- A team with a limited budget could focus on the top-scoring 20% of sessions and still reach about three quarters of likely buyers.

## Limitations

- Session-level data only, with no customer history, so this predicts purchase intent and not customer lifetime value.
- The data comes from a single site and one year, and results may not generalize.
- Predictions are correlational and do not show which marketing actions cause purchases.

## Tools

Python, pandas, scikit-learn, matplotlib
