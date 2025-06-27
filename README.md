# Credit Risk Probability Model for Alternative Data

# Task 1 - Credit Scoring Business Understanding

### 1. Basel II Accord's Influence on Model Requirements

Basel II's Pillar 1 mandates minimum capital requirements tied to **risk-weighted assets** (RWAs), requiring banks to maintain capital reserves ≥8% of RWAs. This directly impacts our model design:

- **Interpretability** is essential because regulators must validate how risk weights are calculated and ensure capital adequacy reflects true risk exposure.
- **Documentation** must transparently justify risk quantifications to satisfy supervisory review (Pillar 2) and market discipline (Pillar 3).  
  Without these, the model fails regulatory scrutiny and could misallocate capital, violating Basel II compliance.

### 2. Proxy Variable Necessity and Risks

**Why a proxy is needed:**  
Direct default labels are absent since eCommerce users haven't undergone traditional lending. We use **RFM patterns** (Recency, Frequency, Monetary) as a behavioral proxy for credit risk.

**Business risks include:**

- **Proxy inaccuracy:** If RFM poorly correlates with true default risk, loans may be mispriced (e.g., underestimating risk increases defaults).
- **Discrimination risk:** Proxies like purchase behavior could correlate with protected attributes (e.g., ZIP codes with race), violating fair lending laws (ECOA).
- **Model drift:** Behavioral shifts (e.g., pandemic spending) may decouple the proxy from actual risk over time.

### 3. Model Trade-offs in Regulated Finance

| **Model Type**          | **Advantages**                                                                                  | **Regulatory Drawbacks**                                                                    |
| ----------------------- | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| **Logistic Regression** | -  Full interpretability (e.g., WoE bins)<br>-  Audit-friendly outputs<br>-  Simpler validation | -  Lower predictive power<br>-  May miss non-linear patterns                                |
| **Gradient Boosting**   | -  Higher accuracy<br>-  Captures complex interactions                                          | -  Black-box decisions hinder explanation<br>-  Harder to validate for capital calculations |

**Resolution:** Basel II favors **interpretability** unless complexity demonstrably improves risk sensitivity without compromising transparency. Hybrid approaches (e.g., SHAP explanations for boosting) may balance this.

## Task 2: Exploratory Data Analysis (EDA)

### Key Findings

1. **Severe target imbalance**  
   Only 0.2% of transactions are fraudulent (193 cases), requiring specialized handling during modeling.

2. **Extreme transaction values**  
   `Amount` and `Value` show:

   - Negative values (down to -1M UGX) suggesting refunds
   - High positive skew (max 9.88M UGX) requiring transformation
   - High standard deviation (123k) indicating volatility

3. **Dominant categorical features**  
   Key categories dominate distributions:

   - 47.5% transactions in `financial_services`
   - 39.9% through `ProviderId_4`
   - 59.5% via `ChannelId_3`

4. **Constant features**  
   `CurrencyCode` (always UGX) and `CountryCode` (always 256) provide no predictive value.

5. **Temporal feature potential**  
   Transaction timestamps show high uniqueness, enabling time-based feature engineering.

[View full EDA notebook](notebooks/1.0-eda.ipynb)
