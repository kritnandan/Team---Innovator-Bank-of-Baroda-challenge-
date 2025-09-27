# CAI-Powered User & Entity Behaviour Analytics (UEBA) for Fraud Detection
**Team:** Innovator 

---

## Problem Statment Overview
This presents a comprehensive, multi-layered solution for fraud detection using a **User and Entity Behavior Analytics (UEBA)** framework. Instead of relying on a single model, our prototype integrates **supervised, unsupervised, relational, and sequential modeling techniques** to create a robust and adaptive system.

The core of our solution is a hybrid intelligence pipeline that:

- Analyzes individual transaction features.
- Learns patterns of "normal" behavior to detect anomalies.
- Models the complex relationships between users, devices, and merchants.
- Understands the temporal sequence of actions to spot behavioral shifts.
- Makes adaptive, real-time decisions that balance security with customer experience.

---

## Important Note on Dataset
No real-world dataset was provided for this Challenge. To build a fully functional end-to-end prototype, we developed a **sophisticated synthetic data generator**.

### Key Points:
- **Class Imbalance:** Fraudulent transactions represent a small fraction (~2.88%) of the total, mimicking real-world scenarios. This can impact traditional accuracy metrics.
- **Performance:** Model performance on synthetic data is for demonstration purposes. Higher precision and accuracy are expected on actual historical data.

### Synthetic Data Generator Features
**Entities:**
- **Users:** With metadata like age, region, and risk scores.
- **Accounts:** Each user has at least one checking account.
- **Merchants:** Categories include retail, utilities, food, and travel.
- **Devices & IPs:** Simulating user access points.

**Realistic Transactions:**
- Monthly salary deposits.
- Recurring bill payments.
- Standard merchant purchases and P2P transfers.

**Injected Fraud Patterns:**
- **Account Takeover (ATO):** Unfamiliar device/IP followed by rapid, high-value transfers.
- **Micro-Transfers:** A burst of small transactions typical of bot activity.
- **Round-Tripping:** Rapid send-and-return transfers to launder funds.
- **Credential Stuffing Tests:** Series of low-value transactions to test stolen credentials.

---

## Solution Architecture: A Multi-Layered Approach
Our solution is a **five-layer pipeline**, where each layer provides a unique signal, culminating in an adaptive, context-aware decision.

### Layer 1: Baseline Supervised Model (XGBoost)
- Trained an XGBoost classifier on engineered transaction features.
- **Features Engineered:** `account_age_days`, `tx_hour`, `is_new_beneficiary`, `last_tx_interval_seconds`, rolling transaction counts/sums.
- **Imbalance Handling:** SMOTE-ENN to balance the training set.
- **Explainability:** SHAP identifies key features driving each prediction.

### Layer 2: Unsupervised Anomaly Detection (VAE)
- Variational Autoencoder trained on **normal transactions only**.
- Outputs **unsupervised_anomaly_score** based on reconstruction error.
- Detects novel or unseen fraud patterns.

### Layer 3: Relational Modeling (GNN)
- Constructed a **heterogeneous graph** with nodes and edges representing accounts, devices, IPs, and merchant interactions.
- Trained a **Graph Neural Network (SAGEConv)** to produce dense embeddings capturing relational context.

### Layer 4: Sequential Behavior Modeling (Behavior Transformer)
- Feeds **GNN embeddings** into a Transformer to learn **temporal dependencies**.
- Sequences of last 15 transaction embeddings per account.
- Outputs **supervised_fraud_prob** reflecting likelihood of fraud based on behavioral trajectory.

### Layer 5: Adaptive Decisioning (Contextual Bandit)
- **LinUCB contextual bandit** chooses the optimal business action:
  - **Approve:** Low-risk transaction.
  - **Challenge:** Medium-risk; requires two-factor authentication (2FA).
  - **Block:** High-risk transaction.
- Context vector includes: transaction amount, Transformer probability, VAE anomaly score.
- Reward function models **real-world outcomes**, balancing fraud prevention and user experience.

---

## 🔍 SHAP Analysis
To ensure interpretability, we applied **SHAP (SHapley Additive exPlanations)** on the supervised XGBoost model.  

<p align="center">
  <img src="078d87c1-fcd0-4bac-ad96-e8a5a8490857.png" alt="SHAP Feature Importance" width="500"/>
</p>

### Key Insights:
- **Risk Score** dominates predictions, indicating it encodes strong prior knowledge about fraud risk.  
- **Age**, **recent transaction counts (tx_count_7d)**, and **account_age_days** are also highly predictive, reflecting behavioral patterns.  
- **Rolling and temporal features** (e.g., `rolling_tx_sum_5`, `tx_hour`, `time_since_last_tx_seconds`) help capture abnormal spikes or unusual timing of transactions.  
- **New Beneficiaries & Ratios** (e.g., `is_new_beneficiary`, `amount_to_avg_ratio`) highlight deviation from user’s baseline habits.  

While `risk_score` is extremely predictive, over-reliance on it may introduce bias or overshadow behavioral signals. Combining static, behavioral, and relational features ensures a **balanced, fair, and resilient fraud detection system**.  

---

## How to Run the Prototype
The solution is contained in `Team_Innovator_Prototype_Solution.ipynb`.

### Dependencies
Install required libraries before running:
```bash
faker pandas numpy tqdm scikit-learn imblearn xgboost shap torch torch-geometric contextualbandits
