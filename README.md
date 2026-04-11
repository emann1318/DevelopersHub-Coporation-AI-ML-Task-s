# Internship Tasks — Eman Faris (DHC-6221)

This repository contains a Google Colab notebook (`DHC-6221_Eman_Faris.ipynb`) covering six machine learning and AI internship tasks. Each task explores a distinct domain, from classic data exploration to LLM-powered chatbots.

---

## Table of Contents

1. [Task 1 — Exploring & Visualizing the Iris Dataset](#task-1--exploring--visualizing-the-iris-dataset)
2. [Task 2 — Stock Price Prediction](#task-2--stock-price-prediction)
3. [Task 3 — Heart Disease Prediction](#task-3--heart-disease-prediction)
4. [Task 4 — Health Chatbot (Prompt Engineering)](#task-4--health-chatbot-prompt-engineering)
5. [Task 5 — Mental Health Chatbot (Fine-Tuned)](#task-5--mental-health-chatbot-fine-tuned)
6. [Task 6 — House Price Prediction](#task-6--house-price-prediction)

---

## Task 1 — Exploring & Visualizing the Iris Dataset

**Goal:** Perform exploratory data analysis (EDA) on the classic Iris dataset and derive species-level insights through visualizations.

**Steps:**
- Loaded the Iris dataset via `seaborn`.
- Examined dataset shape, column names, data types, and summary statistics.
- Produced a scatter plot (sepal length vs. petal length, coloured by species), histograms, and box plots.
- Detected outliers using per-species box plots across all four numerical features.
- Calculated and compared median values of each feature across the three species.

**Key Findings:**
- *Iris-setosa* has noticeably smaller petal dimensions than the other two species.
- *Iris-virginica* has the largest petal lengths and widths on average.
- Petal length and petal width are the most discriminative features between species.

**Libraries:** `pandas`, `seaborn`, `matplotlib`

---

## Task 2 — Stock Price Prediction

**Goal:** Predict the next-day closing price of Apple (AAPL) stock using a Linear Regression model.

**Steps:**
- Downloaded historical AAPL data (2020–2023) with `yfinance`.
- Engineered a `Target` column by shifting the `Close` price one day forward.
- Split features (`Open`, `High`, `Low`, `Volume`) and target into training and test sets (no shuffling to preserve temporal order).
- Trained a `LinearRegression` model and plotted actual vs. predicted prices.

**Key Findings:**
- The model captures the general price trend but struggles during high-volatility periods.
- Large gaps between actual and predicted lines appear around sharp market movements, highlighting the limitations of a simple linear model for financial time series.

**Libraries:** `pandas`, `yfinance`, `scikit-learn`, `matplotlib`

---

## Task 3 — Heart Disease Prediction

**Goal:** Build a binary classifier to predict the presence of heart disease and evaluate it with medical-context metrics.

**Steps:**
- Loaded the `heart.csv` dataset (1,025 entries, 14 features, no missing values).
- Visualised class balance with a count plot and explored feature correlations with a heatmap.
- Trained a `LogisticRegression` classifier on all features.
- Evaluated performance using accuracy score and confusion matrix.

**Results:**

| Metric | Value |
|---|---|
| Accuracy | ~84.9% |
| Recall (Sensitivity) | ~91.1% |
| Precision | ~81.6% |

**Key Findings:**
- High recall (91.1%) is the most important metric here — missing a true positive in a medical context is more dangerous than generating a false alarm.
- Features such as `cp` (chest pain type), `thalach` (max heart rate), and `slope` showed strong positive correlation with the target.
- A convergence warning was observed; scaling features or increasing `max_iter` could further improve results.

**Libraries:** `pandas`, `seaborn`, `scikit-learn`

---

## Task 4 — Health Chatbot (Prompt Engineering)

**Goal:** Demonstrate how prompt engineering controls the behaviour and safety of an LLM-based medical assistant.

**Steps:**
- Composed a role-setting prompt: *"Act like a helpful medical assistant. Answer safely: What causes sore throat?"*
- Accepted user input and implemented a hard-coded safety rule: if the word `"emergency"` appears in the query, the chatbot immediately redirects the user to seek professional medical help.

**Key Insights:**
1. **Prompt controls tone and persona** — explicitly stating a persona shapes the model's style and makes interactions more trustworthy.
2. **Safety rules prevent harmful advice** — a simple keyword check provides a critical safeguard in sensitive healthcare contexts.
3. **Good prompts outperform complex models** — a well-structured, context-setting prompt can yield more accurate and safe answers than relying solely on a large, unguided model.

**Libraries:** `openai`

---

## Task 5 — Mental Health Chatbot (Fine-Tuned)

**Goal:** Fine-tune a transformer model on an emotion dataset to power an empathetic mental health chatbot.

**Steps:**
- Uploaded and loaded `emotion-emotion_69k.csv` (~69,000 labelled emotion entries).
- Inspected dataset structure (shape, columns, types, statistics).
- Set up a `Trainer` from the Hugging Face `transformers` library for fine-tuning.
- Defined sample input/output pairs (e.g., `"I feel anxious"` → `"I'm sorry you're feeling this way..."`).

**Libraries:** `pandas`, `transformers` (Hugging Face)

---

## Task 6 — House Price Prediction

**Goal:** Predict residential house prices using a Linear Regression model after thorough preprocessing.

**Steps:**
- Loaded `Housing.csv` containing numerical and categorical features.
- Handled missing values: numerical columns filled with column mean; categorical columns filled with column mode.
- Encoded binary categorical features (`yes`/`no`) as `1`/`0`.
- Applied one-hot encoding to multi-category columns (e.g., `furnishingstatus`) using `pd.get_dummies`.
- Split data 80/20 (train/test) and trained a `LinearRegression` model.
- Evaluated with MAE and RMSE; plotted Actual vs. Predicted prices.

**Results:**

| Metric | Value |
|---|---|
| MAE | ~970,043 |
| RMSE | ~1,324,507 |

**Key Findings:**
- The model captures the general upward trend but shows significant scatter, indicating limited accuracy for this dataset.
- High error values suggest the linear model is too simple for this problem; non-linear models would likely perform better.

**Suggested Improvements:**
- Feature engineering (interaction terms, polynomial features).
- Advanced models: Ridge, Lasso, Random Forest, XGBoost, or Neural Networks.
- Feature scaling and outlier treatment.
- Cross-validation for more robust evaluation.

**Libraries:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
yfinance
openai
transformers
```

Install all dependencies with:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn yfinance openai transformers
```

> **Note:** This notebook was originally developed in Google Colab. File upload cells (`from google.colab import files`) will need to be replaced with local file paths when running outside of Colab.

---

## Author

**Eman Faris** — DHC-6221
