# Short-Term Dengue Forecasting with LSTM

This repository implements a deep learning pipeline for short-term epidemiological forecasting using Long Short-Term Memory (LSTM) neural networks with uncertainty estimation through Monte Carlo Dropout.

The model was designed to forecast future dengue incidence (or other epidemiological indicators) using historical case counts and auxiliary variables such as climate.

---

# Overview

The pipeline includes:

- Data normalization and preprocessing
- Sliding-window time series construction
- Multi-step forecasting
- LSTM-based neural network architecture
- Monte Carlo Dropout uncertainty estimation
- Cross-validation training
- Forecast export with prediction intervals

The framework supports both:

- retrospective experiments (train/test split)
- operational forecasting using a fixed training cutoff date

---

# Model Architecture

The neural network is composed of:

- 3 stacked LSTM layers
- Dropout layers between LSTMs
- Dense output layer for multi-step forecasting

Architecture:

```text
Input → LSTM → Dropout
      → LSTM → Dropout
      → LSTM → Dropout
      → Dense(predict_n)
```

Main characteristics:

| Component | Description |
|---|---|
| LSTM layers | Capture temporal dependencies |
| GELU activation | Used in the final LSTM layer |
| Dropout | Enables regularization and uncertainty estimation |
| Dense output | Produces multi-step forecasts |
| Monte Carlo Dropout | Generates probabilistic forecasts |

---

# Main Features

## Multi-step Forecasting

The model predicts multiple future weeks simultaneously.

Example:

```python
predict_n = 4
```

Forecasts the next 4 epidemiological weeks.

---

## Sliding Window Input

Historical observations are transformed into supervised learning samples.

Example:

```python
look_back = 12
```

Uses the previous 12 weeks to forecast future values.

---

## Monte Carlo Dropout Uncertainty

During inference, dropout remains active:

```python
model(Xdata, training=True)
```

The model performs 100 stochastic forward passes:

```python
predicted = np.stack(
    [model(Xdata, training=True) for i in range(100)],
    axis=2
)
```

This produces predictive distributions used to compute:

- median forecast
- 50% prediction interval
- 80% prediction interval
- 90% prediction interval
- 95% prediction interval

---

# Data Preprocessing

## Normalization

Features are normalized using maximum absolute scaling:

```python
normalize(..., norm='max')
```

Normalization is computed only using training data to avoid data leakage.

---

## Missing Values

Missing values are replaced with zeros:

```python
df.fillna(0)
```

---


# Training Pipeline

## Build the Model

```python
model = build_lstm(
    hidden=32,
    features=X_train.shape[2],
    predict_n=4,
    look_back=12,
    batch_size=4
)
```

---

## Train the Model

```python
model, hist = train_model(
    model=model,
    df=df,
    city=4106902,
    doenca='dengue',
    predict_n=4,
    look_back=12,
    epochs=100
)
```

---

# Cross-Validation

The training function supports K-Fold cross-validation:

```python
KFold(n_splits=4, shuffle=True, random_state=42)
```

Each fold is trained independently.

---

# Early Stopping

Training uses early stopping to reduce overfitting:

```python
EarlyStopping(
    monitor='val_loss',
    patience=patience,
    restore_best_weights=True
)
```

---

# Learning Rate Scheduling

Exponential learning rate decay:

```python
lr * math.exp(-0.1)
```

---

# Forecast Generation

Forecasts are generated using:

```python
apply_forecast(...)
```

The function:

1. loads the trained model
2. preprocesses the input data
3. generates stochastic forecasts
4. computes prediction intervals
5. exports the results as CSV

---

# Forecast Output

The generated forecast table contains:

| Column | Description |
|---|---|
| date | Forecast date |
| pred | Median prediction |
| lower_50 | Lower 50% interval |
| upper_50 | Upper 50% interval |
| lower_80 | Lower 80% interval |
| upper_80 | Upper 80% interval |
| lower_90 | Lower 90% interval |
| upper_90 | Upper 90% interval |
| lower_95 | Lower 95% interval |
| upper_95 | Upper 95% interval |

---

# Uncertainty Estimation

Prediction intervals are computed from empirical quantiles:

```python
np.percentile(pred, q, axis=2)
```

Example:

```python
2.5%  → lower_95
97.5% → upper_95
```

---

# Inverse Transformations

After prediction:

1. normalization is reversed
2. inverse Box-Cox transformation is applied

```python
pred = inv_boxcox(pred, 0.05) - 1
```

---

# Repository Structure

```text
.
├── saved_models/
├── forecast_tables/
├── tensorboard/
├── model_arima.py
└── lstm_model.py
```

---

# Dependencies

Main libraries:

```text
tensorflow
keras
numpy
pandas
scikit-learn
scipy
epiweeks
```

Install:

```bash
pip install tensorflow keras numpy pandas scikit-learn scipy epiweeks
```

---

# Example Workflow (`run_models.py`)

## 1. Prepare data

```python
df_c = mosq.get_infodengue(api_key = api_key,
                            disease = 'dengue', 
                          start_date = '2010-01-01',
                          end_date = '2025-11-10',
                          geocode = geocode)
```

---

## 2. Build model

```python
model = build_lstm(
    hidden=32,
    features=df.shape[1],
    predict_n=4,
    look_back=12,
    batch_size=4
)
```

---

## 3. Train

```python
model, hist = train_model(
    model=model,
    df=df,
    city=4106902,
    doenca='dengue',
    predict_n=4,
    look_back=12,
    epochs=100
)
```

---

## 4. Forecast

```python
forecast = apply_forecast(
    df=df,
    city=4106902,
    ini_date='2015-01-01',
    end_date='2024-01-01',
    look_back=12,
    predict_n=4,
    model_name='trained_4106902_dengue_model'
)
```

---

# Methodological Notes

This implementation follows a probabilistic deep learning approach for epidemiological forecasting.

Key methodological aspects include:

- sequence-to-sequence forecasting
- recurrent neural networks
- Monte Carlo Dropout approximation for Bayesian inference
- rolling-window supervised learning transformation
- temporal normalization without leakage

---

# Citation

If you use this repository in academic work, please cite the corresponding publication or repository.

---

