# 🌡️ Climate Temperature Prediction with LSTM

This project uses **Long Short-Term Memory (LSTM)** neural networks to predict hourly temperatures from the [Jena Climate Dataset](https://www.kaggle.com/datasets/mnassrib/jena-climate) (2009–2016).

---

## 📂 Project Workflow

### 1. **Data Collection & Processing**

* Dataset: `jena_climate_2009_2016.csv`
* Original shape: **420,551 rows × 15 columns**
* Downsampled to **hourly data** by taking every 6th row → ~70,091 rows
* Converted `Date Time` column to **datetime index** for time-series handling.

---

### 2. **Feature Selection**

* Target Variable: `T (degC)` (temperature in Celsius).
* Used a **sliding window** approach:

  * Past 60 time steps (hours) → predict next 1 step.

---

### 3. **Data Preprocessing**

* Applied **MinMaxScaler** (range 0–1) to normalize features.
* Created training & testing datasets:

  * **80% training**
  * **20% testing**

---

### 4. **Model Architecture (LSTM)** 🧠

* Sequential model with:

  * 2 × LSTM layers (50 units each)
  * Dropout layers (20%) for regularization
  * Dense output layer (1 neuron for temperature prediction)
* Optimizer: **Adam**
* Loss Function: **Mean Squared Error (MSE)**

---

### 5. **Model Training**

* Epochs: 20
* Batch size: 32
* Trained on GPU/CPU

---

### 6. **Evaluation** 📊

* Compared predicted vs. actual temperatures on test set.
* Metrics:

  * **MSE** (Mean Squared Error)
  * **RMSE** (Root Mean Squared Error)

---

### 7. **Visualization** 📈

* Line plots of **Actual vs. Predicted** temperatures.
* Clear overlap showing strong predictive accuracy.

---

## 🚀 How to Run

1. Clone repo & install dependencies:

   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
   ```
2. Download dataset: [Jena Climate](https://www.kaggle.com/datasets/mnassrib/jena-climate)
3. Run Jupyter Notebook / Python script.

---

## 📌 Future Improvements

* Try **GRU** or hybrid CNN-LSTM models.
* Increase prediction horizon (multi-step forecasting).
* Use additional features (humidity, pressure, windspeed, etc.) for multivariate forecasting.

---

🔥 With this LSTM model, we can forecast **future temperatures** with high accuracy using historical climate data! 🌍⚡
