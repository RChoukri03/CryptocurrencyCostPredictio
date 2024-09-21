# **Cryptocurrency Cost Prediction Using Deep Learning**

## **Overview**
This project aims to predict the price movements of cryptocurrencies using advanced deep learning models. The model is structured to handle **binary classification**, where it predicts whether the price of a cryptocurrency will **increase** or **decrease** after a specified time window.

The system integrates **LSTM (Long Short-Term Memory) networks**, **Bidirectional LSTMs**, and **Transformer blocks with attention mechanisms**. These layers help capture temporal patterns in the data, making the model well-suited for handling time series information such as cryptocurrency prices.

In addition to model architecture, the project features **real-time data fetching** from Binance and **configurable settings** for training, validation, and data processing.

## **How It Works**
The model processes historical price data and predicts whether the price will increase or decrease after a given time window:
- **Input Sequence Length (`sequence_length`)**: The model takes in a sequence of **N minutes** (e.g., 120 minutes of historical data).
- **Prediction Window (`prediction_window`)**: The model then predicts whether the price will rise or fall **K minutes** after the input sequence (e.g., after 15 minutes).

### **Binary Classification Task**
- **Output**: The model predicts a binary class:
  - **1**: Price is expected to increase after the `prediction_window`.
  - **0**: Price is expected to decrease after the `prediction_window`.

## **Key Features**
1. **Deep Learning Architecture**:
   - **LSTM and Bi-LSTM Layers**: Capture long-term dependencies in price movements.
   - **Transformer Blocks**: Enhance the model's attention to important parts of the time series data.
   - **Custom Attention Layer**: Further focuses the model on relevant data points to improve prediction accuracy.

2. **Real-Time Data Fetching**:
   - Data is fetched from Binance using their public API, making it easy to obtain the latest cryptocurrency prices for training and evaluation.

3. **Configurable Model and Training**:
   - The model, training parameters, and data are highly configurable through YAML configuration files.

4. **Callbacks for Efficient Training**:
   - Includes **EarlyStopping**, **ModelCheckpoint**, **TensorBoard**, and **TerminateOnNaN** to optimize and monitor training.

## **Configuration**
The model and data pipelines are customizable through configuration files, 'data/config.yaml' for datDownloader configuration, and config.yaml for model & train configuration

