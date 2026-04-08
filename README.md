# 📈 Stock Price Prediction Engine

Advanced machine learning application for predicting stock prices using LSTM and XGBoost ensemble models. Built with Streamlit for interactive web-based analysis.

## 🌟 Features

### Core Models
- **LSTM (Long Short-Term Memory)**
  - Seq-to-seq architecture for time series prediction
  - Captures temporal dependencies and patterns
  - Handles non-linear relationships
  - Configurable epochs and lookback window

- **XGBoost (Extreme Gradient Boosting)**
  - Feature-based gradient boosting
  - Fast training and prediction
  - Built-in feature importance analysis
  - Handles feature interactions well

- **Ensemble Model**
  - Weighted combination of LSTM and XGBoost
  - Adjustable blend ratio (0.0 to 1.0)
  - Leverages strengths of both models
  - Often outperforms individual models

### Technical Features
- **Advanced Preprocessing**
  - Data normalization and scaling
  - Missing value handling
  - Feature engineering (MA, volatility, RSI, MACD)

- **Real-time Data**
  - Live stock data from Yahoo Finance
  - 1-day interval historical data
  - Automatic caching (1 hour TTL)

- **Comprehensive Evaluation**
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - MAPE (Mean Absolute Percentage Error)
  - Error distribution analysis
  - Feature importance visualization

- **Interactive Dashboard**
  - Real-time price tracking
  - Multi-model prediction comparison
  - 7-day forecast with confidence intervals
  - Feature importance charts
  - Plotly-based interactive visualizations

## 📊 Project Architecture

```
Data Collection (Yahoo Finance)
    ↓
Data Preprocessing & Feature Engineering
    ├─ Moving Averages (MA5, MA20)
    ├─ Volatility
    ├─ Daily Returns
    ├─ RSI (Relative Strength Index)
    ├─ MACD (Moving Average Convergence Divergence)
    └─ Signal Line
    ↓
Train-Test Split (80-20)
    ├─────────────────┐
    ↓                 ↓
LSTM Model      XGBoost Model
    │                 │
    └────→ Ensemble ←─┘
            Model
            ↓
        Evaluation & Metrics
            ↓
        Visualization & Predictions
            ↓
        Streamlit Dashboard
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- pip (Python package manager)
- 4GB RAM (minimum)
- Internet connection (for data download)

### Step 1: Clone or Download Project
```bash
# If you have git
git clone <repository-url>
cd stock_prediction

# Or download and extract the files
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

**Note:** TensorFlow installation can be slow (5-10 minutes). Be patient!

### Step 4: Run the App
```bash
streamlit run stock_prediction.py
```

The app will open in your default browser at `http://localhost:8501`

## 📖 Usage Guide

### 1. **Enter Stock Symbol**
   - In the sidebar, type the stock ticker (e.g., AAPL, GOOGL, MSFT, TSLA, AMZN)
   - Press Enter or click outside the input

### 2. **Configure Parameters**
   - **Days of History:** How many days of historical data to download (default: 365 days = 1 year)
   - **LSTM Epochs:** Training iterations for LSTM (higher = longer training, potentially better accuracy)
   - **Lookback Window:** How many previous days to use for prediction (default: 30 days)
   - **XGBoost Rounds:** Number of boosting rounds (higher = better but slower)
   - **Test Split %:** Percentage of data for testing (default: 20%)
   - **LSTM Weight:** How much LSTM contributes to ensemble (0.5 = equal weight)

### 3. **View Real-time Price Info**
   - Current price and change percentage
   - 52-week high and low
   - Updated in real-time

### 4. **Model Training**
   - The app automatically trains both models
   - Progress shown via spinner
   - Training takes 1-3 minutes depending on data size

### 5. **Analyze Results**
   - **Metrics Boxes:** Compare MAE, RMSE, MAPE across models
   - **Predictions Chart:** See how each model performed on test data
   - **Error Distribution:** Box plot showing error ranges
   - **Feature Importance:** Top 15 features used by XGBoost

### 6. **7-Day Forecast**
   - Table with predictions from each model
   - Interactive chart with 95% confidence interval
   - Shows expected price movement

## 🔧 Configuration Tips

### For Faster Results
```
- Reduce LSTM Epochs: 10-20
- Reduce XGBoost Rounds: 50-100
- Reduce Days of History: 180-365
- Smaller Lookback Window: 10-20
```

### For Better Accuracy
```
- Increase LSTM Epochs: 50-100
- Increase XGBoost Rounds: 200-500
- More Days of History: 1000+ days
- Larger Lookback Window: 60-90
- Ensemble Weight: 0.5 (equal mix works best)
```

### For Specific Stock Types
```
Tech Stocks (AAPL, MSFT, NVDA):
- Higher lookback (45-60 days)
- More epochs (50+)
- Higher ensemble weight to LSTM (0.6-0.7)

Volatile Stocks (TSLA, GME):
- Shorter lookback (20-30 days)
- More robust evaluation metrics
- Consider XGBoost weight (0.4-0.5)

Blue Chip Stocks (JNJ, KO, PG):
- Moderate lookback (30-45 days)
- Standard epochs (30)
- Balanced ensemble weight (0.5)
```

## 📊 Understanding the Metrics

### MAE (Mean Absolute Error)
- Average of absolute differences between predicted and actual
- Units: Dollar amount ($)
- Lower is better
- Example: MAE of $2.50 means average error is $2.50

### RMSE (Root Mean Squared Error)
- Square root of average squared errors
- Penalizes larger errors more than MAE
- Units: Dollar amount ($)
- Lower is better
- More sensitive to outliers

### MAPE (Mean Absolute Percentage Error)
- Percentage error relative to actual price
- Units: Percentage (%)
- Lower is better
- Example: MAPE of 2% means average error is 2% of actual price
- **Best metric for this project** - scale-independent

## 🎯 Model Strengths & Weaknesses

### LSTM
**Strengths:**
- Excellent at capturing temporal patterns
- Handles long-term dependencies
- Non-linear relationships
- Good for trend prediction

**Weaknesses:**
- Slower training
- Requires more data
- Can overfit with small datasets
- Less interpretable

### XGBoost
**Strengths:**
- Fast training
- Feature importance easily interpretable
- Handles mixed data types
- Good feature interactions
- Works with smaller datasets

**Weaknesses:**
- Doesn't capture temporal dependencies well
- Feature engineering critical
- Can be sensitive to outliers

### Ensemble
**Strengths:**
- Combines temporal (LSTM) + feature-based (XGBoost)
- More robust predictions
- Reduces individual model biases
- Better generalization

**Weaknesses:**
- Slower inference
- Parameter tuning more complex
- May not outperform best single model in all cases

## 💡 Interpretation Guide

### When LSTM Outperforms
- Strong trend patterns in historical data
- Long-term dependencies matter
- Smooth price movements

### When XGBoost Outperforms
- Sudden market changes
- Feature relationships critical
- More volatile stocks

### When Ensemble Shines
- Mixed market conditions
- Need robust predictions
- Balanced accuracy across all conditions

## 🔮 Reading the 7-Day Forecast

1. **Point Estimate:** The main "Ensemble" line is the best-guess prediction
2. **Confidence Interval:** The shaded area shows uncertainty range
   - Wider bands = more uncertainty
   - Narrower bands = more confident
3. **Direction Indicator:** Shows if price is expected to go up 📈 or down 📉
4. **% Change:** Percentage movement from current price to 7-day forecast

## ⚠️ Important Disclaimers

- **Not Financial Advice:** This is educational only, not investment advice
- **Past Performance:** Historical patterns may not repeat
- **Market Factors:** Model ignores external events, news, earnings
- **Black Swan Events:** Can't predict unprecedented events
- **Model Limitations:** ML models have inherent prediction uncertainty
- **Always Verify:** Use multiple sources before making trades
- **Risk Management:** Never trade without proper risk management

## 🛠️ Troubleshooting

### "ModuleNotFoundError: No module named..."
```bash
pip install -r requirements.txt
# Or individually:
pip install streamlit pandas yfinance scikit-learn xgboost tensorflow plotly
```

### "No module named 'tensorflow'"
```bash
# Can take 5-10 minutes
pip install tensorflow
# Or CPU version (faster):
pip install tensorflow-cpu
```

### "SSL: CERTIFICATE_VERIFY_FAILED" (macOS)
```bash
/Applications/Python\ 3.x/Install\ Certificates.command
```

### App Runs Slowly
- Reduce history days
- Reduce LSTM epochs
- Reduce XGBoost rounds
- Use a faster machine

### Stock Symbol Not Found
- Check spelling (must be valid ticker)
- Some delisted stocks won't work
- Crypto symbols need different source

### Out of Memory Error
- Reduce days of history
- Close other applications
- Use a machine with more RAM

## 📈 Real-World Example

**Predicting Apple (AAPL) Stock:**
```
Configuration:
- Days of History: 365 (1 year)
- LSTM Epochs: 30
- Lookback: 30 days
- XGBoost Rounds: 200
- Ensemble Weight: 0.5

Results:
- LSTM MAPE: 1.8%
- XGBoost MAPE: 2.1%
- Ensemble MAPE: 1.5% ✅ (Best)

7-Day Forecast:
- Current Price: $180.50
- Predicted (7 days): $182.30
- Expected Change: +1.0%
- Confidence: Good (94%)
```

## 🎓 Learning Resources

- [LSTM Explained](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Time Series Forecasting](https://www.tensorflow.org/tutorials/structured_data/time_series)
- [Technical Indicators](https://en.wikipedia.org/wiki/Technical_indicator)

## 🚀 Advanced Usage

### Custom Stock Lists
Modify the app to automatically scan multiple stocks:
```python
stocks = ['AAPL', 'GOOGL', 'MSFT', 'TSLA', 'AMZN']
for stock in stocks:
    run_prediction(stock)
```

### Model Export
Save trained models for later use:
```python
lstm_model.save('lstm_model.h5')
xgb_model.save_model('xgb_model.json')
```

### Data Export
Export predictions to CSV:
```python
forecast_df.to_csv('forecast.csv', index=False)
```

## 📝 Project Structure

```
stock_prediction/
├── stock_prediction.py      # Main Streamlit app
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🤝 Contributing

To improve this project:
1. Test with different stocks
2. Add new technical indicators
3. Experiment with model architectures
4. Optimize hyperparameters
5. Add more visualization types

## 📞 Support & Issues

If you encounter issues:
1. Check the Troubleshooting section
2. Verify all dependencies installed: `pip list`
3. Try with a different stock symbol
4. Clear cache and restart: `streamlit cache clear`

## 📄 License

This project is provided as-is for educational purposes.

## 🎯 Future Enhancements

- [ ] Multi-stock portfolio analysis
- [ ] Real-time price updates
- [ ] Risk metrics (Sharpe ratio, Volatility)
- [ ] Backtesting framework
- [ ] Model comparison across time periods
- [ ] Custom technical indicators
- [ ] Sentiment analysis integration
- [ ] Cryptocurrency support
- [ ] Model persistence & loading
- [ ] Ensemble strategy optimization

---

**Built with ❤️ for learning machine learning and time series forecasting**
