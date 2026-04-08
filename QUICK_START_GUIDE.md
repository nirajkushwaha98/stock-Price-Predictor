# 🚀 Stock Price Prediction - Quick Start Guide

## English Version

### Installation (Windows/Mac/Linux)

**Step 1: Install Python Packages**
```bash
pip install -r requirements.txt
```

**Step 2: Run the App**
```bash
streamlit run stock_prediction.py
```

**Step 3: Open Browser**
Browser will open at: `http://localhost:8501`

### First-Time Setup
1. Enter stock symbol (e.g., `AAPL`, `GOOGL`, `MSFT`)
2. Leave default settings for quick test
3. Click and wait 1-2 minutes for training
4. Explore the dashboard!

### Key Features
- **LSTM Model:** Captures time series patterns
- **XGBoost Model:** Feature-based predictions
- **Ensemble:** Best of both worlds
- **Live Data:** Real-time stock prices
- **7-Day Forecast:** With confidence intervals

### Common Stocks to Try
- Tech: AAPL, MSFT, GOOGL, NVDA, META
- Finance: JPM, BAC, GS, WFC
- Retail: AMZN, WMT, TGT
- Auto: TSLA, F, GM
- Energy: XOM, CVX, COP

---

## 🇮🇳 हिंदी संस्करण

### Installation (Windows/Mac/Linux)

**Step 1: Python Packages Install करो**
```bash
pip install -r requirements.txt
```

**Step 2: App चलाओ**
```bash
streamlit run stock_prediction.py
```

**Step 3: Browser में खोलो**
`http://localhost:8501` पर जाओ

### पहली बार Setup
1. Stock symbol enter करो (जैसे: `AAPL`, `GOOGL`, `TCS`)
2. Default settings रखो quick test के लिए
3. 1-2 मिनट wait करो training के लिए
4. Dashboard explore करो!

### कौन से Stocks Try करें
- Tech: AAPL, MSFT, GOOGL, META
- Finance: JPM, HDFC (भारतीय: RELIANCE, TCS, INFY)
- Retail: AMZN, WMT
- Auto: TSLA, BAJAJ
- Energy: XOM, OIL

---

## 📊 Dashboard समझना

### Top Metrics (ऊपर)
- **Current Price:** आज का stock price
- **Change %:** कितना ऊपर या नीचे गया
- **52-Week High/Low:** एक साल में highest और lowest

### Model Performance (Metrics)
```
MAE (Mean Absolute Error) - कितनी अंतर की गलती 
RMSE - बड़ी गलतियों को ज्यादा वजन देता है
MAPE (%) - सबसे अच्छी metric (percentage में)
```

### 3 Models की तुलना
```
LSTM      → Time series patterns पकड़ता है
XGBoost   → Features के relations देखता है
Ensemble  → दोनों को मिलाता है (आमतौर पर बेस्ट)
```

### Charts
1. **Predictions vs Actual:** Model कितना सही निकला
2. **Error Distribution:** Errors कहाँ हैं
3. **Feature Importance:** कौन सी चीजें मायने रखती हैं
4. **7-Day Forecast:** अगले 7 दिन का prediction

---

## ⚙️ Settings समझना

### Basic Settings
```
Days of History: कितना पुराना data चाहिए
  └─ 365 = 1 साल (अच्छा शुरुआत)
  └─ 500+ = ज्यादा सटीक

Lookback Window: कितने दिन पहले के data से सीखे
  └─ 30 = 30 दिन का pattern देखे
  └─ 60 = 60 दिन का pattern (slow but detailed)

Test Split: कितना data testing के लिए
  └─ 20% (standard)
```

### Model Settings
```
LSTM Epochs: कितनी बार training करो
  └─ 30 = तेज़ (1 मिनट)
  └─ 100 = धीमा लेकिन सटीक (5 मिनट)

XGBoost Rounds: Boosting iterations
  └─ 200 = default (अच्छा balance)
  └─ 500 = बेहतर लेकिन slow

Ensemble Weight: कितना LSTM का वजन
  └─ 0.5 = बराबर (अक्सर सबसे अच्छा)
  └─ 0.7 = LSTM को ज्यादा trust करो
  └─ 0.3 = XGBoost को ज्यादा trust करो
```

---

## 📈 Results को समझना

### MAPE % (सबसे महत्वपूर्ण)
```
< 1%    → Excellent! (शानदार)
1-2%    → Very Good (बहुत अच्छा)
2-3%    → Good (अच्छा)
3-5%    → Acceptable (ठीक है)
> 5%    → Poor (कमजोर)
```

### 7-Day Forecast
```
Ensemble = सबसे अच्छा prediction
Confidence Interval = कितना uncertainty है

Examples:
$100 ± $2 = काफी confident (narrow band)
$100 ± $5 = कम confident (wide band)
```

### कौन सा Model बेहतर है?
```
अगर LSTM MAPE < XGBoost MAPE
  → Trend-based patterns मजबूत हैं

अगर XGBoost MAPE < LSTM MAPE
  → Feature relationships मायने रखती हैं

अगर Ensemble MAPE सबसे कम है
  → Ensemble सबसे robust है (ideal)
```

---

## 🎯 Trading Tips (शिक्षा के लिए)

### सही तरीका
✅ Multiple models use करो (LSTM, XGBoost, Ensemble)
✅ Long history use करो (6-12 months)
✅ Confidence interval देखो
✅ News और sentiment भी check करो
✅ Risk management करो (stop loss set करो)
✅ Portfolio diversify करो

### गलत तरीका
❌ Single model पर 100% भरोसा
❌ सिर्फ 3 महीने का data
❌ Prediction को आँख बंद करके follow करो
❌ सब पैसा एक stock में
❌ Emotions के साथ trade करना
❌ Real money से पहले practice न करना

---

## 🔧 Troubleshooting

### Problem: "No module named 'tensorflow'"
**Solution:**
```bash
pip install tensorflow
# या यह try करो (faster):
pip install tensorflow-cpu
```

### Problem: App खुलकर crash हो गया
**Solution:**
```bash
# Cache clear करो
streamlit cache clear
# फिर से चलाओ
streamlit run stock_prediction.py
```

### Problem: Stock symbol काम नहीं कर रहा
**Solutions:**
- Spelling check करो (e.g., AAPL not AAP)
- Yahoo Finance पर symbol verify करो
- Delisted stocks काम नहीं करते

### Problem: बहुत slow चल रहा है
**Solutions:**
```
- Days of History: 365 से 180 करो
- LSTM Epochs: 100 से 20 करो
- XGBoost Rounds: 500 से 100 करो
```

---

## 💡 Experiment करने के लिए Ideas

### Experiment 1: सबसे सटीक settings खोजो
```
Different epoch और lookback combinations try करो
हर combination के लिए MAPE record करो
Pattern देखो क्या सबसे अच्छा है
```

### Experiment 2: Different stocks compare करो
```
Tech stocks (AAPL, MSFT)
Finance stocks (JPM, BAC)
Energy stocks (XOM, CVX)

क्या अलग sectors को अलग settings चाहिए?
```

### Experiment 3: Ensemble weight optimize करो
```
0.3, 0.4, 0.5, 0.6, 0.7 try करो
कौन सा सबसे अच्छा है अलग-अलग stocks के लिए?
```

### Experiment 4: Long term forecast accuracy
```
आज का forecast करो
7 दिन बाद actual price देखो
Model कितना सही था?
Record करो और सीखो!
```

---

## 📚 अगला कदम

### Basic से Advanced
1. **अभी:** Stock price prediction करो
2. **अगला:** Multiple stocks की portfolio analyze करो
3. **तब:** Risk metrics add करो (Sharpe ratio, etc.)
4. **Later:** Cryptocurrency और commodities try करो

### Skills Development
- Time series forecasting (अभी)
- Feature engineering (अगला)
- Model interpretability (तब)
- Production deployment (बाद में)

### Resources
- [Streamlit Docs](https://docs.streamlit.io)
- [Keras/TensorFlow](https://www.tensorflow.org)
- [XGBoost Docs](https://xgboost.readthedocs.io)
- [Technical Analysis](https://en.wikipedia.org/wiki/Technical_analysis)

---

## ⚠️ महत्वपूर्ण Notes

### यह शिक्षा है, निवेश सलाह नहीं
- ML models गलत हो सकते हैं
- Past performance future guarantee नहीं है
- Black swan events predict नहीं हो सकते
- Always verify से पहले trade करो
- Real money से पहले backtesting करो
- Stop losses लगाना बहुत जरूरी है

### Model की limitations
- News/events account नहीं करता
- Sudden market crashes predict नहीं कर सकता
- IPOs, stock splits को handle नहीं करता
- Limited historical data पर काम करता है

---

## 🎓 सीखने के लिए यह project क्यों अच्छा है

1. **Real-world Data:** Yahoo Finance से live data
2. **Multiple Models:** LSTM vs XGBoost comparison
3. **Visualization:** Plotly से interactive charts
4. **Production App:** Streamlit से web app
5. **Time Series:** सबसे challenging ML problem
6. **Portfolio Value:** Resume के लिए बढ़िया project

---

## 🚀 Success Metrics

अगर यह काम कर रहा है:
- ✅ App बिना error के चलता है
- ✅ Charts दिखाई देते हैं
- ✅ 3 models के लिए MAPE < 5%
- ✅ 7-day forecast बनता है
- ✅ Different stocks काम करते हैं

अगर सब काम कर रहा है, तो **Congratulations! 🎉**

---

**Happy Learning! अगर कोई सवाल है तो README.md पढ़ो।**

**Made with ❤️ for learners**
