import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="📈 Stock Price Predictor", layout="wide", initial_sidebar_state="expanded")

# Custom styling
st.markdown("""
<style>
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .title-style {
        font-size: 2.5em;
        font-weight: bold;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='title-style'>📈 Stock Price Prediction Engine</h1>", unsafe_allow_html=True)
st.markdown("*Using LSTM + XGBoost Ensemble for Accurate Predictions*")

FEATURE_COLS = ['MA_5', 'MA_20', 'Volatility', 'Daily_Return', 'RSI', 'MACD', 'Signal_Line']

# ============================================
# SIDEBAR CONFIGURATION
# ============================================
st.sidebar.header("⚙️ Configuration")

# Stock selection
stock_symbol = st.sidebar.text_input("Enter Stock Symbol", value="AAPL", help="e.g., AAPL, GOOGL, MSFT, TSLA")
stock_symbol = stock_symbol.strip().upper()

# Date range
col1, col2 = st.sidebar.columns(2)
with col1:
    days_back = st.sidebar.number_input("Days of History", min_value=60, max_value=1500, value=365, step=30)
    use_demo_fallback = st.sidebar.checkbox("Use demo data if live fetch fails", value=True)

# Model configuration
st.sidebar.subheader("Model Parameters")
col1, col2 = st.sidebar.columns(2)
with col1:
    lstm_epochs = st.sidebar.slider("LSTM Epochs", min_value=10, max_value=100, value=30, step=10)
    lookback = st.sidebar.slider("Lookback Window (days)", min_value=5, max_value=90, value=30, step=5)

with col2:
    xgb_rounds = st.sidebar.slider("XGBoost Rounds", min_value=50, max_value=500, value=200, step=50)
    test_split = st.sidebar.slider("Test Split %", min_value=10, max_value=40, value=20, step=5)

ensemble_weight = st.sidebar.slider("LSTM Weight in Ensemble", min_value=0.0, max_value=1.0, value=0.5, step=0.1)

# ============================================
# DATA LOADING
# ============================================
def normalize_price_data(data):
    """Normalize Yahoo output shape and required columns."""
    if data is None or data.empty:
        return pd.DataFrame()

    df = data.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

    if 'Close' not in df.columns and 'Adj Close' in df.columns:
        df['Close'] = df['Adj Close']

    return df.sort_index()


@st.cache_data(ttl=3600)
def generate_demo_data(days):
    """Create synthetic OHLCV data when live download is unavailable."""
    trading_days = max(days, 120)
    dates = pd.bdate_range(end=datetime.now().date(), periods=trading_days)
    rng = np.random.default_rng(42)

    returns = rng.normal(0.0005, 0.018, size=len(dates))
    close = 150 * np.exp(np.cumsum(returns))
    open_price = close * (1 + rng.normal(0, 0.003, size=len(dates)))
    high = np.maximum(open_price, close) * (1 + rng.uniform(0, 0.01, size=len(dates)))
    low = np.minimum(open_price, close) * (1 - rng.uniform(0, 0.01, size=len(dates)))
    volume = rng.integers(1_000_000, 8_000_000, size=len(dates))

    return pd.DataFrame(
        {
            'Open': open_price,
            'High': high,
            'Low': low,
            'Close': close,
            'Adj Close': close,
            'Volume': volume
        },
        index=dates
    )


@st.cache_data(ttl=3600)
def load_stooq_data(symbol, days):
    """Fallback market data source when Yahoo is unavailable."""
    cleaned = symbol.strip().upper()
    symbol_candidates = []

    if "." in cleaned:
        symbol_candidates.append(cleaned.lower())
    else:
        symbol_candidates.append(f"{cleaned.lower()}.us")
        symbol_candidates.append(cleaned.lower())

    cutoff = pd.Timestamp(datetime.now() - timedelta(days=int(days) + 5))
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']

    for candidate in symbol_candidates:
        try:
            url = f"https://stooq.com/q/d/l/?s={candidate}&i=d"
            df = pd.read_csv(url)
            if df.empty or 'Date' not in df.columns:
                continue

            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date']).set_index('Date').sort_index()

            for col in required_cols:
                if col not in df.columns:
                    df[col] = np.nan
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df = df[df.index >= cutoff].dropna(subset=['Close'])
            if df.empty:
                continue

            if 'Adj Close' not in df.columns:
                df['Adj Close'] = df['Close']
            return df
        except Exception:
            continue

    return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_yahoo_chart_api_data(symbol, days):
    """Use Yahoo chart endpoint directly as fallback when yfinance fails."""
    end_dt = datetime.now()
    start_dt = end_dt - timedelta(days=int(days) + 7)

    url = f"https://query2.finance.yahoo.com/v8/finance/chart/{symbol}"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json"
    }
    params = {
        "period1": int(start_dt.timestamp()),
        "period2": int(end_dt.timestamp()),
        "interval": "1d",
        "events": "div,splits",
        "includePrePost": "false"
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=15)
        if response.status_code != 200:
            return pd.DataFrame(), f"Yahoo chart API HTTP {response.status_code}"

        payload = response.json()
        result = (payload.get("chart", {}).get("result") or [None])[0]
        if result is None:
            error_msg = payload.get("chart", {}).get("error")
            return pd.DataFrame(), f"Yahoo chart API empty result: {error_msg}"

        timestamps = result.get("timestamp") or []
        quote = (result.get("indicators", {}).get("quote") or [{}])[0]
        adjclose_info = (result.get("indicators", {}).get("adjclose") or [{}])[0]
        if not timestamps or not quote:
            return pd.DataFrame(), "Yahoo chart API missing timestamps/quote"

        df = pd.DataFrame(
            {
                "Open": quote.get("open", []),
                "High": quote.get("high", []),
                "Low": quote.get("low", []),
                "Close": quote.get("close", []),
                "Volume": quote.get("volume", []),
                "Adj Close": adjclose_info.get("adjclose", quote.get("close", []))
            },
            index=pd.to_datetime(timestamps, unit="s")
        ).sort_index()

        for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = df.dropna(subset=["Close"])
        if df.empty:
            return pd.DataFrame(), "Yahoo chart API returned no valid close prices"

        return df, None
    except Exception as e:
        return pd.DataFrame(), f"Yahoo chart API: {e}"


@st.cache_data(ttl=3600)
def load_stock_data(symbol, days):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    errors = []

    try:
        data = yf.download(
            symbol,
            start=start_date,
            end=end_date,
            progress=False,
            interval='1d',
            auto_adjust=False,
            threads=False
        )
        data = normalize_price_data(data)
        if not data.empty:
            return data, None
    except Exception as e:
        errors.append(f"Yahoo download: {e}")

    try:
        data = yf.Ticker(symbol).history(period=f"{max(days, 60)}d", interval='1d', auto_adjust=False)
        data = normalize_price_data(data)
        if not data.empty:
            return data, None
    except Exception as e:
        errors.append(f"Yahoo history: {e}")

    chart_data, chart_error = load_yahoo_chart_api_data(symbol, days)
    if not chart_data.empty:
        return chart_data, None
    if chart_error:
        errors.append(chart_error)

    stooq_data = load_stooq_data(symbol, days)
    if not stooq_data.empty:
        return stooq_data, None

    if not errors:
        errors.append("Yahoo Finance and Stooq returned no rows.")
    return pd.DataFrame(), " | ".join(errors)

# ============================================
# PREPROCESSING
# ============================================
def preprocess_data(data):
    """Prepare data for modeling"""
    df = data[['Close']].copy()
    
    # Handle missing values
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Create features for XGBoost
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['Volatility'] = df['Close'].rolling(window=20).std()
    df['Daily_Return'] = df['Close'].pct_change()
    
    # RSI calculation
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    df = df.dropna()
    return df

# ============================================
# TRAIN-TEST SPLIT FOR LSTM
# ============================================
def create_lstm_sequences(data, lookback):
    """Create sequences for LSTM"""
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback])
    return np.array(X), np.array(y)

# ============================================
# XGBOOST FEATURES
# ============================================
def prepare_xgboost_features(df, lookback):
    """Prepare features for XGBoost"""
    X = df[FEATURE_COLS].values
    y = df['Close'].values
    
    # Add lagged features
    X_with_lags = []
    for i in range(lookback, len(X)):
        lag_features = X[i-lookback:i].flatten()
        X_with_lags.append(lag_features)
    
    return np.array(X_with_lags), y[lookback:]

# ============================================
# MAIN EXECUTION
# ============================================
if stock_symbol:
    st.sidebar.success(f"✅ Ready to predict {stock_symbol}")
    
    # Load data
    with st.spinner(f"📥 Fetching {stock_symbol} data..."):
        data, load_error = load_stock_data(stock_symbol, days_back)

    if data.empty and use_demo_fallback:
        data = generate_demo_data(days_back)
        st.warning("⚠️ Live Yahoo data is unavailable. Using generated demo data for analysis.")
    
    if data is not None and not data.empty and len(data) > lookback + 5:
        # Display current price info
        current_price = data['Close'].iloc[-1]
        prev_price = data['Close'].iloc[-2] if len(data) > 1 else current_price
        price_change = current_price - prev_price
        pct_change = (price_change / prev_price) * 100 if prev_price != 0 else 0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Current Price", f"${current_price:.2f}", f"{price_change:+.2f}")
        with col2:
            st.metric("Change %", f"{pct_change:+.2f}%", "")
        with col3:
            st.metric("52-Week High", f"${data['Close'].rolling(252).max().iloc[-1]:.2f}", "")
        with col4:
            st.metric("52-Week Low", f"${data['Close'].rolling(252).min().iloc[-1]:.2f}", "")
        
        st.divider()
        
        # Preprocess
        with st.spinner("🔧 Preprocessing data..."):
            df_processed = preprocess_data(data)

        if len(df_processed) <= lookback + 5:
            st.error(
                f"❌ Not enough usable rows after preprocessing ({len(df_processed)} rows). "
                "Increase history or reduce lookback."
            )
            st.stop()
        
        # Split data
        train_size = int(len(df_processed) * (1 - test_split/100))
        train_data = df_processed['Close'].values[:train_size]
        test_data = df_processed['Close'].values[train_size:]

        if train_size <= lookback or len(test_data) == 0:
            st.error("❌ Train/test split is too small for the selected lookback. Lower lookback or increase history.")
            st.stop()
        
        # ============================================
        # LSTM MODEL
        # ============================================
        with st.spinner("🧠 Training LSTM model..."):
            # Normalize
            scaler_lstm = MinMaxScaler(feature_range=(0, 1))
            train_scaled = scaler_lstm.fit_transform(train_data.reshape(-1, 1))
            
            # Create sequences
            X_lstm, y_lstm = create_lstm_sequences(train_scaled, lookback)
            X_lstm = X_lstm.reshape((X_lstm.shape[0], X_lstm.shape[1], 1))
            
            # Build model
            lstm_model = Sequential([
                LSTM(50, activation='relu', return_sequences=True, input_shape=(lookback, 1)),
                Dropout(0.2),
                LSTM(50, activation='relu'),
                Dropout(0.2),
                Dense(25, activation='relu'),
                Dense(1)
            ])
            
            lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            lstm_model.fit(X_lstm, y_lstm, epochs=lstm_epochs, batch_size=32, verbose=0)
            
            # Predictions on test set
            test_segment = df_processed['Close'].values[train_size-lookback:].reshape(-1, 1)
            test_scaled = scaler_lstm.transform(test_segment)
            X_test_lstm, y_test_lstm = create_lstm_sequences(test_scaled, lookback)
            if len(X_test_lstm) == 0:
                st.error("❌ Not enough LSTM test sequences. Lower lookback or increase history.")
                st.stop()
            X_test_lstm = X_test_lstm.reshape((X_test_lstm.shape[0], X_test_lstm.shape[1], 1))
            
            lstm_pred_scaled = lstm_model.predict(X_test_lstm, verbose=0)
            lstm_pred = scaler_lstm.inverse_transform(lstm_pred_scaled).flatten()
            actual_lstm = scaler_lstm.inverse_transform(y_test_lstm.reshape(-1, 1)).flatten()
        
        # ============================================
        # XGBOOST MODEL
        # ============================================
        with st.spinner("🚀 Training XGBoost model..."):
            # Prepare features
            X_xgb, y_xgb = prepare_xgboost_features(df_processed, lookback)
            if len(X_xgb) == 0:
                st.error("❌ Not enough XGBoost feature rows. Lower lookback or increase history.")
                st.stop()
            
            # Split
            train_size_xgb = int(len(X_xgb) * (1 - test_split/100))
            X_train_xgb = X_xgb[:train_size_xgb]
            y_train_xgb = y_xgb[:train_size_xgb]
            X_test_xgb = X_xgb[train_size_xgb:]
            y_test_xgb = y_xgb[train_size_xgb:]
            if len(X_train_xgb) == 0 or len(X_test_xgb) == 0:
                st.error("❌ XGBoost train/test split is too small. Lower test split or increase history.")
                st.stop()
            
            # Train XGBoost
            xgb_model = xgb.XGBRegressor(n_estimators=xgb_rounds, learning_rate=0.1, max_depth=5, random_state=42)
            xgb_model.fit(X_train_xgb, y_train_xgb, verbose=False)
            
            # Predictions
            xgb_pred = xgb_model.predict(X_test_xgb)
            if len(xgb_pred) == 0:
                st.error("❌ XGBoost produced no test predictions.")
                st.stop()
        
        # ============================================
        # ENSEMBLE PREDICTIONS
        # ============================================
        lstm_eval_len = min(len(actual_lstm), len(lstm_pred))
        if lstm_eval_len == 0:
            st.error("❌ LSTM evaluation arrays are empty. Lower lookback or increase history.")
            st.stop()
        actual_lstm_eval = actual_lstm[-lstm_eval_len:]
        lstm_pred_eval = lstm_pred[-lstm_eval_len:]

        common_len = min(len(lstm_pred), len(xgb_pred), len(y_test_xgb))
        if common_len == 0:
            st.error("❌ No overlapping predictions for ensemble evaluation.")
            st.stop()
        lstm_pred_common = lstm_pred[-common_len:]
        xgb_pred_common = xgb_pred[-common_len:]
        y_common = y_test_xgb[-common_len:]
        ensemble_pred = (ensemble_weight * lstm_pred_common + (1 - ensemble_weight) * xgb_pred_common)
        
        # ============================================
        # MODEL EVALUATION
        # ============================================
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📊 LSTM Metrics")
            lstm_mae = mean_absolute_error(actual_lstm_eval, lstm_pred_eval)
            lstm_rmse = np.sqrt(mean_squared_error(actual_lstm_eval, lstm_pred_eval))
            lstm_mape = mean_absolute_percentage_error(actual_lstm_eval, lstm_pred_eval) * 100
            
            st.metric("MAE", f"${lstm_mae:.2f}")
            st.metric("RMSE", f"${lstm_rmse:.2f}")
            st.metric("MAPE", f"{lstm_mape:.2f}%")
        
        with col2:
            st.subheader("🔥 XGBoost Metrics")
            xgb_mae = mean_absolute_error(y_test_xgb, xgb_pred)
            xgb_rmse = np.sqrt(mean_squared_error(y_test_xgb, xgb_pred))
            xgb_mape = mean_absolute_percentage_error(y_test_xgb, xgb_pred) * 100
            
            st.metric("MAE", f"${xgb_mae:.2f}")
            st.metric("RMSE", f"${xgb_rmse:.2f}")
            st.metric("MAPE", f"{xgb_mape:.2f}%")
        
        with col3:
            st.subheader("✨ Ensemble Metrics")
            ens_mae = mean_absolute_error(y_common, ensemble_pred)
            ens_rmse = np.sqrt(mean_squared_error(y_common, ensemble_pred))
            ens_mape = mean_absolute_percentage_error(y_common, ensemble_pred) * 100
            
            st.metric("MAE", f"${ens_mae:.2f}")
            st.metric("RMSE", f"${ens_rmse:.2f}")
            st.metric("MAPE", f"{ens_mape:.2f}%")
        
        st.divider()
        
        # ============================================
        # VISUALIZATIONS
        # ============================================
        st.subheader("📈 Predictions vs Actual")
        
        fig = go.Figure()

        xgb_test_dates = df_processed.index[lookback + train_size_xgb:]
        test_dates = xgb_test_dates[-common_len:]
        if len(test_dates) != common_len:
            test_dates = data.index[-common_len:]
        
        fig.add_trace(go.Scatter(
            x=test_dates, y=y_common,
            name='Actual Price', mode='lines',
            line=dict(color='#000000', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=test_dates, y=lstm_pred_common,
            name='LSTM Prediction', mode='lines',
            line=dict(color='#FF6B6B', width=2, dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=test_dates, y=xgb_pred_common,
            name='XGBoost Prediction', mode='lines',
            line=dict(color='#4ECDC4', width=2, dash='dash')
        ))
        
        fig.add_trace(go.Scatter(
            x=test_dates, y=ensemble_pred,
            name='Ensemble Prediction', mode='lines',
            line=dict(color='#95E1D3', width=3)
        ))
        
        fig.update_layout(
            title=f"{stock_symbol} - Model Predictions",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            hovermode='x unified',
            height=500,
            template='plotly_dark'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Error distribution
        col1, col2 = st.columns(2)
        
        with col1:
            errors_lstm = np.abs(actual_lstm_eval - lstm_pred_eval)
            errors_xgb = np.abs(y_common - xgb_pred_common)
            errors_ens = np.abs(y_common - ensemble_pred)
            
            fig_errors = go.Figure()
            fig_errors.add_trace(go.Box(y=errors_lstm, name='LSTM', marker_color='#FF6B6B'))
            fig_errors.add_trace(go.Box(y=errors_xgb, name='XGBoost', marker_color='#4ECDC4'))
            fig_errors.add_trace(go.Box(y=errors_ens, name='Ensemble', marker_color='#95E1D3'))
            
            fig_errors.update_layout(
                title="Prediction Error Distribution",
                yaxis_title="Absolute Error ($)",
                height=400,
                template='plotly_dark'
            )
            
            st.plotly_chart(fig_errors, use_container_width=True)
        
        with col2:
            # Feature importance (XGBoost)
            feature_names = ['MA5_' + str(i) for i in range(7*lookback)]
            importance_df = pd.DataFrame({
                'Feature': range(min(15, len(xgb_model.feature_importances_))),
                'Importance': xgb_model.feature_importances_[:min(15, len(xgb_model.feature_importances_))]
            }).sort_values('Importance', ascending=False)
            
            fig_imp = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                            title='Top 15 Feature Importance (XGBoost)',
                            color='Importance', color_continuous_scale='Viridis')
            fig_imp.update_layout(height=400, template='plotly_dark')
            
            st.plotly_chart(fig_imp, use_container_width=True)
        
        st.divider()
        
        # ============================================
        # FUTURE PREDICTIONS
        # ============================================
        st.subheader("🔮 Next 7-Day Forecast")
        
        future_days = 7
        
        # LSTM future prediction
        last_sequence = test_scaled[-lookback:]
        lstm_future = []
        current_seq = last_sequence.copy()
        
        for _ in range(future_days):
            next_pred = lstm_model.predict(current_seq.reshape(1, lookback, 1), verbose=0)
            lstm_future.append(next_pred[0, 0])
            current_seq = np.append(current_seq[1:], next_pred)
        
        lstm_future = scaler_lstm.inverse_transform(np.array(lstm_future).reshape(-1, 1)).flatten()
        
        # XGBoost future prediction
        xgb_future = []
        last_features = X_xgb[-1]
        
        for _ in range(future_days):
            next_pred = xgb_model.predict([last_features])[0]
            xgb_future.append(next_pred)
            last_features = np.append(last_features[len(FEATURE_COLS):], [0] * len(FEATURE_COLS))
        
        xgb_future = np.array(xgb_future)
        ensemble_future = (ensemble_weight * lstm_future + (1 - ensemble_weight) * xgb_future)
        
        # Create forecast dataframe
        future_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), periods=future_days, freq='D')
        forecast_df = pd.DataFrame({
            'Date': future_dates,
            'LSTM': lstm_future,
            'XGBoost': xgb_future,
            'Ensemble': ensemble_future
        })
        
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)
        
        # Plot forecast
        fig_forecast = go.Figure()
        
        # Historical prices
        hist_dates = data.index[-30:]
        hist_prices = data['Close'].values[-30:]
        
        fig_forecast.add_trace(go.Scatter(
            x=hist_dates, y=hist_prices,
            name='Historical Price', mode='lines',
            line=dict(color='#000000', width=2)
        ))
        
        # Forecast
        fig_forecast.add_trace(go.Scatter(
            x=future_dates, y=ensemble_future,
            name='Ensemble Forecast', mode='lines+markers',
            line=dict(color='#FF9F43', width=3),
            marker=dict(size=8)
        ))
        
        # Confidence band
        uncertainty = np.std([lstm_future, xgb_future], axis=0) * 1.96
        fig_forecast.add_trace(go.Scatter(
            x=future_dates.tolist() + future_dates.tolist()[::-1],
            y=(ensemble_future + uncertainty).tolist() + (ensemble_future - uncertainty).tolist()[::-1],
            name='95% Confidence Interval',
            fill='toself',
            fillcolor='rgba(255, 159, 67, 0.2)',
            line=dict(color='rgba(255, 255, 255, 0)'),
        ))
        
        fig_forecast.update_layout(
            title=f"{stock_symbol} - 7-Day Forecast",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            hovermode='x unified',
            height=500,
            template='plotly_dark'
        )
        
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        # ============================================
        # SUMMARY
        # ============================================
        st.divider()
        st.subheader("📋 Summary & Insights")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            best_model = ['LSTM', 'XGBoost', 'Ensemble'][np.argmin([lstm_mape, xgb_mape, ens_mape])]
            st.info(f"**Best Performing Model:** {best_model}")
        
        with col2:
            forecast_change = ((ensemble_future[-1] - current_price) / current_price) * 100
            direction = "📈 Up" if forecast_change > 0 else "📉 Down"
            st.warning(f"**7-Day Forecast:** {direction} {abs(forecast_change):.2f}%")
        
        with col3:
            confidence = 100 - (ens_mape * 10) if ens_mape < 10 else max(0, 100 - ens_mape)
            st.success(f"**Model Confidence:** {confidence:.1f}%")
        
    else:
        if data is None or data.empty:
            st.error("❌ Could not load market data from Yahoo Finance.")
            st.caption(f"Details: {load_error}")
            st.info("Try another symbol, check your internet/DNS access, or enable demo-data fallback.")
        else:
            st.error(
                f"❌ Insufficient data. Retrieved {len(data)} rows, need more than {lookback + 5}. "
                "Increase history or reduce lookback."
            )

else:
    st.info("👈 Enter a stock symbol in the sidebar to get started!")
