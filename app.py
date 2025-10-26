import os
import time
import json
import threading
import logging
from datetime import datetime
from dotenv import load_dotenv
from flask import Flask, jsonify
import pandas as pd
import numpy as np
from binance.spot import Spot as Client
from binance.lib.utils import config_logging
import requests
from sklearn.linear_model import LinearRegression

# Load environment variables first
load_dotenv()

# Configuration dengan fallback values
class Config:
    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
    BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '')
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_ADMIN_ID = os.getenv('TELEGRAM_ADMIN_ID', '')
    INITIAL_BALANCE = float(os.getenv('INITIAL_BALANCE', '5.5'))
    TRADING_SYMBOL = os.getenv('TRADING_SYMBOL', 'BTCUSDT')
    PORT = int(os.getenv('PORT', '5000'))
    RENDER = os.getenv('RENDER', 'false').lower() == 'true'

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_log.txt', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize Binance client dengan error handling
try:
    client = Client(
        Config.BINANCE_API_KEY, 
        Config.BINANCE_SECRET_KEY, 
        base_url='https://api.binance.com'
    )
    logger.info("Binance client initialized successfully")
except Exception as e:
    logger.error("Failed to initialize Binance client: %s", e)
    client = None

# Load configuration files dengan error handling
def load_json_file(filename, default=None):
    """Load JSON file dengan error handling yang robust"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning("Could not load %s: %s. Using default.", filename, e)
        return default or {}

def save_json_file(filename, data):
    """Save JSON file dengan error handling"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error("Failed to save %s: %s", filename, e)
        return False

# Load initial configs dengan default values
DEFAULT_BOT_CONFIG = {
    "ema_short": 9,
    "ema_long": 21,
    "rsi_period": 14,
    "rsi_upper": 70,
    "rsi_lower": 30,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "volume_ratio_threshold": 2.0,
    "take_profit": 0.62,
    "stop_loss": 1.6,
    "trailing_stop_activation": 0.4,
    "trailing_stop": 0.8,
    "position_size": 40,
    "dynamic_threshold": 65,
    "min_notional": 10.0
}

DEFAULT_BOT_STATE = {
    'loss_streak': 0,
    'win_streak': 0,
    'dynamic_threshold': 65,
    'total_trades': 0,
    'winning_trades': 0,
    'is_running': False,
    'last_trades': [],
    'last_analysis_time': None
}

bot_config = load_json_file('bot_config.json', DEFAULT_BOT_CONFIG)
bot_state = load_json_file('bot_state.json', DEFAULT_BOT_STATE)
trade_history = load_json_file('trade_history.json', [])

# Save default config jika file tidak ada
if not os.path.exists('bot_config.json'):
    save_json_file('bot_config.json', bot_config)
if not os.path.exists('bot_state.json'):
    save_json_file('bot_state.json', bot_state)
if not os.path.exists('trade_history.json'):
    save_json_file('trade_history.json', trade_history)

# Flask Health Server
def create_health_server():
    """Membuat Flask server untuk health checks"""
    flask_app = Flask(__name__)
    
    # Suppress Flask logs
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    
    @flask_app.route('/')
    def index():
        return jsonify({
            "status": "running", 
            "service": "binance-trading-bot",
            "timestamp": datetime.now().isoformat()
        })
    
    @flask_app.route('/health')
    def health():
        return jsonify({
            "status": "healthy", 
            "timestamp": datetime.now().isoformat(),
            "bot_running": bot_state.get('is_running', False)
        })
    
    @flask_app.route('/status')
    def status():
        return jsonify({
            "bot_state": bot_state,
            "config": {k: v for k, v in bot_config.items() if not isinstance(v, (float, int)) or k != 'secret_keys'}
        })
    
    def run_server():
        port = Config.PORT
        logger.info("Starting health server on port %d", port)
        if Config.RENDER:
            # Gunakan gunicorn untuk production
            try:
                import gunicorn.app.base
                
                class GunicornApp(gunicorn.app.base.BaseApplication):
                    def __init__(self, app, options=None):
                        self.options = options or {}
                        self.application = app
                        super().__init__()

                    def load_config(self):
                        for key, value in self.options.items():
                            self.cfg.set(key.lower(), value)

                    def load(self):
                        return self.application
                
                options = {
                    'bind': f'0.0.0.0:{port}',
                    'workers': 1,
                    'timeout': 60,
                    'preload_app': True,
                    'accesslog': '-',
                    'errorlog': '-',
                    'loglevel': 'info'
                }
                
                GunicornApp(flask_app, options).run()
                
            except ImportError:
                logger.warning("Gunicorn not available, using Flask development server")
                flask_app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
        else:
            flask_app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    logger.info("Health server thread started")

# Technical Indicators dengan rate limiting
class TechnicalIndicators:
    def __init__(self):
        self.last_request_time = 0
        self.min_interval = 0.5  # 500ms antara requests
        
    def _rate_limit(self):
        """Implement rate limiting untuk menghindari too many requests"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request_time = time.time()
    
    @staticmethod
    def calculate_ema(prices, period):
        if len(prices) < period:
            return prices[-1] if prices else 0
        return pd.Series(prices).ewm(span=period, adjust=False).mean().iloc[-1]
    
    @staticmethod
    def calculate_rsi(prices, period=14):
        if len(prices) <= period:
            return 50
        try:
            delta = pd.Series(prices).diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
        except Exception as e:
            logger.error("RSI calculation error: %s", e)
            return 50
    
    @staticmethod
    def calculate_macd(prices, fast=12, slow=26, signal=9):
        if len(prices) < slow:
            return {'macd': 0, 'signal': 0, 'histogram': 0}
        try:
            exp1 = pd.Series(prices).ewm(span=fast, adjust=False).mean()
            exp2 = pd.Series(prices).ewm(span=slow, adjust=False).mean()
            macd = exp1 - exp2
            signal_line = macd.ewm(span=signal, adjust=False).mean()
            histogram = macd - signal_line
            return {
                'macd': macd.iloc[-1],
                'signal': signal_line.iloc[-1],
                'histogram': histogram.iloc[-1]
            }
        except Exception as e:
            logger.error("MACD calculation error: %s", e)
            return {'macd': 0, 'signal': 0, 'histogram': 0}
    
    @staticmethod
    def calculate_linear_regression_oscillator(prices, period=20):
        if len(prices) < period:
            return 0
        try:
            x = np.array(range(period)).reshape(-1, 1)
            y = prices[-period:]
            model = LinearRegression()
            model.fit(x, y)
            lr_value = model.predict([[period-1]])[0]
            current_price = prices[-1]
            return ((current_price - lr_value) / lr_value) * 100
        except Exception as e:
            logger.error("LRO calculation error: %s", e)
            return 0
    
    @staticmethod
    def calculate_volume_ratio(volumes, period=20):
        if len(volumes) < period:
            return 1.0
        try:
            current_volume = volumes[-1]
            avg_volume = np.mean(volumes[-period:-1])
            return current_volume / avg_volume if avg_volume > 0 else 1.0
        except Exception as e:
            logger.error("Volume ratio calculation error: %s", e)
            return 1.0

# Telegram Handler
class TelegramHandler:
    def __init__(self):
        self.token = Config.TELEGRAM_BOT_TOKEN
        self.admin_id = Config.TELEGRAM_ADMIN_ID
        self.base_url = f"https://api.telegram.org/bot{self.token}"
    
    def send_message(self, text):
        if not self.token or not self.admin_id:
            logger.warning("Telegram credentials not set, skipping message")
            return False
            
        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                "chat_id": self.admin_id,
                "text": text,
                "parse_mode": "HTML"
            }
            response = requests.post(url, data=data, timeout=10)
            if response.status_code != 200:
                logger.error("Telegram API error: %s", response.text)
            return response.status_code == 200
        except Exception as e:
            logger.error("Telegram send error: %s", e)
            return False

# Trading Core dengan improved error handling
class TradingCore:
    def __init__(self):
        self.symbol = Config.TRADING_SYMBOL
        self.telegram = TelegramHandler()
        self.indicators = TechnicalIndicators()
        self.current_position = None
        self.entry_price = 0
        self.highest_price = 0
        self.last_trade_time = 0
        self.min_trade_interval = 60  # Minimum 1 minute between trades
    
    def get_klines(self, interval, limit=100):
        """Get klines data dengan error handling dan rate limiting"""
        if not client:
            logger.error("Binance client not initialized")
            return [], [], [], []
            
        try:
            self.indicators._rate_limit()
            klines = client.klines(symbol=self.symbol, interval=interval, limit=limit)
            closes = [float(k[4]) for k in klines]
            volumes = [float(k[5]) for k in klines]
            highs = [float(k[2]) for k in klines]
            lows = [float(k[3]) for k in klines]
            return closes, volumes, highs, lows
        except Exception as e:
            logger.error("Error getting klines for %s: %s", interval, e)
            return [], [], [], []
    
    def analyze_signals(self):
        """Analyze trading signals dengan comprehensive error handling"""
        try:
            # Get data from multiple timeframes
            closes_15m, volumes_15m, highs_15m, lows_15m = self.get_klines('15m')
            closes_5m, volumes_5m, highs_5m, lows_5m = self.get_klines('5m')
            
            if not closes_15m or not closes_5m:
                logger.warning("No data received from Binance API")
                return None
            
            # Calculate indicators for 15m
            ema_short_15m = self.indicators.calculate_ema(closes_15m, bot_config['ema_short'])
            ema_long_15m = self.indicators.calculate_ema(closes_15m, bot_config['ema_long'])
            rsi_15m = self.indicators.calculate_rsi(closes_15m, bot_config['rsi_period'])
            macd_15m = self.indicators.calculate_macd(
                closes_15m, 
                bot_config['macd_fast'], 
                bot_config['macd_slow'], 
                bot_config['macd_signal']
            )
            lro_15m = self.indicators.calculate_linear_regression_oscillator(closes_15m)
            volume_ratio_15m = self.indicators.calculate_volume_ratio(volumes_15m)
            
            # Calculate indicators for 5m
            ema_short_5m = self.indicators.calculate_ema(closes_5m, bot_config['ema_short'])
            ema_long_5m = self.indicators.calculate_ema(closes_5m, bot_config['ema_long'])
            rsi_5m = self.indicators.calculate_rsi(closes_5m, bot_config['rsi_period'])
            macd_5m = self.indicators.calculate_macd(
                closes_5m, 
                bot_config['macd_fast'], 
                bot_config['macd_slow'], 
                bot_config['macd_signal']
            )
            lro_5m = self.indicators.calculate_linear_regression_oscillator(closes_5m)
            volume_ratio_5m = self.indicators.calculate_volume_ratio(volumes_5m)
            
            # Calculate confidence score (0-100)
            confidence = 0
            
            # EMA alignment (25 points)
            if ema_short_15m > ema_long_15m and ema_short_5m > ema_long_5m:
                confidence += 25
            
            # RSI conditions (25 points)
            if rsi_15m < bot_config['rsi_upper'] and rsi_5m < bot_config['rsi_upper']:
                confidence += 15
            if rsi_15m > bot_config['rsi_lower'] and rsi_5m > bot_config['rsi_lower']:
                confidence += 10
            
            # MACD conditions (25 points)
            if macd_15m['histogram'] > 0 and macd_5m['histogram'] > 0:
                confidence += 15
            if macd_15m['macd'] > macd_15m['signal'] and macd_5m['macd'] > macd_5m['signal']:
                confidence += 10
            
            # LRO and Volume (25 points)
            if lro_15m > 0 and lro_5m > 0:
                confidence += 15
            if volume_ratio_15m > bot_config['volume_ratio_threshold'] or volume_ratio_5m > bot_config['volume_ratio_threshold']:
                confidence += 10
            
            current_price = closes_5m[-1]
            
            return {
                'confidence': confidence,
                'price': current_price,
                'should_buy': confidence >= bot_state['dynamic_threshold'],
                'indicators_15m': {
                    'ema_short': ema_short_15m,
                    'ema_long': ema_long_15m,
                    'rsi': rsi_15m,
                    'macd_histogram': macd_15m['histogram'],
                    'lro': lro_15m,
                    'volume_ratio': volume_ratio_15m
                },
                'indicators_5m': {
                    'ema_short': ema_short_5m,
                    'ema_long': ema_long_5m,
                    'rsi': rsi_5m,
                    'macd_histogram': macd_5m['histogram'],
                    'lro': lro_5m,
                    'volume_ratio': volume_ratio_5m
                }
            }
            
        except Exception as e:
            logger.error("Error in signal analysis: %s", e)
            return None

    def execute_trade(self):
        """Execute trading logic dengan rate limiting"""
        try:
            current_time = time.time()
            
            # Rate limiting untuk trading
            if current_time - self.last_trade_time < self.min_trade_interval:
                return
                
            analysis = self.analyze_signals()
            if not analysis:
                return
            
            current_price = analysis['price']
            
            # Check exit conditions first
            if self.current_position:
                exit_reason = self.check_exit_conditions(current_price)
                if exit_reason:
                    self.close_position(current_price, exit_reason)
                    self.last_trade_time = current_time
                    return
            
            # Check entry conditions
            if not self.current_position and analysis['should_buy']:
                self.open_position(current_price, analysis)
                self.last_trade_time = current_time
                
        except Exception as e:
            logger.error("Trade execution error: %s", e)

    # ... (rest of TradingCore methods remain the same as previous version)

# Feedback System
class FeedbackSystem:
    def __init__(self):
        self.telegram = TelegramHandler()
    
    def update_parameters(self):
        """Update trading parameters berdasarkan performance"""
        global bot_state, bot_config
        
        try:
            # Check loss streak condition
            if bot_state['loss_streak'] >= 2:
                old_threshold = bot_state['dynamic_threshold']
                bot_state['dynamic_threshold'] += 8
                self.telegram.send_message(
                    f"📈 Increased dynamic threshold: {old_threshold}% → {bot_state['dynamic_threshold']}% "
                    f"(Loss streak: {bot_state['loss_streak']})"
                )
            
            # Check win streak condition
            elif bot_state['win_streak'] >= 3:
                old_threshold = bot_state['dynamic_threshold']
                bot_state['dynamic_threshold'] = max(50, bot_state['dynamic_threshold'] - 5)
                self.telegram.send_message(
                    f"📉 Decreased dynamic threshold: {old_threshold}% → {bot_state['dynamic_threshold']}% "
                    f"(Win streak: {bot_state['win_streak']})"
                )
            
            # Check win rate condition
            recent_trades = bot_state['last_trades']
            if len(recent_trades) >= 5:
                win_rate = sum(recent_trades) / len(recent_trades) * 100
                if win_rate < 20:
                    bot_state['is_running'] = False
                    self.telegram.send_message(
                        f"⏸️ Trading paused for 30 minutes (Win rate: {win_rate:.1f}% < 20%)"
                    )
                    # Schedule restart after 30 minutes
                    threading.Timer(1800, self.restart_trading).start()
            
            save_json_file('bot_state.json', bot_state)
            
        except Exception as e:
            logger.error("Error in feedback system: %s", e)

    def restart_trading(self):
        """Restart trading setelah pause"""
        bot_state['is_running'] = True
        save_json_file('bot_state.json', bot_state)
        self.telegram.send_message("🔄 Trading resumed after 30 minute pause")

# Main Application
def main():
    """Main application entry point"""
    logger.info("Starting Binance Trading Bot on Render...")
    
    # Validate required environment variables
    required_vars = ['BINANCE_API_KEY', 'BINANCE_SECRET_KEY']
    missing_vars = [var for var in required_vars if not getattr(Config, var)]
    
    if missing_vars:
        logger.error("Missing required environment variables: %s", missing_vars)
        return
    
    # Start health server
    create_health_server()
    
    # Initialize components
    trading_core = TradingCore()
    feedback_system = FeedbackSystem()
    telegram_handler = TelegramHandler()
    
    # Send startup message
    telegram_handler.send_message("🚀 <b>Binance Trading Bot Started on Render!</b>")
    
    # Rate limiting configuration
    trade_interval = 60  # Check every minute
    last_feedback_update = 0
    feedback_interval = 300  # Update feedback every 5 minutes
    
    logger.info("Bot main loop started with trade interval: %ds", trade_interval)
    
    try:
        while True:
            if bot_state['is_running']:
                # Execute trading logic
                trading_core.execute_trade()
                
                # Update feedback system periodically
                current_time = time.time()
                if current_time - last_feedback_update >= feedback_interval:
                    feedback_system.update_parameters()
                    last_feedback_update = current_time
            
            # Sleep to avoid rate limiting
            time.sleep(trade_interval)
            
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
        telegram_handler.send_message("🛑 <b>Bot stopped manually</b>")
    except Exception as e:
        logger.error("Unexpected error in main loop: %s", e)
        telegram_handler.send_message(f"❌ <b>Bot crashed:</b> {str(e)}")

if __name__ == "__main__":
    main()
