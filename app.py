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

# Load environment variables
load_dotenv()

# Configuration
class Config:
    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
    BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    TELEGRAM_ADMIN_ID = os.getenv('TELEGRAM_ADMIN_ID')
    INITIAL_BALANCE = float(os.getenv('INITIAL_BALANCE', 5.5))
    TRADING_SYMBOL = os.getenv('TRADING_SYMBOL', 'BTCUSDT')
    PORT = int(os.getenv('PORT', 5000))

# Initialize Binance client
client = Client(Config.BINANCE_API_KEY, Config.BINANCE_SECRET_KEY, base_url='https://api.binance.com')

# Load configuration files
def load_json_file(filename, default=None):
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return default or {}

def save_json_file(filename, data):
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2)

# Load initial configs
bot_config = load_json_file('bot_config.json')
bot_state = load_json_file('bot_state.json', {
    'loss_streak': 0,
    'win_streak': 0,
    'dynamic_threshold': 65,
    'total_trades': 0,
    'winning_trades': 0,
    'is_running': False,
    'last_trades': []
})
trade_history = load_json_file('trade_history.json', [])

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_log.txt'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Flask Health Server
def create_health_server():
    flask_app = Flask(__name__)
    
    # Suppress Flask logs
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    
    @flask_app.route('/')
    def index():
        return jsonify({"status": "running", "bot_status": "active"})
    
    @flask_app.route('/health')
    def health():
        return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})
    
    def run_server():
        flask_app.run(host='0.0.0.0', port=Config.PORT, debug=False, use_reloader=False)
    
    server_thread = threading.Thread(target=run_server, daemon=True)
    server_thread.start()
    logger.info("Health server started on port %s", Config.PORT)

# Technical Indicators
class TechnicalIndicators:
    @staticmethod
    def calculate_ema(prices, period):
        return pd.Series(prices).ewm(span=period, adjust=False).mean().iloc[-1]
    
    @staticmethod
    def calculate_rsi(prices, period=14):
        delta = pd.Series(prices).diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
    
    @staticmethod
    def calculate_macd(prices, fast=12, slow=26, signal=9):
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
    
    @staticmethod
    def calculate_linear_regression_oscillator(prices, period=20):
        if len(prices) < period:
            return 0
        x = np.array(range(period)).reshape(-1, 1)
        y = prices[-period:]
        model = LinearRegression()
        model.fit(x, y)
        lr_value = model.predict([[period-1]])[0]
        current_price = prices[-1]
        return ((current_price - lr_value) / lr_value) * 100
    
    @staticmethod
    def calculate_volume_ratio(volumes, period=20):
        if len(volumes) < period:
            return 1.0
        current_volume = volumes[-1]
        avg_volume = np.mean(volumes[-period:-1])
        return current_volume / avg_volume if avg_volume > 0 else 1.0

# Telegram Handler
class TelegramHandler:
    def __init__(self):
        self.token = Config.TELEGRAM_BOT_TOKEN
        self.admin_id = Config.TELEGRAM_ADMIN_ID
        self.base_url = f"https://api.telegram.org/bot{self.token}"
    
    def send_message(self, text):
        try:
            url = f"{self.base_url}/sendMessage"
            data = {
                "chat_id": self.admin_id,
                "text": text,
                "parse_mode": "HTML"
            }
            response = requests.post(url, data=data, timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error("Telegram send error: %s", e)
            return False
    
    def process_webhook(self, update):
        try:
            message = update.get('message', {})
            text = message.get('text', '')
            chat_id = message.get('chat', {}).get('id')
            
            if str(chat_id) != self.admin_id:
                return {"status": "unauthorized"}
            
            if text.startswith('/'):
                return self.handle_command(text[1:].lower())
                
        except Exception as e:
            logger.error("Webhook processing error: %s", e)
        
        return {"status": "processed"}
    
    def handle_command(self, command):
        global bot_state
        
        if command == 'start':
            bot_state['is_running'] = True
            save_json_file('bot_state.json', bot_state)
            self.send_message("🤖 <b>Bot Trading Started</b>")
            return {"status": "started"}
            
        elif command == 'stop':
            bot_state['is_running'] = False
            save_json_file('bot_state.json', bot_state)
            self.send_message("🛑 <b>Bot Trading Stopped</b>")
            return {"status": "stopped"}
            
        elif command == 'status':
            status_text = self.get_status_text()
            self.send_message(status_text)
            return {"status": "status_sent"}
            
        elif command == 'log':
            log_text = self.get_recent_logs()
            self.send_message(f"<code>{log_text}</code>")
            return {"status": "log_sent"}
            
        elif command == 'config':
            config_text = self.get_config_text()
            self.send_message(f"<code>{config_text}</code>")
            return {"status": "config_sent"}
            
        elif command == 'ip':
            ip = self.get_public_ip()
            self.send_message(f"🌐 <b>Public IP:</b> <code>{ip}</code>")
            return {"status": "ip_sent"}
        
        return {"status": "unknown_command"}
    
    def get_status_text(self):
        win_rate = (bot_state['winning_trades'] / bot_state['total_trades'] * 100) if bot_state['total_trades'] > 0 else 0
        return f"""📊 <b>Bot Status</b>

🔧 State: {'🟢 RUNNING' if bot_state['is_running'] else '🔴 STOPPED'}
💰 Balance: ${Config.INITIAL_BALANCE:.2f}
📈 Total Trades: {bot_state['total_trades']}
✅ Winning Trades: {bot_state['winning_trades']}
📊 Win Rate: {win_rate:.1f}%
🔥 Win Streak: {bot_state['win_streak']}
💔 Loss Streak: {bot_state['loss_streak']}
🎯 Dynamic Threshold: {bot_state['dynamic_threshold']}%
"""

    def get_recent_logs(self):
        try:
            with open('trading_log.txt', 'r') as f:
                lines = f.readlines()[-10:]
            return ''.join(lines) if lines else "No logs available"
        except FileNotFoundError:
            return "Log file not found"
    
    def get_config_text(self):
        config_str = json.dumps(bot_config, indent=2)
        return f"Bot Configuration:\n{config_str}"
    
    def get_public_ip(self):
        try:
            response = requests.get('https://api.ipify.org', timeout=5)
            return response.text
        except:
            return "Unable to determine IP"

# Trading Core
class TradingCore:
    def __init__(self):
        self.symbol = Config.TRADING_SYMBOL
        self.telegram = TelegramHandler()
        self.indicators = TechnicalIndicators()
        self.current_position = None
        self.entry_price = 0
        self.highest_price = 0
    
    def get_klines(self, interval, limit=100):
        try:
            klines = client.klines(symbol=self.symbol, interval=interval, limit=limit)
            closes = [float(k[4]) for k in klines]
            volumes = [float(k[5]) for k in klines]
            highs = [float(k[2]) for k in klines]
            lows = [float(k[3]) for k in klines]
            return closes, volumes, highs, lows
        except Exception as e:
            logger.error("Error getting klines: %s", e)
            return [], [], [], []
    
    def analyze_signals(self):
        # Get data from multiple timeframes
        closes_15m, volumes_15m, highs_15m, lows_15m = self.get_klines('15m')
        closes_5m, volumes_5m, highs_5m, lows_5m = self.get_klines('5m')
        
        if not closes_15m or not closes_5m:
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
    
    def calculate_position_size(self):
        balance = Config.INITIAL_BALANCE
        position_percent = bot_config['position_size'] / 100
        
        # Adjust position size based on streaks
        if bot_state['loss_streak'] >= 2:
            position_percent *= 0.7  # Reduce 30%
        elif bot_state['win_streak'] >= 3:
            position_percent *= 1.15  # Increase 15%
        
        position_size = balance * position_percent
        
        # Ensure minimum notional
        if position_size < bot_config['min_notional']:
            position_size = bot_config['min_notional']
        
        return min(position_size, balance * 0.95)  # Maximum 95% of balance
    
    def check_exit_conditions(self, current_price):
        if not self.current_position:
            return None
        
        # Calculate P&L percentage
        pnl_percent = ((current_price - self.entry_price) / self.entry_price) * 100
        
        # Update highest price for trailing stop
        if current_price > self.highest_price:
            self.highest_price = current_price
        
        # Take Profit
        if pnl_percent >= bot_config['take_profit']:
            return 'TP'
        
        # Stop Loss
        if pnl_percent <= -bot_config['stop_loss']:
            return 'SL'
        
        # Trailing Stop
        if self.highest_price > self.entry_price * (1 + bot_config['trailing_stop_activation'] / 100):
            trailing_stop_price = self.highest_price * (1 - bot_config['trailing_stop'] / 100)
            if current_price <= trailing_stop_price:
                return 'TS'
        
        return None
    
    def execute_trade(self):
        try:
            analysis = self.analyze_signals()
            if not analysis:
                return
            
            current_price = analysis['price']
            
            # Check exit conditions first
            if self.current_position:
                exit_reason = self.check_exit_conditions(current_price)
                if exit_reason:
                    self.close_position(current_price, exit_reason)
                    return
            
            # Check entry conditions
            if not self.current_position and analysis['should_buy']:
                self.open_position(current_price, analysis)
                
        except Exception as e:
            logger.error("Trade execution error: %s", e)
    
    def open_position(self, price, analysis):
        position_size = self.calculate_position_size()
        quantity = position_size / price
        
        # In real trading, you would place actual order here
        # For safety, we'll just simulate for now
        self.current_position = {
            'entry_price': price,
            'quantity': quantity,
            'size': position_size,
            'timestamp': datetime.now().isoformat()
        }
        self.entry_price = price
        self.highest_price = price
        
        # Log trade
        trade_data = {
            'action': 'BUY',
            'price': price,
            'quantity': quantity,
            'size': position_size,
            'timestamp': self.current_position['timestamp'],
            'confidence': analysis['confidence']
        }
        trade_history.append(trade_data)
        save_json_file('trade_history.json', trade_history)
        
        # Send notification
        message = f"""🎯 <b>POSITION OPENED</b>

💰 Size: ${position_size:.2f}
📊 Quantity: {quantity:.6f}
💵 Price: ${price:.2f}
🎯 Confidence: {analysis['confidence']}%
📈 15m RSI: {analysis['indicators_15m']['rsi']:.1f}
📊 5m RSI: {analysis['indicators_5m']['rsi']:.1f}
"""
        self.telegram.send_message(message)
        logger.info("Position opened at $%.2f", price)
    
    def close_position(self, price, reason):
        if not self.current_position:
            return
        
        position_size = self.current_position['size']
        pnl_percent = ((price - self.entry_price) / self.entry_price) * 100
        pnl_amount = position_size * (pnl_percent / 100)
        
        # Update bot state
        is_win = pnl_percent > 0
        bot_state['total_trades'] += 1
        
        if is_win:
            bot_state['winning_trades'] += 1
            bot_state['win_streak'] += 1
            bot_state['loss_streak'] = 0
        else:
            bot_state['loss_streak'] += 1
            bot_state['win_streak'] = 0
        
        # Update last trades for win rate calculation
        bot_state['last_trades'] = bot_state['last_trades'][-4:] + [is_win]
        
        # Log trade
        trade_data = {
            'action': 'SELL',
            'price': price,
            'pnl_percent': pnl_percent,
            'pnl_amount': pnl_amount,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        }
        trade_history.append(trade_data)
        save_json_file('trade_history.json', trade_history)
        
        # Send notification
        emoji = "🟢" if is_win else "🔴"
        message = f"""🏁 <b>POSITION CLOSED</b> {emoji}

💵 Exit Price: ${price:.2f}
📊 P&L: {pnl_percent:+.2f}% (${pnl_amount:+.2f})
🎯 Reason: {reason}
🔥 Win Streak: {bot_state['win_streak']}
💔 Loss Streak: {bot_state['loss_streak']}
"""
        self.telegram.send_message(message)
        logger.info("Position closed at $%.2f, P&L: %.2f%%, Reason: %s", price, pnl_percent, reason)
        
        # Reset position
        self.current_position = None
        self.entry_price = 0
        self.highest_price = 0

# Feedback System
class FeedbackSystem:
    def __init__(self):
        self.telegram = TelegramHandler()
    
    def update_parameters(self):
        global bot_state, bot_config
        
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
    
    def restart_trading(self):
        bot_state['is_running'] = True
        save_json_file('bot_state.json', bot_state)
        self.telegram.send_message("🔄 Trading resumed after 30 minute pause")

# Main Application
def main():
    logger.info("Starting Binance Trading Bot...")
    
    # Start health server
    create_health_server()
    
    # Initialize components
    trading_core = TradingCore()
    feedback_system = FeedbackSystem()
    telegram_handler = TelegramHandler()
    
    # Send startup message
    telegram_handler.send_message("🚀 <b>Binance Trading Bot Started Successfully!</b>")
    
    # Rate limiting - avoid too many requests
    trade_interval = 60  # Check every minute
    last_feedback_update = 0
    feedback_interval = 300  # Update feedback every 5 minutes
    
    logger.info("Bot main loop started")
    
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
