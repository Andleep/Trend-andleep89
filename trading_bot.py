import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import warnings
import random
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class TradingStrategy(Enum):
    CONSERVATIVE = "محافظ"
    MODERATE = "مستمر" 
    AGGRESSIVE = "عدواني"

@dataclass
class TradingConfig:
    """إعدادات التداول المحسنة"""
    initial_capital: float = 1000.0
    risk_per_trade: float = 0.02
    max_position_size: float = 0.3
    selected_pairs: List[str] = None
    strategy: TradingStrategy = TradingStrategy.MODERATE
    min_trade_amount: float = 1.0
    
    def __post_init__(self):
        if self.selected_pairs is None:
            self.selected_pairs = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]

class ImprovedTradingBot:
    """
    بوت تداول محسن مع ربح تراكمي فوري
    """
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.initial_balance = config.initial_capital
        self.current_balance = config.initial_capital
        self.positions = {}
        self.trade_history = []
        
        # إحصائيات محسنة
        self.real_time_stats = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_profit': 0.0,
            'max_drawdown': 0.0,
            'current_streak': 0,
            'max_win_streak': 0,
            'max_loss_streak': 0,
            'compounded_growth': 0.0,
            'equity_curve': [config.initial_capital],
            'consecutive_losses': 0
        }
        
        logger.info(f"🚀 البوت المحسن جاهز | الرصيد: ${self.current_balance:.2f}")
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """مؤشرات فنية متقدمة"""
        df = df.copy()
        
        # المتوسطات المتحركة
        for period in [5, 10, 20]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        
        # RSI
        for period in [7, 14]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # بولنجر باندز
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # مؤشرات مخصصة
        df['trend_up'] = (df['ema_5'] > df['ema_10']) & (df['ema_10'] > df['ema_20'])
        df['trend_down'] = (df['ema_5'] < df['ema_10']) & (df['ema_10'] < df['ema_20'])
        df['momentum'] = df['close'] / df['close'].shift(5) - 1
        
        return df
    
    def generate_trading_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """إشارات تداول ذكية"""
        df = df.copy()
        
        # نظام تصويت متقدم
        df['buy_score'] = 0
        df['sell_score'] = 0
        
        # شروط الشراء
        df.loc[df['trend_up'], 'buy_score'] += 2
        df.loc[df['rsi_14'] < 35, 'buy_score'] += 2
        df.loc[df['macd_hist'] > 0, 'buy_score'] += 1
        df.loc[df['close'] < df['bb_lower'], 'buy_score'] += 2
        df.loc[df['momentum'] > 0.01, 'buy_score'] += 1
        
        # شروط البيع
        df.loc[df['trend_down'], 'sell_score'] += 2
        df.loc[df['rsi_14'] > 65, 'sell_score'] += 2
        df.loc[df['macd_hist'] < 0, 'sell_score'] += 1
        df.loc[df['close'] > df['bb_upper'], 'sell_score'] += 2
        df.loc[df['momentum'] < -0.01, 'sell_score'] += 1
        
        # قرارات التداول
        df['buy_signal'] = df['buy_score'] >= 5
        df['sell_signal'] = df['sell_score'] >= 5
        
        return df
    
    def calculate_position_size(self, current_price: float, confidence: float) -> float:
        """حجم مركز ذكي للمبالغ الصغيرة"""
        if self.current_balance < 10:
            # صفقات صغيرة جداً لرأس المال الصغير
            min_trade = max(0.5, self.current_balance * 0.1)
        else:
            min_trade = self.config.min_trade_amount
        
        risk_amount = self.current_balance * self.config.risk_per_trade * confidence
        position_value = max(risk_amount * 1.5, min_trade)
        max_position_value = self.current_balance * self.config.max_position_size
        position_value = min(position_value, max_position_value)
        
        quantity = position_value / current_price if current_price > 0 else 0
        return round(quantity, 6)
    
    def execute_compounding(self, profit: float, trade_info: Dict):
        """نظام ربح تراكمي فوري"""
        old_balance = self.current_balance
        self.current_balance += profit
        
        self.real_time_stats['total_profit'] += profit
        self.real_time_stats['total_trades'] += 1
        
        if profit > 0:
            self.real_time_stats['winning_trades'] += 1
            self.real_time_stats['current_streak'] = max(self.real_time_stats['current_streak'] + 1, 0)
            self.real_time_stats['consecutive_losses'] = 0
            
            growth_rate = profit / old_balance
            self.real_time_stats['compounded_growth'] = (
                (1 + self.real_time_stats['compounded_growth']) * (1 + growth_rate) - 1
            )
        else:
            self.real_time_stats['losing_trades'] += 1
            self.real_time_stats['current_streak'] = min(self.real_time_stats['current_streak'] - 1, 0)
            self.real_time_stats['consecutive_losses'] += 1
        
        self.real_time_stats['equity_curve'].append(self.current_balance)
        
        peak = max(self.real_time_stats['equity_curve'])
        current_drawdown = (peak - self.current_balance) / peak * 100
        self.real_time_stats['max_drawdown'] = max(
            self.real_time_stats['max_drawdown'],
            current_drawdown
        )
    
    def run_backtest(self, market_data: Dict, symbols: List[str]) -> Dict:
        """المحاكاة الرئيسية - هذا ما تبحث عنه app.py"""
        results = {}
        
        logger.info(f"🎯 بدء المحاكاة المحسنة على {len(symbols)} عملات")
        
        for symbol in symbols:
            if symbol not in market_data:
                continue
                
            df = market_data[symbol].copy()
            df = self.calculate_technical_indicators(df)
            df = self.generate_trading_signals(df)
            
            symbol_trades = []
            
            for i, (timestamp, row) in enumerate(df.iterrows()):
                if i < 20:
                    continue
                    
                current_price = row['close']
                
                # دخول صفقة شراء
                if (row['buy_signal'] and symbol not in self.positions and 
                    self.current_balance > 1.0):
                    
                    confidence = min(1.0, row['buy_score'] / 8.0)
                    quantity = self.calculate_position_size(current_price, confidence)
                    
                    if quantity * current_price >= 0.5:  # صفقة بحد أدنى 50 سنت
                        self.positions[symbol] = {
                            'entry_time': timestamp,
                            'entry_price': current_price,
                            'quantity': quantity,
                            'investment': quantity * current_price,
                            'stop_loss': current_price * 0.98,
                            'take_profit': current_price * 1.03,
                            'type': 'LONG'
                        }
                        
                        self.current_balance -= quantity * current_price
                        
                        trade_record = {
                            'symbol': symbol,
                            'action': 'BUY',
                            'timestamp': timestamp,
                            'price': current_price,
                            'quantity': quantity,
                            'amount': quantity * current_price,
                            'status': 'OPEN'
                        }
                        
                        symbol_trades.append(trade_record)
                        self.trade_history.append(trade_record)
                
                # إغلاق الصفقات
                elif symbol in self.positions:
                    position = self.positions[symbol]
                    close_trade = False
                    
                    if row['sell_signal']:
                        close_trade = True
                        reason = "إشارة بيع"
                    elif current_price <= position['stop_loss']:
                        close_trade = True
                        reason = "وقف خسارة"
                    elif current_price >= position['take_profit']:
                        close_trade = True
                        reason = "جني أرباح"
                    
                    if close_trade:
                        profit = (current_price - position['entry_price']) * position['quantity']
                        
                        self.execute_compounding(profit, {
                            'symbol': symbol,
                            'position': position,
                            'exit_price': current_price,
                            'reason': reason
                        })
                        
                        trade_record = {
                            'symbol': symbol,
                            'action': 'SELL',
                            'timestamp': timestamp,
                            'price': current_price,
                            'quantity': position['quantity'],
                            'amount': position['investment'],
                            'profit': profit,
                            'profit_pct': (profit / position['investment']) * 100,
                            'status': 'CLOSED',
                            'close_reason': reason
                        }
                        
                        symbol_trades.append(trade_record)
                        self.trade_history.append(trade_record)
                        del self.positions[symbol]
            
            results[symbol] = {
                'trades': symbol_trades,
                'total_trades': len([t for t in symbol_trades if t.get('status') == 'CLOSED']),
                'profitable_trades': len([t for t in symbol_trades if t.get('profit', 0) > 0])
            }
        
        # حساب النتائج النهائية
        return self.calculate_final_metrics()
    
    def calculate_final_metrics(self) -> Dict:
        """حساب المقاييس النهائية"""
        total_profit = self.current_balance - self.initial_balance
        total_return = (total_profit / self.initial_balance) * 100
        
        closed_trades = [t for t in self.trade_history if t.get('status') == 'CLOSED']
        winning_trades = [t for t in closed_trades if t.get('profit', 0) > 0]
        win_rate = (len(winning_trades) / len(closed_trades)) * 100 if closed_trades else 0
        
        gross_profit = sum(t.get('profit', 0) for t in winning_trades)
        gross_loss = abs(sum(t.get('profit', 0) for t in closed_trades if t.get('profit', 0) < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        avg_profit = np.mean([t.get('profit', 0) for t in closed_trades]) if closed_trades else 0
        
        # حساب تقييم الأداء
        performance_score = min(100, max(0, (
            win_rate * 0.4 +
            min(total_return * 2, 40) +
            min(profit_factor * 20, 20)
        )))
        
        final_metrics = {
            'initial_balance': self.initial_balance,
            'final_balance': self.current_balance,
            'total_profit': total_profit,
            'total_return': total_return,
            'total_trades': len(closed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(closed_trades) - len(winning_trades),
            'win_rate': win_rate,
            'max_drawdown': self.real_time_stats['max_drawdown'],
            'compounded_growth': self.real_time_stats['compounded_growth'] * 100,
            'profit_factor': profit_factor,
            'avg_profit': avg_profit,
            'performance_score': performance_score
        }
        
        logger.info(f"🎊 انتهت المحاكاة | الربح: ${total_profit:.2f} ({total_return:.2f}%)")
        
        return {
            'symbol_results': {},
            'performance_metrics': final_metrics,
            'trade_history': self.trade_history,
            'equity_curve': self.real_time_stats['equity_curve']
        }
