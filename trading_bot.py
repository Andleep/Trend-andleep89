import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class TradingStrategy(Enum):
    CONSERVATIVE = "محافظ"
    MODERATE = "مستمر" 
    AGGRESSIVE = "عدواني"

@dataclass
class TradingConfig:
    """إعدادات التداول"""
    initial_capital: float = 1000.0
    risk_per_trade: float = 0.02
    max_position_size: float = 0.3
    selected_pairs: List[str] = None
    strategy: TradingStrategy = TradingStrategy.MODERATE
    
    def __post_init__(self):
        if self.selected_pairs is None:
            self.selected_pairs = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]

class AdvancedTradingBot:
    """
    بوت تداول متقدم مع ربح تراكمي فوري
    """
    
    def __init__(self, config: TradingConfig):
        self.config = config
        self.initial_balance = config.initial_capital
        self.current_balance = config.initial_capital
        self.positions = {}
        self.trade_history = []
        self.performance_metrics = {}
        
        # إحصائيات الوقت الحقيقي
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
            'equity_curve': [config.initial_capital]
        }
        
        logger.info(f"🚀 البوت المتقدم جاهز | الرصيد: ${self.current_balance:.2f}")
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """حساب المؤشرات الفنية المتقدمة"""
        df = df.copy()
        
        # المتوسطات المتحركة
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # بولنجر باندز
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        # ستوكاستك
        low_14 = df['low'].rolling(14).min()
        high_14 = df['high'].rolling(14).max()
        df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        
        # ATR (مدى التداول الحقيقي)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        df['atr'] = true_range.rolling(14).mean()
        
        # إشارات التداول
        df['trend_up'] = df['sma_20'] > df['sma_50']
        df['rsi_oversold'] = df['rsi'] < 30
        df['rsi_overbought'] = df['rsi'] > 70
        df['macd_bullish'] = df['macd'] > df['macd_signal']
        df['bb_buy_signal'] = df['close'] < df['bb_lower']
        df['bb_sell_signal'] = df['close'] > df['bb_upper']
        df['stoch_oversold'] = df['stoch_k'] < 20
        df['stoch_overbought'] = df['stoch_k'] > 80
        
        return df
    
    def generate_trading_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """توليد إشارات التداول الذكية"""
        df = df.copy()
        
        # نظام تصويت للإشارات
        df['buy_signals'] = 0
        df['sell_signals'] = 0
        
        # إشارات الشراء
        df.loc[df['trend_up'], 'buy_signals'] += 1
        df.loc[df['rsi_oversold'], 'buy_signals'] += 2
        df.loc[df['macd_bullish'], 'buy_signals'] += 1
        df.loc[df['bb_buy_signal'], 'buy_signals'] += 2
        df.loc[df['stoch_oversold'], 'buy_signals'] += 1
        
        # إشارات البيع
        df.loc[~df['trend_up'], 'sell_signals'] += 1
        df.loc[df['rsi_overbought'], 'sell_signals'] += 2
        df.loc[~df['macd_bullish'], 'sell_signals'] += 1
        df.loc[df['bb_sell_signal'], 'sell_signals'] += 2
        df.loc[df['stoch_overbought'], 'sell_signals'] += 1
        
        # قرارات التداول النهائية
        df['buy_signal'] = df['buy_signals'] >= 4  # عتبة الشراء
        df['sell_signal'] = df['sell_signals'] >= 4  # عتبة البيع
        
        return df
    
    def calculate_dynamic_position_size(self, current_price: float, confidence: float) -> float:
        """حجم المركز الديناميكي مع إدارة المخاطر"""
        # حساب المخاطرة بناءً على الثقة والأداء
        base_risk = self.config.risk_per_trade
        
        # تعديل المخاطرة بناءً على الأداء
        if self.real_time_stats['current_streak'] > 0:
            risk_multiplier = min(1.5, 1 + (self.real_time_stats['current_streak'] * 0.1))
        else:
            risk_multiplier = max(0.5, 1 + (self.real_time_stats['current_streak'] * 0.05))
        
        adjusted_risk = base_risk * risk_multiplier * confidence
        
        # حساب حجم المركز
        risk_amount = self.current_balance * adjusted_risk
        max_position_value = self.current_balance * self.config.max_position_size
        
        position_value = min(risk_amount * 2, max_position_value)  # رافعة 2:1
        quantity = position_value / current_price if current_price > 0 else 0
        
        return quantity
    
    def execute_instant_compounding(self, profit: float, trade_info: Dict):
        """نظام الربح التراكمي الفوري"""
        old_balance = self.current_balance
        self.current_balance += profit
        
        # تحديث الإحصائيات
        self.real_time_stats['total_profit'] += profit
        self.real_time_stats['total_trades'] += 1
        
        if profit > 0:
            self.real_time_stats['winning_trades'] += 1
            self.real_time_stats['current_streak'] = max(self.real_time_stats['current_streak'] + 1, 0)
            self.real_time_stats['max_win_streak'] = max(
                self.real_time_stats['max_win_streak'],
                self.real_time_stats['current_streak']
            )
            
            # حساب النمو التراكمي
            growth_rate = profit / old_balance
            self.real_time_stats['compounded_growth'] = (
                (1 + self.real_time_stats['compounded_growth']) * (1 + growth_rate) - 1
            )
            
            logger.info(f"💰 ربح تراكمي فوري: +${profit:.2f} | الرصيد: ${self.current_balance:.2f}")
        else:
            self.real_time_stats['losing_trades'] += 1
            self.real_time_stats['current_streak'] = min(self.real_time_stats['current_streak'] - 1, 0)
            self.real_time_stats['max_loss_streak'] = min(
                self.real_time_stats['max_loss_streak'],
                self.real_time_stats['current_streak']
            )
        
        # تحديث منحنى رأس المال
        self.real_time_stats['equity_curve'].append(self.current_balance)
        
        # تحديث أقصى خسارة
        peak = max(self.real_time_stats['equity_curve'])
        current_drawdown = (peak - self.current_balance) / peak * 100
        self.real_time_stats['max_drawdown'] = max(
            self.real_time_stats['max_drawdown'],
            current_drawdown
        )
    
    def run_backtest(self, market_data: Dict, symbols: List[str]) -> Dict:
        """تشغيل محاكاة كاملة"""
        results = {}
        
        logger.info(f"🎯 بدء المحاكاة على {len(symbols)} عملات")
        
        for symbol in symbols:
            if symbol not in market_data:
                continue
                
            logger.info(f"🔍 تحليل {symbol}...")
            df = market_data[symbol].copy()
            
            # حساب المؤشرات وإشارات التداول
            df = self.calculate_technical_indicators(df)
            df = self.generate_trading_signals(df)
            
            symbol_trades = []
            
            for i, (timestamp, row) in enumerate(df.iterrows()):
                current_price = row['close']
                atr_value = row.get('atr', current_price * 0.02)
                
                # فتح صفقة شراء
                if (row['buy_signal'] and symbol not in self.positions and 
                    self.current_balance > 10):  # حد أدنى للرصيد
                    
                    confidence = min(1.0, row['buy_signals'] / 8.0)  # ثقة من 0 إلى 1
                    quantity = self.calculate_dynamic_position_size(current_price, confidence)
                    
                    if quantity > 0:
                        # حساب وقف الخسارة وجني الأرباح
                        stop_loss = current_price - (2 * atr_value)
                        take_profit = current_price + (4 * atr_value)
                        
                        self.positions[symbol] = {
                            'entry_time': timestamp,
                            'entry_price': current_price,
                            'quantity': quantity,
                            'investment': quantity * current_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'type': 'LONG'
                        }
                        
                        # خصم المبلغ من الرصيد
                        self.current_balance -= quantity * current_price
                        
                        trade_record = {
                            'symbol': symbol,
                            'action': 'BUY',
                            'timestamp': timestamp,
                            'price': current_price,
                            'quantity': quantity,
                            'amount': quantity * current_price,
                            'confidence': confidence,
                            'status': 'OPEN'
                        }
                        
                        symbol_trades.append(trade_record)
                        self.trade_history.append(trade_record)
                
                # إغلاق الصفقات
                elif symbol in self.positions:
                    position = self.positions[symbol]
                    
                    # شروط الإغلاق
                    close_trade = False
                    close_reason = ""
                    
                    if row['sell_signal']:
                        close_trade = True
                        close_reason = "إشارة بيع"
                    elif current_price <= position['stop_loss']:
                        close_trade = True
                        close_reason = "وقف خسارة"
                    elif current_price >= position['take_profit']:
                        close_trade = True
                        close_reason = "جني أرباح"
                    
                    if close_trade:
                        profit = (current_price - position['entry_price']) * position['quantity']
                        
                        # تطبيق الربح التراكمي الفوري
                        self.execute_instant_compounding(profit, {
                            'symbol': symbol,
                            'position': position,
                            'exit_price': current_price,
                            'reason': close_reason
                        })
                        
                        # تسجيل الصفقة المغلقة
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
                            'close_reason': close_reason
                        }
                        
                        symbol_trades.append(trade_record)
                        self.trade_history.append(trade_record)
                        
                        # إزالة المركز
                        del self.positions[symbol]
            
            results[symbol] = {
                'trades': symbol_trades,
                'total_trades': len([t for t in symbol_trades if t.get('status') == 'CLOSED']),
                'profitable_trades': len([t for t in symbol_trades if t.get('profit', 0) > 0])
            }
        
        # حساب المقاييس النهائية
        total_profit = self.current_balance - self.initial_balance
        total_return = (total_profit / self.initial_balance) * 100
        
        closed_trades = [t for t in self.trade_history if t.get('status') == 'CLOSED']
        winning_trades = [t for t in closed_trades if t.get('profit', 0) > 0]
        win_rate = (len(winning_trades) / len(closed_trades)) * 100 if closed_trades else 0
        
        # حساب عامل الربح
        gross_profit = sum(t.get('profit', 0) for t in winning_trades)
        gross_loss = abs(sum(t.get('profit', 0) for t in closed_trades if t.get('profit', 0) < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # متوسط الربح
        avg_profit = np.mean([t.get('profit', 0) for t in closed_trades]) if closed_trades else 0
        
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
            'current_streak': self.real_time_stats['current_streak'],
            'max_win_streak': self.real_time_stats['max_win_streak']
        }
        
        logger.info(f"🎊 انتهت المحاكاة | الربح: ${total_profit:.2f} ({total_return:.2f}%)")
        
        return {
            'symbol_results': results,
            'performance_metrics': final_metrics,
            'trade_history': self.trade_history,
            'equity_curve': self.real_time_stats['equity_curve']
        }
