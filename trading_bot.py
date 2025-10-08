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
        """مؤشرات فنية مبسطة وفعالة"""
        df = df.copy()
        
        try:
            # المتوسطات المتحركة الأساسية
            periods = [5, 10, 20]
            for period in periods:
                df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
                df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
            
            # RSI مبسط
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # بولنجر باندز مبسطة
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            
            # مؤشرات الاتجاه
            df['trend_up'] = (df['ema_5'] > df['ema_10']) & (df['ema_10'] > df['ema_20'])
            df['trend_down'] = (df['ema_5'] < df['ema_10']) & (df['ema_10'] < df['ema_20'])
            
            # الزخم
            df['momentum'] = df['close'] / df['close'].shift(3) - 1
            
            # تعبئة القيم NaN
            df = df.fillna(method='bfill').fillna(method='ffill')
            
        except Exception as e:
            logger.error(f"خطأ في حساب المؤشرات: {e}")
            # قيم افتراضية
            for period in [5, 10, 20]:
                df[f'ema_{period}'] = df['close']
                df[f'sma_{period}'] = df['close']
            df['rsi'] = 50
            df['bb_upper'] = df['close'] * 1.1
            df['bb_lower'] = df['close'] * 0.9
            df['trend_up'] = False
            df['trend_down'] = False
            df['momentum'] = 0
        
        return df
    
    def generate_trading_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """إشارات تداول مبسطة وفعالة"""
        df = df.copy()
        
        try:
            # نظام تصويت مبسط
            df['buy_signals'] = 0
            df['sell_signals'] = 0
            
            # شروط شراء بسيطة وفعالة
            if 'rsi' in df.columns:
                df.loc[df['rsi'] < 35, 'buy_signals'] += 2
                df.loc[df['rsi'] > 65, 'sell_signals'] += 2
            
            if 'trend_up' in df.columns:
                df.loc[df['trend_up'], 'buy_signals'] += 1
                df.loc[df['trend_down'], 'sell_signals'] += 1
            
            if 'bb_lower' in df.columns:
                df.loc[df['close'] < df['bb_lower'], 'buy_signals'] += 1
                df.loc[df['close'] > df['bb_upper'], 'sell_signals'] += 1
            
            if 'momentum' in df.columns:
                df.loc[df['momentum'] > 0.02, 'buy_signals'] += 1
                df.loc[df['momentum'] < -0.02, 'sell_signals'] += 1
            
            # قرارات التداول - عتبات منخفضة لزيادة الصفقات
            df['buy_signal'] = df['buy_signals'] >= 2
            df['sell_signal'] = df['sell_signals'] >= 2
            
        except Exception as e:
            logger.error(f"خطأ في توليد الإشارات: {e}")
            df['buy_signal'] = False
            df['sell_signal'] = False
        
        return df
    
    def calculate_position_size(self, current_price: float) -> float:
        """حجم مركز مبسط للمبالغ الصغيرة"""
        try:
            # حجم ثابت للمبالغ الصغيرة
            if self.current_balance < 50:
                position_value = max(2.0, self.current_balance * 0.2)  # 20% للرؤوس الصغيرة
            else:
                position_value = self.current_balance * self.config.risk_per_trade * 2
            
            # عدم تجاوز الحد الأقصى
            max_position_value = self.current_balance * self.config.max_position_size
            position_value = min(position_value, max_position_value)
            
            # حد أدنى للصفقة
            position_value = max(position_value, self.config.min_trade_amount)
            
            quantity = position_value / current_price if current_price > 0 else 0
            return round(quantity, 6)
            
        except Exception as e:
            logger.error(f"خطأ في حساب حجم المركز: {e}")
            return 0.0
    
    def execute_compounding(self, profit: float, trade_info: Dict):
        """نظام ربح تراكمي فوري"""
        try:
            old_balance = self.current_balance
            self.current_balance += profit
            
            self.real_time_stats['total_profit'] += profit
            
            if profit > 0:
                self.real_time_stats['winning_trades'] += 1
                self.real_time_stats['current_streak'] = max(self.real_time_stats['current_streak'] + 1, 0)
                self.real_time_stats['consecutive_losses'] = 0
                
                # نمو تراكمي
                growth_rate = profit / old_balance if old_balance > 0 else 0
                self.real_time_stats['compounded_growth'] = (
                    (1 + self.real_time_stats['compounded_growth']) * (1 + growth_rate) - 1
                )
                
                logger.info(f"💰 ربح تراكمي: +${profit:.4f}")
            else:
                self.real_time_stats['losing_trades'] += 1
                self.real_time_stats['current_streak'] = min(self.real_time_stats['current_streak'] - 1, 0)
                self.real_time_stats['consecutive_losses'] += 1
            
            # تحديث منحنى رأس المال
            self.real_time_stats['equity_curve'].append(self.current_balance)
            
            # تحديث أقصى خسارة
            if len(self.real_time_stats['equity_curve']) > 0:
                peak = max(self.real_time_stats['equity_curve'])
                current_drawdown = (peak - self.current_balance) / peak * 100 if peak > 0 else 0
                self.real_time_stats['max_drawdown'] = max(
                    self.real_time_stats['max_drawdown'],
                    current_drawdown
                )
                
        except Exception as e:
            logger.error(f"خطأ في النظام التراكمي: {e}")
    
    def run_backtest(self, market_data: Dict, symbols: List[str]) -> Dict:
        """المحاكاة الرئيسية - مبسطة وفعالة"""
        try:
            logger.info(f"🎯 بدء المحاكاة على {len(symbols)} عملات")
            
            for symbol in symbols:
                if symbol not in market_data:
                    continue
                    
                df = market_data[symbol].copy()
                
                # حساب المؤشرات
                df = self.calculate_technical_indicators(df)
                df = self.generate_trading_signals(df)
                
                for i, (timestamp, row) in enumerate(df.iterrows()):
                    if i < 20:  # تأكد من وجود بيانات كافية
                        continue
                        
                    current_price = row['close']
                    
                    # دخول صفقة شراء
                    if (row.get('buy_signal', False) and 
                        symbol not in self.positions and 
                        self.current_balance > 5.0):  # حد أدنى للرصيد
                        
                        quantity = self.calculate_position_size(current_price)
                        
                        if quantity > 0:
                            # إدارة المخاطرة
                            stop_loss = current_price * 0.97  # 3% وقف خسارة
                            take_profit = current_price * 1.04  # 4% جني أرباح
                            
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
                                'status': 'OPEN'
                            }
                            
                            self.trade_history.append(trade_record)
                    
                    # إغلاق الصفقات
                    elif symbol in self.positions:
                        position = self.positions[symbol]
                        close_trade = False
                        close_reason = ""
                        
                        # شروط إغلاق بسيطة
                        if row.get('sell_signal', False):
                            close_trade = True
                            close_reason = "إشارة بيع"
                        elif current_price <= position['stop_loss']:
                            close_trade = True
                            close_reason = "وقف خسارة"
                        elif current_price >= position['take_profit']:
                            close_trade = True
                            close_reason = "جني أرباح"
                        # إغلاق تلقائي بعد 5 فترات
                        elif len([t for t in self.trade_history if t.get('symbol') == symbol and t.get('status') == 'OPEN']) >= 5:
                            close_trade = True
                            close_reason = "إغلاق زمني"
                        
                        if close_trade:
                            profit = (current_price - position['entry_price']) * position['quantity']
                            
                            # تطبيق الربح التراكمي
                            self.execute_compounding(profit, {
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
                            
                            self.trade_history.append(trade_record)
                            
                            # إزالة المركز
                            del self.positions[symbol]
            
            # حساب النتائج النهائية
            return self.calculate_final_metrics()
            
        except Exception as e:
            logger.error(f"خطأ في المحاكاة: {e}")
            return self.get_default_results()
    
    def calculate_final_metrics(self) -> Dict:
        """حساب المقاييس النهائية"""
        try:
            total_profit = self.current_balance - self.initial_balance
            total_return = (total_profit / self.initial_balance) * 100 if self.initial_balance > 0 else 0
            
            closed_trades = [t for t in self.trade_history if t.get('status') == 'CLOSED']
            winning_trades = [t for t in closed_trades if t.get('profit', 0) > 0]
            win_rate = (len(winning_trades) / len(closed_trades)) * 100 if closed_trades else 0
            
            # حساب عامل الربح
            gross_profit = sum(t.get('profit', 0) for t in winning_trades)
            losing_trades = [t for t in closed_trades if t.get('profit', 0) < 0]
            gross_loss = abs(sum(t.get('profit', 0) for t in losing_trades)) if losing_trades else 0.001
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0
            
            # متوسط الربح
            avg_profit = np.mean([t.get('profit', 0) for t in closed_trades]) if closed_trades else 0
            
            # تقييم الأداء
            performance_score = min(100, max(0, (
                win_rate * 0.4 +
                min(max(total_return, 0) * 2, 30) +
                min(profit_factor * 20, 30)
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
            
            logger.info(f"🎊 انتهت المحاكاة | الربح: ${total_profit:.2f} ({total_return:.2f}%) | النجاح: {win_rate:.1f}%")
            
            return {
                'symbol_results': {},
                'performance_metrics': final_metrics,
                'trade_history': self.trade_history,
                'equity_curve': self.real_time_stats['equity_curve']
            }
            
        except Exception as e:
            logger.error(f"خطأ في حساب المقاييس: {e}")
            return self.get_default_results()
    
    def get_default_results(self) -> Dict:
        """نتائج افتراضية في حالة حدوث أخطاء"""
        return {
            'symbol_results': {},
            'performance_metrics': {
                'initial_balance': self.initial_balance,
                'final_balance': self.initial_balance,
                'total_profit': 0.0,
                'total_return': 0.0,
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0.0,
                'max_drawdown': 0.0,
                'compounded_growth': 0.0,
                'profit_factor': 0.0,
                'avg_profit': 0.0,
                'performance_score': 0.0
            },
            'trade_history': [],
            'equity_curve': [self.initial_balance]
        }
