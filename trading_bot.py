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
    min_trade_amount: float = 5.0  # حد أدنى للصفقة
    
    def __post_init__(self):
        if self.selected_pairs is None:
            self.selected_pairs = ["BTCUSDT", "ETHUSDT", "BNBUSDT"]

class ImprovedTradingBot:
    """
    بوت تداول محسن مع ربح تراكمي فوري ودعم للمبالغ الصغيرة
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
    
    def calculate_advanced_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """مؤشرات فنية متقدمة بدقة أعلى"""
        df = df.copy()
        
        # المتوسطات المتحركة متعددة الفترات
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
        
        # RSI متعدد الفترات
        for period in [7, 14]:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
            rs = gain / loss
            df[f'rsi_{period}'] = 100 - (100 / (1 + rs))
        
        # MACD متقدم
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # بولنجر باندز متعدد المستويات
        for period in [20]:
            df[f'bb_middle_{period}'] = df['close'].rolling(period).mean()
            bb_std = df['close'].rolling(period).std()
            df[f'bb_upper_{period}'] = df[f'bb_middle_{period}'] + (bb_std * 2)
            df[f'bb_lower_{period}'] = df[f'bb_middle_{period}'] - (bb_std * 2)
            df[f'bb_width_{period}'] = (df[f'bb_upper_{period}'] - df[f'bb_lower_{period}']) / df[f'bb_middle_{period}']
        
        # مؤشر القوة النسبية الوزني
        df['weighted_rsi'] = (df['rsi_7'] * 0.3 + df['rsi_14'] * 0.7)
        
        # اتجاه قوي
        df['strong_trend_up'] = (df['ema_5'] > df['ema_10']) & (df['ema_10'] > df['ema_20']) & (df['ema_20'] > df['ema_50'])
        df['strong_trend_down'] = (df['ema_5'] < df['ema_10']) & (df['ema_10'] < df['ema_20']) & (df['ema_20'] < df['ema_50'])
        
        # تقلبات السوق
        df['volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
        
        # حجم التداول النسبي
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        return df
    
    def generate_intelligent_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """إشارات تداول ذكية بدقة عالية"""
        df = df.copy()
        
        # نظام تصويت متقدم
        df['buy_score'] = 0
        df['sell_score'] = 0
        
        # شروط الشراء القوية (3 نقاط)
        df.loc[df['strong_trend_up'], 'buy_score'] += 3
        df.loc[df['weighted_rsi'] < 35, 'buy_score'] += 2
        df.loc[df['macd_hist'] > 0, 'buy_score'] += 2
        df.loc[df['close'] < df['bb_lower_20'], 'buy_score'] += 2
        df.loc[df['volume_ratio'] > 1.5, 'buy_score'] += 1
        
        # شروط البيع القوية (3 نقاط)
        df.loc[df['strong_trend_down'], 'sell_score'] += 3
        df.loc[df['weighted_rsi'] > 65, 'sell_score'] += 2
        df.loc[df['macd_hist'] < 0, 'sell_score'] += 2
        df.loc[df['close'] > df['bb_upper_20'], 'sell_score'] += 2
        df.loc[df['volume_ratio'] > 1.5, 'sell_score'] += 1
        
        # شروط إضافية لتحسين الدقة
        df['momentum'] = df['close'] / df['close'].shift(5) - 1
        df.loc[df['momentum'] > 0.02, 'buy_score'] += 1
        df.loc[df['momentum'] < -0.02, 'sell_score'] += 1
        
        # قرارات التداول الذكية
        df['buy_signal'] = df['buy_score'] >= 6  # عتبة عالية للشراء
        df['sell_signal'] = df['sell_score'] >= 6  # عتبة عالية للبيع
        
        # تصفية الإشارات الضعيفة
        df['signal_strength'] = df['buy_score'] - df['sell_score']
        df['strong_buy'] = (df['buy_signal'] & (df['signal_strength'] >= 3))
        df['strong_sell'] = (df['sell_signal'] & (df['signal_strength'] <= -3))
        
        return df
    
    def calculate_smart_position_size(self, current_price: float, confidence: float, symbol: str) -> float:
        """حجم مركز ذكي مع دعم للمبالغ الصغيرة"""
        # تكييف المخاطرة بناءً على الأداء
        base_risk = self.config.risk_per_trade
        
        # تقليل المخاطرة بعد الخسائر المتتالية
        if self.real_time_stats['consecutive_losses'] >= 2:
            base_risk *= 0.5
        
        # زيادة المخاطرة بعد الأرباح المتتالية
        if self.real_time_stats['current_streak'] >= 2:
            base_risk = min(base_risk * 1.3, 0.05)  # حد أقصى 5%
        
        # تكييف مع حجم رأس المال
        if self.current_balance < 50:
            min_trade = max(1.0, self.current_balance * 0.1)  # 10% كحد أدنى للصفقات الصغيرة
        else:
            min_trade = self.config.min_trade_amount
        
        # حساب حجم المركز
        risk_amount = self.current_balance * base_risk * confidence
        position_value = max(risk_amount * 1.5, min_trade)  # رافعة محافظة
        
        # عدم تجاوز الحد الأقصى
        max_position_value = self.current_balance * self.config.max_position_size
        position_value = min(position_value, max_position_value)
        
        # حساب الكمية
        quantity = position_value / current_price if current_price > 0 else 0
        
        # تقريب لـ 4 منازل عشرية للعملات الرقمية
        quantity = round(quantity, 6)
        
        return quantity
    
    def execute_improved_compounding(self, profit: float, trade_info: Dict):
        """نظام ربح تراكمي محسن"""
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
            self.real_time_stats['consecutive_losses'] = 0
            
            # نمو تراكمي محسن
            growth_rate = profit / old_balance
            self.real_time_stats['compounded_growth'] = (
                (1 + self.real_time_stats['compounded_growth']) * (1 + growth_rate) - 1
            )
            
            logger.info(f"💰 ربح تراكمي: +${profit:.4f} | الرصيد: ${self.current_balance:.4f}")
        else:
            self.real_time_stats['losing_trades'] += 1
            self.real_time_stats['current_streak'] = min(self.real_time_stats['current_streak'] - 1, 0)
            self.real_time_stats['consecutive_losses'] += 1
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
    
    def should_enter_trade(self, symbol: str, row, current_price: float) -> bool:
        """تحديد إذا كان يجب الدخول في صفقة"""
        # عدم الدخول إذا كان الرصيد غير كافي
        if self.current_balance < 1.0:
            return False
        
        # عدم الدخول إذا كان هناك صفقة مفتوحة في نفس الزوج
        if symbol in self.positions:
            return False
        
        # الدخول فقط في الإشارات القوية
        if not row.get('strong_buy', False):
            return False
        
        # تجنب التداول في ظروف التقلب العالي
        if row.get('volatility', 0) > 0.05:  # تجنب التقلب فوق 5%
            return False
        
        return True
    
    def run_improved_backtest(self, market_data: Dict, symbols: List[str]) -> Dict:
        """محاكاة محسنة مع نتائج أفضل"""
        results = {}
        
        logger.info(f"🎯 بدء المحاكاة المحسنة على {len(symbols)} عملات")
        
        for symbol in symbols:
            if symbol not in market_data:
                continue
                
            logger.info(f"🔍 تحليل {symbol}...")
            df = market_data[symbol].copy()
            
            # حساب المؤشرات المتقدمة
            df = self.calculate_advanced_indicators(df)
            df = self.generate_intelligent_signals(df)
            
            symbol_trades = []
            
            for i, (timestamp, row) in enumerate(df.iterrows()):
                if i < 50:  # تأكد من وجود بيانات كافية
                    continue
                    
                current_price = row['close']
                atr_value = row.get('volatility', 0.02) * current_price
                
                # الدخول في صفقة شراء
                if self.should_enter_trade(symbol, row, current_price):
                    confidence = min(1.0, row['buy_score'] / 10.0)
                    quantity = self.calculate_smart_position_size(current_price, confidence, symbol)
                    
                    if quantity * current_price >= 1.0:  # صفقة ذات حجم معقول
                        # وقف خسارة وجني أرباح ذكي
                        stop_loss = current_price * 0.98  # 2% وقف خسارة
                        take_profit = current_price * 1.04  # 4% جني أرباح
                        
                        self.positions[symbol] = {
                            'entry_time': timestamp,
                            'entry_price': current_price,
                            'quantity': quantity,
                            'investment': quantity * current_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'type': 'LONG',
                            'confidence': confidence
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
                    
                    # شروط الإغلاق المحسنة
                    close_trade = False
                    close_reason = ""
                    exit_price = current_price
                    
                    if row.get('strong_sell', False):
                        close_trade = True
                        close_reason = "إشارة بيع قوية"
                    elif current_price <= position['stop_loss']:
                        close_trade = True
                        close_reason = "وقف خسارة"
                        exit_price = position['stop_loss']
                    elif current_price >= position['take_profit']:
                        close_trade = True
                        close_reason = "جني أرباح"
                        exit_price = position['take_profit']
                    # إغلاق بعد فترة زمنية (5 فترات)
                    elif len([t for t in symbol_trades if t.get('symbol') == symbol and t.get('status') == 'OPEN']) >= 5:
                        close_trade = True
                        close_reason = "إغلاق زمني"
                    
                    if close_trade:
                        profit = (exit_price - position['entry_price']) * position['quantity']
                        
                        # تطبيق الربح التراكمي المحسن
                        self.execute_improved_compounding(profit, {
                            'symbol': symbol,
                            'position': position,
                            'exit_price': exit_price,
                            'reason': close_reason
                        })
                        
                        # تسجيل الصفقة المغلقة
                        trade_record = {
                            'symbol': symbol,
                            'action': 'SELL',
                            'timestamp': timestamp,
                            'price': exit_price,
                            'quantity': position['quantity'],
                            'amount': position['investment'],
                            'profit': profit,
                            'profit_pct': (profit / position['investment']) * 100,
                            'status': 'CLOSED',
                            'close_reason': close_reason,
                            'confidence': position['confidence']
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
        
        # حساب المقاييس النهائية المحسنة
        return self.calculate_improved_metrics()
    
    def calculate_improved_metrics(self) -> Dict:
        """حساب مقاييس أداء محسنة"""
        total_profit = self.current_balance - self.initial_balance
        total_return = (total_profit / self.initial_balance) * 100
        
        closed_trades = [t for t in self.trade_history if t.get('status') == 'CLOSED']
        winning_trades = [t for t in closed_trades if t.get('profit', 0) > 0]
        win_rate = (len(winning_trades) / len(closed_trades)) * 100 if closed_trades else 0
        
        # حساب عامل الربح
        gross_profit = sum(t.get('profit', 0) for t in winning_trades)
        gross_loss = abs(sum(t.get('profit', 0) for t in closed_trades if t.get('profit', 0) < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # متوسط الربح وتحسينه
        avg_profit = np.mean([t.get('profit', 0) for t in closed_trades]) if closed_trades else 0
        avg_win = np.mean([t.get('profit', 0) for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.get('profit', 0) for t in closed_trades if t.get('profit', 0) < 0]) if len(closed_trades) > len(winning_trades) else 0
        
        # نسبة العائد إلى المخاطرة
        risk_reward_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # تقييم الأداء
        performance_score = self.calculate_performance_score(win_rate, profit_factor, total_return)
        
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
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'risk_reward_ratio': risk_reward_ratio,
            'performance_score': performance_score,
            'current_streak': self.real_time_stats['current_streak']
        }
        
        logger.info(f"🎊 انتهت المحاكاة | الربح: ${total_profit:.2f} ({total_return:.2f}%) | النجاح: {win_rate:.1f}%")
        
        return {
            'symbol_results': {},
            'performance_metrics': final_metrics,
            'trade_history': self.trade_history,
            'equity_curve': self.real_time_stats['equity_curve']
        }
    
    def calculate_performance_score(self, win_rate: float, profit_factor: float, total_return: float) -> float:
        """حساب تقييم أداء شامل"""
        score = 0
        
        # وزن معدل النجاح (40%)
        if win_rate >= 70:
            score += 40
        elif win_rate >= 60:
            score += 30
        elif win_rate >= 50:
            score += 20
        elif win_rate >= 40:
            score += 10
        
        # وزن عامل الربح (30%)
        if profit_factor >= 2.0:
            score += 30
        elif profit_factor >= 1.5:
            score += 25
        elif profit_factor >= 1.2:
            score += 20
        elif profit_factor >= 1.0:
            score += 15
        elif profit_factor >= 0.8:
            score += 10
        
        # وزن العائد الكلي (30%)
        if total_return >= 50:
            score += 30
        elif total_return >= 30:
            score += 25
        elif total_return >= 20:
            score += 20
        elif total_return >= 10:
            score += 15
        elif total_return >= 0:
            score += 10
        elif total_return >= -10:
            score += 5
        
        return min(100, score)
