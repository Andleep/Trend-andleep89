import os
from dataclasses import dataclass
from typing import List

@dataclass
class AppConfig:
    """إعدادات التطبيق"""
    # إعدادات Render
    RENDER_ENV: bool = os.getenv('RENDER', 'False').lower() == 'true'
    
    # إعدادات التداول الافتراضية
    DEFAULT_CAPITAL: float = 1000.0
    DEFAULT_RISK: float = 0.02
    DEFAULT_STRATEGY: str = "مستمر"
    
    # العملات المتاحة
    AVAILABLE_CRYPTO_PAIRS: List[str] = [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT",
        "SOLUSDT", "DOTUSDT", "LINKUSDT", "LTCUSDT", "MATICUSDT",
        "DOGEUSDT", "AVAXUSDT", "ATOMUSDT", "ETCUSDT", "BCHUSDT"
    ]
    
    # فترات المحاكاة
    SIMULATION_PERIODS = {
        "1 يوم": 1,
        "1 أسبوع": 7, 
        "1 شهر": 30,
        "3 أشهر": 90,
        "6 أشهر": 180
    }
    
    # إعدادات الأداء
    PERFORMANCE_THRESHOLDS = {
        'excellent': 80,
        'good': 60,
        'fair': 40,
        'poor': 0
    }

# إعدادات API (للتنفيذ الحقيقي)
BINANCE_CONFIG = {
    "api_key": os.getenv('BINANCE_API_KEY', ''),
    "api_secret": os.getenv('BINANCE_API_SECRET', ''),
    "testnet": True  # استخدام الشبكة التجريبية للاختبار
}
