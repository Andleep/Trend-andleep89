import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import requests
import json
from trading_bot import ImprovedTradingBot as AdvancedTradingBot, TradingConfig
import warnings
warnings.filterwarnings('ignore')

# إعداد الصفحة
st.set_page_config(
    page_title="البوت التداولي الذكي - ALGOX",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# تنسيق CSS مخصص
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: bold;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .profit-positive {
        color: #00ff00;
        font-weight: bold;
    }
    .profit-negative {
        color: #ff4444;
        font-weight: bold;
    }
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 10px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

class TradingDashboard:
    def __init__(self):
        self.bot = None
        self.results = None
        
    def setup_sidebar(self):
        """إعداد الشريط الجانبي مع مفاتيح فريدة"""
        st.sidebar.markdown("<h2 style='text-align: center; color: white;'>⚙️ إعدادات البوت</h2>", unsafe_allow_html=True)
        
        # إعدادات رأس المال
        st.sidebar.markdown("### 💰 رأس المال")
        capital = st.sidebar.number_input(
            "رأس المال الأولي ($)",
            min_value=10.0,
            max_value=100000.0,
            value=1000.0,
            step=100.0,
            help="ابدأ برأس مال مناسب للمخاطرة",
            key="capital_input_unique"
        )
        
        # إدارة المخاطر
        st.sidebar.markdown("### 🛡️ إدارة المخاطر")
        risk_level = st.sidebar.selectbox(
            "مستوى المخاطرة:",
            ["منخفض", "متوسط", "عالي", "عدواني"],
            index=1,
            key="risk_level_unique"
        )
        
        risk_mapping = {
            "منخفض": 0.01,    # 1%
            "متوسط": 0.02,    # 2%
            "عالي": 0.03,     # 3%
            "عدواني": 0.05    # 5%
        }
        risk_per_trade = risk_mapping[risk_level]
        
        # أزواج التداول
        st.sidebar.markdown("### 🎯 أزواج التداول")
        
        crypto_pairs = [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "XRPUSDT",
            "SOLUSDT", "DOTUSDT", "LINKUSDT", "LTCUSDT", "MATICUSDT"
        ]
        
        selected_pairs = st.sidebar.multiselect(
            "اختر العملات:",
            crypto_pairs,
            default=["BTCUSDT", "ETHUSDT", "BNBUSDT"],
            max_selections=5,
            key="pairs_select_unique"
        )
        
        # استراتيجية التداول
        st.sidebar.markdown("### 📈 الاستراتيجية")
        strategy = st.sidebar.selectbox(
            "استراتيجية التداول:",
            ["مستمر", "محافظ", "عدواني"],
            index=0,
            key="strategy_select_unique"
        )
        
        # فترة المحاكاة
        st.sidebar.markdown("### ⏰ فترة المحاكاة")
        simulation_period = st.sidebar.selectbox(
            "المدة:",
            ["1 يوم", "1 أسبوع", "1 شهر", "3 أشهر", "6 أشهر"],
            index=2,
            key="period_select_unique"
        )
        
        period_mapping = {
            "1 يوم": 1,
            "1 أسبوع": 7,
            "1 شهر": 30,
            "3 أشهر": 90,
            "6 أشهر": 180
        }
        days = period_mapping[simulation_period]
        
        return {
            'capital': capital,
            'risk_per_trade': risk_per_trade,
            'selected_pairs': selected_pairs,
            'strategy': strategy,
            'days': days
        }
    
    def generate_realistic_sample_data(self, symbols, days=30):
        """إنشاء بيانات واقعية مع تقلبات حقيقية"""
        sample_data = {}
        
        for symbol in symbols:
            # إنشاء تواريخ
            dates = pd.date_range(end=datetime.now(), periods=days*24, freq='H')
            
            # إنشاء أسعار واقعية مع اتجاهات وتقلبات
            np.random.seed(hash(symbol) % 10000)  # بذور مختلفة لكل عملة
            
            # سعر ابتدائي واقعي
            if "BTC" in symbol:
                base_price = 45000
            elif "ETH" in symbol:
                base_price = 3000
            else:
                base_price = np.random.uniform(10, 500)
            
            prices = [base_price]
            volumes = [np.random.uniform(1000, 50000)]
            
            for i in range(1, len(dates)):
                # تقلبات واقعية مع اتجاهات
                trend = np.random.choice([-0.001, 0, 0.001], p=[0.3, 0.4, 0.3])
                volatility = np.random.uniform(-0.02, 0.02)
                change = trend + volatility
                
                new_price = prices[-1] * (1 + change)
                prices.append(new_price)
                
                # حجم متغير
                new_volume = volumes[-1] * np.random.uniform(0.8, 1.2)
                volumes.append(new_volume)
            
            df = pd.DataFrame({
                'timestamp': dates,
                'open': [p * np.random.uniform(0.999, 1.001) for p in prices],
                'high': [p * np.random.uniform(1.001, 1.005) for p in prices],
                'low': [p * np.random.uniform(0.995, 0.999) for p in prices],
                'close': prices,
                'volume': volumes
            })
            
            df.set_index('timestamp', inplace=True)
            sample_data[symbol] = df
        
        return sample_data
    
    def run_simulation(self, settings):
        """تشغيل المحاكاة مع بيانات واقعية"""
        if not settings['selected_pairs']:
            st.error("⚠️ الرجاء اختيار عملات للتداول أولاً")
            return None
        
        # شريط التقدم
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # إنشاء إعدادات البوت
        config = TradingConfig(
            initial_capital=settings['capital'],
            risk_per_trade=settings['risk_per_trade'],
            selected_pairs=settings['selected_pairs'],
            min_trade_amount=1.0
        )
        
        # إنشاء البوت
        self.bot = AdvancedTradingBot(config)
        
        # محاكاة التقدم
        for i in range(100):
            progress_bar.progress(i + 1)
            if i < 25:
                status_text.text("🔄 جلب بيانات السوق...")
            elif i < 50:
                status_text.text("📊 تحليل المؤشرات الفنية...")
            elif i < 75:
                status_text.text("🤖 البوت يتخذ القرارات...")
            else:
                status_text.text("💰 حساب الأرباح التراكمية...")
            time.sleep(0.01)
        
        # إنشاء بيانات واقعية
        market_data = self.generate_realistic_sample_data(settings['selected_pairs'], settings['days'])
        
        # تشغيل المحاكاة
        try:
            self.results = self.bot.run_backtest(market_data, settings['selected_pairs'])
        except Exception as e:
            st.error(f"❌ خطأ في المحاكاة: {str(e)}")
            return None
        
        progress_bar.empty()
        status_text.empty()
        
        return self.results
    
    def display_performance_metrics(self, results):
        """عرض مقاييس الأداء"""
        st.markdown("<h2 class='main-header'>📊 تقرير الأداء الشامل</h2>", unsafe_allow_html=True)
        
        metrics = results['performance_metrics']
        
        # البطاقات الرئيسية
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>💰 الرصيد النهائي</h3>
                <h2>${metrics['final_balance']:,.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            profit_color = "profit-positive" if metrics['total_profit'] > 0 else "profit-negative"
            st.markdown(f"""
            <div class='metric-card'>
                <h3>🎯 إجمالي الربح</h3>
                <h2 class='{profit_color}'>${metrics['total_profit']:,.2f}</h2>
                <p>{metrics['total_return']:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class='metric-card'>
                <h3>📈 معدل النجاح</h3>
                <h2>{metrics['win_rate']:.1f}%</h2>
                <p>{metrics['winning_trades']} / {metrics['total_trades']} صفقات</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            growth_color = "profit-positive" if metrics['compounded_growth'] > 0 else "profit-negative"
            st.markdown(f"""
            <div class='metric-card'>
                <h3>🔄 النمو التراكمي</h3>
                <h2 class='{growth_color}'>{metrics['compounded_growth']:.2f}%</h2>
                <p>ربح فوري بعد كل صفقة</p>
            </div>
            """, unsafe_allow_html=True)
        
        # مقاييس إضافية
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 متوسط الربح/صفقة", f"${metrics.get('avg_profit', 0):.4f}")
        with col2:
            st.metric("⚖️ عامل الربح", f"{metrics.get('profit_factor', 0):.2f}")
        with col3:
            st.metric("📉 أقصى خسارة", f"{metrics.get('max_drawdown', 0):.2f}%")
        with col4:
            st.metric("🔢 إجمالي الصفقات", metrics['total_trades'])
    
    def plot_equity_curve(self, results):
        """رسم منحنى رأس المال"""
        equity_curve = results.get('equity_curve', [])
        if len(equity_curve) <= 1:
            st.info("📊 لا توجد بيانات كافية لعرض منحنى رأس المال")
            return
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=list(range(len(equity_curve))),
            y=equity_curve,
            mode='lines+markers',
            name='رأس المال التراكمي',
            line=dict(color='#00FF88', width=4),
            marker=dict(size=4),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 136, 0.1)'
        ))
        
        # إضافة خط الربح الأولي
        fig.add_hline(y=results['performance_metrics']['initial_balance'], 
                     line_dash="dash", line_color="white", 
                     annotation_text="رأس المال الأولي")
        
        fig.update_layout(
            title="📈 منحنى النمو التراكمي - نظام الربح الفوري",
            xaxis_title="عدد الصفقات",
            yaxis_title="رأس المال ($)",
            height=400,
            template="plotly_dark",
            showlegend=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def display_trade_analysis(self, results):
        """تحليل الصفقات"""
        st.markdown("### 📋 التحليل التفصيلي للصفقات")
        
        trades_df = pd.DataFrame(results['trade_history'])
        if trades_df.empty:
            st.info("📭 لا توجد صفقات لتحليلها في هذه المحاكاة")
            return
        
        closed_trades = trades_df[trades_df['status'] == 'CLOSED']
        
        if closed_trades.empty:
            st.info("📭 لا توجد صفقات مغلقة لتحليلها")
            return
        
        # إحصائيات الصفقات
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_profit = closed_trades['profit'].mean()
            st.metric("💵 متوسط الربح/صفقة", f"${avg_profit:.4f}")
        
        with col2:
            best_trade = closed_trades['profit'].max()
            st.metric("🚀 أفضل صفقة", f"${best_trade:.4f}")
        
        with col3:
            worst_trade = closed_trades['profit'].min()
            st.metric("📉 أسوأ صفقة", f"${worst_trade:.4f}")
        
        # توزيع الأرباح
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=closed_trades['profit'],
            nbinsx=20,
            name='توزيع الأرباح',
            marker_color='#636EFA',
            opacity=0.7
        ))
        
        fig.update_layout(
            title="توزيع أرباح الصفقات",
            xaxis_title="الربح ($)",
            yaxis_title="عدد الصفقات",
            height=300,
            template="plotly_dark"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # عرض الصفقات
        st.markdown("### 💼 سجل الصفقات")
        
        display_columns = ['symbol', 'timestamp', 'action', 'price', 'amount', 'profit', 'close_reason']
        available_columns = [col for col in display_columns if col in closed_trades.columns]
        
        display_df = closed_trades[available_columns].copy()
        
        if 'profit' in display_df.columns:
            display_df['profit'] = display_df['profit'].apply(
                lambda x: f"${x:.4f}" if pd.notnull(x) else "-"
            )
        
        if 'amount' in display_df.columns:
            display_df['amount'] = display_df['amount'].apply(
                lambda x: f"${x:.2f}" if pd.notnull(x) else "-"
            )
        
        st.dataframe(display_df.head(20), use_container_width=True)
        
        if len(display_df) > 20:
            st.info(f"📋 عرض 20 من أصل {len(display_df)} صفقة")
    
    def display_strategy_insights(self, results):
        """عرض insights عن الاستراتيجية"""
        st.markdown("### 🧠 تحليل الاستراتيجية")
        
        metrics = results['performance_metrics']
        
        # تقييم الأداء
        performance_score = metrics.get('performance_score', 0)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 تقييم البوت")
            st.progress(performance_score / 100)
            st.write(f"**{performance_score:.1f}%** - تقييم شامل للأداء")
            
            # توصيات
            if performance_score >= 80:
                st.success("🎉 البوت ممتاز! يمكن زيادة رأس المال")
            elif performance_score >= 60:
                st.info("👍 البوت جيد! استمر في الاستخدام")
            elif performance_score >= 40:
                st.warning("⚠️ البوت مقبول! قد تحتاج لتعديل الإعدادات")
            else:
                st.error("❌ البوت يحتاج تحسين! راجع الإعدادات")
        
        with col2:
            st.markdown("#### 💡 نصائح للتحسين")
            
            tips = []
            if metrics['win_rate'] < 60:
                tips.append("🔻 خفض مستوى المخاطرة لزيادة دقة الصفقات")
            if metrics.get('max_drawdown', 0) > 15:
                tips.append("🛡️ زيادة وقف الخسارة لتقليل الخسائر")
            if metrics['total_trades'] < 5:
                tips.append("⚡ اختر المزيد من أزواج التداول لفرص أكثر")
            if metrics['total_trades'] > 50:
                tips.append("🎯 رفع عتبات الدخول لتحسين جودة الصفقات")
            if not tips:
                tips.append("✅ الإعدادات ممتازة! حافظ على نفس الاستراتيجية")
            
            for tip in tips:
                st.write(tip)
    
    def run_live_demo(self):
        """تشغيل العرض التجريبي المباشر"""
        st.markdown("### 🔄 التداول المباشر التجريبي")
        
        # محاكاة التداول المباشر
        live_placeholder = st.empty()
        
        demo_results = {
            'initial_balance': 1000,
            'final_balance': 1245.50,
            'total_profit': 245.50,
            'total_return': 24.55,
            'total_trades': 15,
            'winning_trades': 11,
            'win_rate': 73.3,
            'compounded_growth': 24.55
        }
        
        for i in range(3):
            with live_placeholder.container():
                st.info(f"🔄 جولة التداول {i+1}/3 - تحليل السوق الحالي...")
                
                # محاكاة قرارات التداول
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("BTCUSDT", "$45,231.50", "+1.2%")
                with col2:
                    st.metric("ETHUSDT", "$3,215.80", "+0.8%")
                with col3:
                    st.metric("BNBUSDT", "$585.30", "+2.1%")
                
                # محاكاة صفقة
                if i % 2 == 0:
                    st.success("✅ صفقة شراء ناجحة! +$24.50 تمت إضافتها للرصيد")
                else:
                    st.warning("⏸️ انتظار لإشارة أفضل...")
                
            time.sleep(2)
        
        st.success("🎊 انتهت جولة التداول المباشر!")
        
        # عرض نتائج تجريبية
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("رأس المال النهائي", f"${demo_results['final_balance']:.2f}")
        col2.metric("إجمالي الربح", f"${demo_results['total_profit']:.2f}")
        col3.metric("معدل النجاح", f"{demo_results['win_rate']:.1f}%")
        col4.metric("النمو التراكمي", f"{demo_results['compounded_growth']:.2f}%")
    
    def main(self):
        """الدالة الرئيسية"""
        st.markdown("<h1 class='main-header'>🚀 البوت التداولي الذكي - ALGOX</h1>", unsafe_allow_html=True)
        st.markdown("### 🤖 نظام ربح تراكمي فوري - بداية من $10 إلى ما لا نهاية")
        
        # تحميل الإعدادات مرة واحدة فقط
        settings = self.setup_sidebar()
        
        # علامات التبويب
        tab1, tab2, tab3 = st.tabs(["🎯 المحاكاة الشاملة", "📊 التحليل الفني", "🚀 التداول المباشر"])
        
        with tab1:
            st.markdown("### ⚡ محاكاة أداء البوت")
            
            # زر التشغيل الرئيسي
            if st.button("🚀 بدء المحاكاة الشاملة", use_container_width=True, type="primary"):
                with st.spinner("جاري تشغيل المحاكاة المتقدمة..."):
                    results = self.run_simulation(settings)
                    
                    if results:
                        # عرض النتائج
                        self.display_performance_metrics(results)
                        self.plot_equity_curve(results)
                        self.display_trade_analysis(results)
                        self.display_strategy_insights(results)
            
            # معلومات سريعة
            st.markdown("---")
            st.markdown("### 💎 مميزات النظام")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.info("""
                **📈 الربح التراكمي**
                - ربح فوري بعد كل صفقة
                - إضافة تلقائية للرصيد
                - نمو مضاعف مستمر
                """)
            with col2:
                st.info("""
                **🛡️ إدارة المخاطر**
                - وقف خسارة ذكي
                - حجم صفقات ديناميكي
                - حماية رأس المال
                """)
            with col3:
                st.info("""
                **🤖 الذكاء الاصطناعي**
                - تحليل فني متقدم
                - تعلم من الصفقات
                - تكيف مع السوق
                """)
        
        with tab2:
            st.markdown("### 📊 التحليل الفني المتقدم")
            
            # مؤشرات فنية تفاعلية
            st.info("📈 تحليل أنماط السوق والفرص المتاحة...")
            
            # إنشاء بيانات نموذجية للعرض
            dates = pd.date_range(end=datetime.now(), periods=100, freq='H')
            prices = [100]
            for i in range(1, 100):
                change = np.random.normal(0, 0.01)
                prices.append(prices[-1] * (1 + change))
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('الاتجاه العام', 'القوة النسبية RSI', 'الحجم', 'المتوسطات المتحركة'),
                vertical_spacing=0.1
            )
            
            # الرسم البياني للسعر
            fig.add_trace(go.Scatter(x=dates, y=prices, name='السعر', line=dict(color='#00FF88')), row=1, col=1)
            
            # RSI
            rsi = [50 + 20 * np.sin(i/10) for i in range(100)]
            fig.add_trace(go.Scatter(x=dates, y=rsi, name='RSI', line=dict(color='#FF6B6B')), row=1, col=2)
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=1, col=2)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=1, col=2)
            
            # الحجم
            volume = [np.random.uniform(1000, 5000) for _ in range(100)]
            fig.add_trace(go.Bar(x=dates, y=volume, name='الحجم', marker_color='#4ECDC4'), row=2, col=1)
            
            # المتوسطات
            fig.add_trace(go.Scatter(x=dates, y=prices, name='السعر', line=dict(color='#00FF88')), row=2, col=2)
            ma_20 = pd.Series(prices).rolling(20).mean()
            fig.add_trace(go.Scatter(x=dates, y=ma_20, name='EMA 20', line=dict(color='#FFE66D')), row=2, col=2)
            
            fig.update_layout(height=600, template="plotly_dark", showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("### 🔄 نظام التداول المباشر")
            
            if st.button("🎯 بدء التداول المباشر التجريبي", use_container_width=True, type="secondary"):
                self.run_live_demo()

# تشغيل التطبيق
if __name__ == "__main__":
    dashboard = TradingDashboard()
    dashboard.main()
