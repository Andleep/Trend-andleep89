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
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #2c3e50 0%, #3498db 100%);
    }
</style>
""", unsafe_allow_html=True)

class TradingDashboard:
    def __init__(self):
        self.bot = None
        self.results = None
        
    def setup_sidebar(self):
        """إعداد الشريط الجانبي"""
        st.sidebar.markdown("<h2 style='text-align: center; color: white;'>⚙️ إعدادات البوت</h2>", unsafe_allow_html=True)
        
        # إعدادات رأس المال
        st.sidebar.markdown("### 💰 رأس المال")
        capital = st.sidebar.number_input(
            "رأس المال الأولي ($)",
            min_value=10.0,
            max_value=100000.0,
            value=1000.0,
            step=100.0,
            help="ابدأ برأس مال مناسب للمخاطرة"
        )
        
        # إدارة المخاطر
        st.sidebar.markdown("### 🛡️ إدارة المخاطر")
        risk_level = st.sidebar.select_slider(
            "مستوى المخاطرة:",
            options=["منخفض", "متوسط", "عالي", "عدواني"],
            value="متوسط"
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
            "SOLUSDT", "DOTUSDT", "LINKUSDT", "LTCUSDT", "MATICUSDT",
            "DOGEUSDT", "AVAXUSDT", "ATOMUSDT", "ETCUSDT", "BCHUSDT"
        ]
        
        selected_pairs = st.sidebar.multiselect(
            "اختر العملات:",
            crypto_pairs,
            default=["BTCUSDT", "ETHUSDT", "BNBUSDT"],
            max_selections=8
        )
        
        # استراتيجية التداول
        st.sidebar.markdown("### 📈 الاستراتيجية")
        strategy = st.sidebar.selectbox(
            "استراتيجية التداول:",
            ["مستمر", "محافظ", "عدواني"],
            index=0
        )
        
        # فترة المحاكاة
        st.sidebar.markdown("### ⏰ فترة المحاكاة")
        simulation_period = st.sidebar.selectbox(
            "المدة:",
            ["1 يوم", "1 أسبوع", "1 شهر", "3 أشهر", "6 أشهر"],
            index=2
        )
        
        period_mapping = {
            "1 يوم": 1,
            "1 أسبوع": 7,
            "1 شهر": 30,
            "3 أشهر": 90,
            "6 أشهر": 180
        }
        days = period_mapping[simulation_period]
        
        # التداول المباشر
        st.sidebar.markdown("### 🔄 التداول المباشر")
        live_trading = st.sidebar.checkbox("تفعيل التداول المباشر", value=False)
        
        return {
            'capital': capital,
            'risk_per_trade': risk_per_trade,
            'selected_pairs': selected_pairs,
            'strategy': strategy,
            'days': days,
            'live_trading': live_trading
        }
    
    def generate_sample_data(self, symbols, days=30):
        """إنشاء بيانات نموذجية للمحاكاة"""
        import random
        from datetime import datetime, timedelta
        
        sample_data = {}
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        for symbol in symbols:
            # إنشاء بيانات تاريخية عشوائية واقعية
            dates = pd.date_range(start=start_date, end=end_date, freq='1H')
            n_periods = len(dates)
            
            # إنشاء أسعار واقعية
            base_price = random.uniform(10, 50000)  # سعر أساسي عشوائي
            prices = [base_price]
            
            for i in range(1, n_periods):
                # تغيير سعري واقعي (±2%)
                change = random.uniform(-0.02, 0.02)
                new_price = prices[-1] * (1 + change)
                prices.append(new_price)
            
            df = pd.DataFrame({
                'timestamp': dates,
                'open': [p * random.uniform(0.998, 1.002) for p in prices],
                'high': [p * random.uniform(1.001, 1.005) for p in prices],
                'low': [p * random.uniform(0.995, 0.999) for p in prices],
                'close': prices,
                'volume': [random.uniform(1000, 100000) for _ in prices]
            })
            
            df.set_index('timestamp', inplace=True)
            sample_data[symbol] = df
        
        return sample_data
    
    def run_simulation(self, settings):
        """تشغيل المحاكاة"""
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
            selected_pairs=settings['selected_pairs']
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
            time.sleep(0.02)
        
        # إنشاء بيانات نموذجية
        market_data = self.generate_sample_data(settings['selected_pairs'], settings['days'])
        
        # تشغيل المحاكاة
        self.results = self.bot.run_backtest(market_data, settings['selected_pairs'])
        
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
            profit_color = "profit-positive" if metrics['total_profit'] > 0 else "profit-negative"
            st.markdown(f"""
            <div class='metric-card'>
                <h3>💰 الرصيد النهائي</h3>
                <h2>${metrics['final_balance']:,.2f}</h2>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
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
            st.markdown(f"""
            <div class='metric-card'>
                <h3>🔄 النمو التراكمي</h3>
                <h2 class='profit-positive'>{metrics['compounded_growth']:.2f}%</h2>
                <p>ربح فوري بعد كل صفقة</p>
            </div>
            """, unsafe_allow_html=True)
        
        # مقاييس إضافية
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📊 متوسط الربح/صفقة", f"${metrics.get('avg_profit', 0):.2f}")
        with col2:
            st.metric("⚖️ عامل الربح", f"{metrics.get('profit_factor', 0):.2f}")
        with col3:
            st.metric("📉 أقصى خسارة", f"{metrics.get('max_drawdown', 0):.2f}%")
        with col4:
            st.metric("🔢 إجمالي الصفقات", metrics['total_trades'])
    
    def plot_equity_curve(self, results):
        """رسم منحنى رأس المال"""
        equity_curve = results.get('equity_curve', [])
        if not equity_curve:
            return
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=list(range(len(equity_curve))),
            y=equity_curve,
            mode='lines',
            name='رأس المال التراكمي',
            line=dict(color='#00FF88', width=4),
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
            st.info("لا توجد صفقات لتحليلها")
            return
        
        # إحصائيات الصفقات
        col1, col2, col3 = st.columns(3)
        
        closed_trades = trades_df[trades_df['status'] == 'CLOSED']
        
        with col1:
            if not closed_trades.empty:
                avg_profit = closed_trades['profit'].mean()
                st.metric("💵 متوسط الربح/صفقة", f"${avg_profit:.2f}")
        
        with col2:
            if not closed_trades.empty:
                best_trade = closed_trades['profit'].max()
                st.metric("🚀 أفضل صفقة", f"${best_trade:.2f}")
        
        with col3:
            if not closed_trades.empty:
                worst_trade = closed_trades['profit'].min()
                st.metric("📉 أسوأ صفقة", f"${worst_trade:.2f}")
        
        # توزيع الأرباح
        if not closed_trades.empty:
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=closed_trades['profit'],
                nbinsx=30,
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
        
        display_columns = ['symbol', 'timestamp', 'action', 'price', 'amount', 'profit', 'status']
        available_columns = [col for col in display_columns if col in trades_df.columns]
        
        display_df = trades_df[available_columns].copy()
        
        if 'profit' in display_df.columns:
            display_df['profit'] = display_df['profit'].apply(
                lambda x: f"${x:.2f}" if pd.notnull(x) else "-"
            )
        
        if 'amount' in display_df.columns:
            display_df['amount'] = display_df['amount'].apply(
                lambda x: f"${x:.2f}" if pd.notnull(x) else "-"
            )
        
        st.dataframe(display_df, use_container_width=True)
    
    def display_strategy_insights(self, results):
        """عرض insights عن الاستراتيجية"""
        st.markdown("### 🧠 تحليل الاستراتيجية")
        
        metrics = results['performance_metrics']
        
        # تقييم الأداء
        performance_score = min(100, max(0, (
            metrics['win_rate'] * 0.3 +
            (metrics['total_return'] * 2) * 0.3 +
            (100 - metrics.get('max_drawdown', 0)) * 0.2 +
            min(metrics.get('profit_factor', 0) * 20, 20)
        )))
        
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
            if metrics['total_trades'] < 10:
                tips.append("⚡ زيادة عدد أزواج التداول لفرص أكثر")
            if not tips:
                tips.append("✅ الإعدادات ممتازة! حافظ على نفس الاستراتيجية")
            
            for tip in tips:
                st.write(tip)
    
    def run_live_demo(self, settings):
        """تشغيل العرض التجريبي المباشر"""
        st.markdown("### 🔄 التداول المباشر التجريبي")
        
        # محاكاة التداول المباشر
        live_placeholder = st.empty()
        
        for i in range(5):
            with live_placeholder.container():
                st.info(f"🔄 جولة التداول {i+1}/5 - تحليل السوق الحالي...")
                
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
        
        st.success("🎊 انتهت جولة التداول المباشر! راجع النتائج الكاملة في تقرير المحاكاة.")
    
    def main(self):
        """الدالة الرئيسية"""
        st.markdown("<h1 class='main-header'>🚀 البوت التداولي الذكي - ALGOX</h1>", unsafe_allow_html=True)
        st.markdown("### 🤖 نظام ربح تراكمي فوري - بداية من $10 إلى ما لا نهاية")
        
        # علامات التبويب
        tab1, tab2, tab3 = st.tabs(["🎯 المحاكاة الشاملة", "📊 التحليل الفني", "🚀 التداول المباشر"])
        
        with tab1:
            st.markdown("### ⚡ محاكاة أداء البوت")
            
            # إعدادات المستخدم
            settings = self.setup_sidebar()
            
            # زر التشغيل الرئيسي
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 بدء المحاكاة الشاملة", use_container_width=True):
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
            st.markdown("### 💎 لمحة سريعة عن النظام")
            
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
            
            # محاكاة التحليل الفني
            st.info("🔄 جاري تحليل أنماط السوق والفرص المتاحة...")
            
            # مؤشرات فنية
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('الاتجاه العام', 'القوة النسبية', 'الحجم', 'التقلب')
            )
            
            # إضافة بيانات نموذجية
            x = list(range(50))
            fig.add_trace(go.Scatter(x=x, y=np.cumsum(np.random.randn(50)), name='الاتجاه'), row=1, col=1)
            fig.add_trace(go.Scatter(x=x, y=50 + np.random.randn(50), name='RSI'), row=1, col=2)
            fig.add_trace(go.Bar(x=x, y=np.random.exponential(100, 50), name='الحجم'), row=2, col=1)
            fig.add_trace(go.Scatter(x=x, y=np.random.randn(50).cumsum(), name='التقلب'), row=2, col=2)
            
            fig.update_layout(height=600, template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.markdown("### 🔄 نظام التداول المباشر")
            
            if st.button("🎯 بدء التداول المباشر التجريبي", type="primary"):
                settings = self.setup_sidebar()
                self.run_live_demo(settings)

# تشغيل التطبيق
if __name__ == "__main__":
    dashboard = TradingDashboard()
    dashboard.main()
