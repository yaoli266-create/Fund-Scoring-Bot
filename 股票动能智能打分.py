import akshare as ak
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from datetime import datetime, timedelta
import warnings
import time
import random

warnings.filterwarnings('ignore')

print("🚀 启动全维量化大脑 V3.0 Pro (主升浪起爆点 + 智能防守防线)...")

# ==========================================
# 🛡️ 核心强化：智能重试引擎 (防超时与限流)
# ==========================================
def robust_akshare_call(func, *args, max_retries=3, **kwargs):
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if attempt < max_retries - 1:
                sleep_time = (2 ** attempt) + random.uniform(0.5, 1.5)
                time.sleep(sleep_time)
            else:
                raise e

# ==========================================
# 🛑 0. 顶层风控：大盘趋势嗅探
# ==========================================
def check_market_environment():
    print("正在嗅探系统性风险 (上证指数趋势)...")
    try:
        df_index = robust_akshare_call(ak.stock_zh_index_daily_em, symbol="sh000001")
        df_index['MA20'] = df_index['close'].rolling(window=20).mean()
        today_close = df_index.iloc[-1]['close']
        ma20 = df_index.iloc[-1]['MA20']
        
        is_market_good = today_close > ma20
        status_text = "🟢 安全 (站上20日线)" if is_market_good else "🔴 极度危险 (跌破20日线)"
        print(f"大盘状态: {status_text} | 最新点位: {today_close:.2f} | MA20: {ma20:.2f}")
        return is_market_good, today_close, ma20
    except Exception as e:
        print(f"⚠️ 大盘数据获取失败，默认放行: {e}")
        return True, 0, 0

IS_MARKET_GOOD, SH_CLOSE, SH_MA20 = check_market_environment()

# ==========================================
# 🔍 1. 实时粗筛：锁定高换手中小盘活跃股
# ==========================================
def get_active_stock_pool(top_n=500):
    print(f"正在扫描全市场，执行基本面与流动性双重过滤...")
    try:
        spot_df = robust_akshare_call(ak.stock_zh_a_spot_em)
        spot_df = spot_df[~spot_df['名称'].str.contains('ST|退|C|N')]
        spot_df = spot_df[(spot_df['涨跌幅'] < 9.8) & (spot_df['涨跌幅'] > -9.8)]
        spot_df = spot_df[spot_df['换手率'] >= 3.0]
        spot_df = spot_df[spot_df['流通市值'] <= 50000000000]
        
        active_pool = spot_df.sort_values(by='成交额', ascending=False).head(top_n)
        stock_list = active_pool['代码'].tolist()
        name_dict = dict(zip(active_pool['代码'], active_pool['名称']))
        spot_info_dict = active_pool.set_index('代码').to_dict('index')
        
        print(f"✅ 成功锁定 {len(stock_list)} 只 [高换手+中小盘] 标的，开启游资级量价诊断...")
        return stock_list, name_dict, spot_info_dict
    except Exception as e:
        print(f"⚠️ 粗筛网络彻底瘫痪: {e}")
        fallback_list = ['600519', '000858']
        return fallback_list, {c: c for c in fallback_list}, {}

stock_list, name_dict, spot_info_dict = get_active_stock_pool(top_n=500)

# ==========================================
# ⚡️ 2. 核心算力：V3.0 Pro 游资主升浪打分引擎
# ==========================================
def process_single_stock(code):
    time.sleep(random.uniform(0.5, 1.5)) # 🔴 强制休眠伪装人类
    
    code = str(code).zfill(6)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=200) 
    
    try:
        df_k = robust_akshare_call(
            ak.stock_zh_a_hist, symbol=code, period="daily", 
            start_date=start_date.strftime("%Y%m%d"), 
            end_date=end_date.strftime("%Y%m%d"), adjust="qfq"
        )
        
        if df_k is None or df_k.empty or len(df_k) < 65: 
            return None 
            
        df_k['date'] = pd.to_datetime(df_k['日期'])
        df_k['close'] = df_k['收盘'].astype(float)
        df_k['open'] = df_k['开盘'].astype(float)
        df_k['high'] = df_k['最高'].astype(float)
        df_k['low'] = df_k['最低'].astype(float) 
        df_k['vol'] = df_k['成交量'].astype(float)
        df_k['pct_change'] = df_k['涨跌幅'].astype(float)
        
        df_k['MA5'] = df_k['close'].rolling(window=5).mean()
        df_k['MA10'] = df_k['close'].rolling(window=10).mean()
        df_k['MA20'] = df_k['close'].rolling(window=20).mean()
        df_k['MA60'] = df_k['close'].rolling(window=60).mean() 
        df_k['Vol_MA20'] = df_k['vol'].rolling(window=20).mean()
        df_k['BIAS20'] = (df_k['close'] - df_k['MA20']) / df_k['MA20']
        
        exp1 = df_k['close'].ewm(span=12, adjust=False).mean()
        exp2 = df_k['close'].ewm(span=26, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        df_k['MACD_hist'] = macd_line - signal_line

        df_k['is_limit_up'] = (df_k['pct_change'] >= 9.5).astype(int)
        df_k['recent_limit_up'] = df_k['is_limit_up'].rolling(window=20).sum()
        df_k['rolling_high_20'] = df_k['high'].rolling(window=20).max().shift(1)

        df_k['body_size'] = abs(df_k['close'] - df_k['open'])
        df_k['upper_shadow'] = df_k['high'] - df_k[['close', 'open']].max(axis=1)
        df_k['shadow_ratio'] = df_k['upper_shadow'] / (df_k['body_size'] + 0.001)

        df_k['rolling_high_10'] = df_k['high'].rolling(window=10).max().shift(1)
        df_k['rolling_low_10'] = df_k['low'].rolling(window=10).min().shift(1)
        df_k['consolidation_range'] = (df_k['rolling_high_10'] - df_k['rolling_low_10']) / df_k['rolling_low_10']

        today = df_k.iloc[-1]
        yesterday = df_k.iloc[-2]
        
        BASE_SCORE = 50
        final_score = BASE_SCORE

        if not IS_MARKET_GOOD: final_score -= 30

        if today['close'] > today['MA20'] and today['MA20'] > today['MA60']: final_score += 15  
        elif today['close'] < today['MA60']: final_score -= 30  
        if today['close'] > today['MA5'] and today['MA5'] > today['MA10'] and today['MA10'] > today['MA20']: final_score += 15  

        if today['recent_limit_up'] >= 1: final_score += 10  
        else: final_score -= 5   

        if today['close'] > today['rolling_high_20']:
            final_score += 15  
            if today['consolidation_range'] < 0.12: final_score += 20  
            elif today['consolidation_range'] > 0.30: final_score -= 15 

        vol_ratio = today['vol'] / yesterday['Vol_MA20'] if yesterday['Vol_MA20'] > 0 else 1
        if 1.5 <= vol_ratio <= 4.0 and today['close'] > today['open']: final_score += 15  
        elif vol_ratio > 5.0: final_score -= 15  
        elif vol_ratio < 0.6 and today['close'] < today['open']: final_score -= 10  

        if today['MACD_hist'] > yesterday['MACD_hist'] and today['MACD_hist'] > 0: final_score += 10  
        bias_val = today['BIAS20']
        if bias_val > 0.15: final_score -= 20  
        elif -0.02 <= bias_val <= 0.08: final_score += 10  

        if today['shadow_ratio'] > 2.0 and today['upper_shadow'] > today['close'] * 0.02:
            final_score -= 40  

        spot_info = spot_info_dict.get(code, {})
        latest_price = spot_info.get('最新价', today['close'])
        pct_change = spot_info.get('涨跌幅', 0.0)
        industry = spot_info.get('所属行业', '未知')
        turnover = spot_info.get('换手率', 0.0)

        if final_score >= 100: signal_text = "🔥 龙头上车-绝佳突破口"
        elif final_score >= 80: signal_text = "📈 强势共振-低吸待涨"
        elif final_score >= 60: signal_text = "⏳ 多空博弈-等待确认"
        else: signal_text = "☢️ 诱多陷阱-坚决规避"
        
        return {
            "代码": code, "名称": name_dict.get(code, code), "所属板块": industry,
            "动能得分": round(final_score, 1), "行动策略": signal_text,
            "最新价": round(latest_price, 2), "今日涨幅": f"{pct_change}%",
            "换手率": f"{turnover}%", "今日量比": round(vol_ratio, 2),
            "近20日涨停": int(today['recent_limit_up']), "20日乖离率": f"{bias_val*100:.2f}%"
        }
    except Exception as e:
        return None

# ==========================================
# 🚀 3. 多线程并发扫描 (降维防脱网)
# ==========================================
results = []
MAX_WORKERS = 3  # 🔴 控制在3线程并发，以时间换空间
print(f"⚡ 正在分配火力，开启 {MAX_WORKERS} 个并发线程...")

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    future_to_fund = {executor.submit(process_single_stock, code): code for code in stock_list}
    for future in as_completed(future_to_fund):
        try:
            data = future.result()
            if data and data['动能得分'] >= 70: 
                results.append(data)
        except Exception:
            pass

# ==========================================
# 4. 结果汇总、板块共振统计与邮件推送
# ==========================================
summary_text = ""
excel_filename = "A股主升浪动能突破名单.xlsx"

if IS_MARKET_GOOD:
    market_alert = f"🟢 【大盘环境安全】上证指数 ({SH_CLOSE:.2f}) 稳居20日线上方，情绪处于进攻期。\n"
else:
    market_alert = f"🚨 【系统风险警告】上证指数 ({SH_CLOSE:.2f}) 跌破20日线！开启「防御模式」！\n"

summary_text += market_alert + "=" * 45 + "\n"

if len(results) > 0:
    df_output = pd.DataFrame(results)
    sector_counts = df_output['所属板块'].value_counts()
    hot_sectors = sector_counts[sector_counts >= 2].index.tolist()
    
    if hot_sectors:
        summary_text += f"🌪️ 【系统侦测主线板块】：{', '.join(hot_sectors)}\n"
        summary_text += "(*优先买入属于这些板块的标的，享受题材共振溢价*)\n"
        summary_text += "=" * 45 + "\n"
    
    df_output.sort_values(by=["所属板块", "动能得分"], ascending=[True, False], inplace=True)
    top10 = df_output.head(10)
    summary_text += "🎯 【V3.0 Pro 游资起爆点 Top 10】\n"
    summary_text += "-" * 45 + "\n"
    for idx, row in top10.iterrows():
        gene_str = "🔥有涨停基因" if row['近20日涨停'] > 0 else "无涨停基因"
        summary_text += f"▪️ {row['名称']} ({row['代码']}) - 【{row['所属板块']}】\n"
        summary_text += f"   得分: {row['动能得分']} | {gene_str} | 状态: {row['行动策略']}\n"
        summary_text += f"   真实量比: {row['今日量比']}倍 | 偏离度: {row['20日乖离率']} | 换手率: {row['换手率']}\n"
        summary_text += "-" * 45 + "\n"
    
    df_output.to_excel(excel_filename, index=False, sheet_name='主升浪猎手')
    print(f"\n🎉 演算完毕！共捕获 {len(df_output)} 只起爆标的。")
else:
    print("\n⚠️ 今日无符合强势动能特征的标的。")
    summary_text += "📉 今日打分系统【交白卷】！没有股票通过严苛审核，请持币观望。\n"

def send_excel_via_email(file_path, email_body_summary):
    sender = os.getenv("EMAIL_SENDER")
    password = os.getenv("EMAIL_PASSWORD") 
    receiver = os.getenv("EMAIL_RECEIVER")
    smtp_server = os.getenv("SMTP_SERVER", "smtp.qq.com")
    smtp_port = int(os.getenv("SMTP_PORT", 465))
    
    if not all([sender, password, receiver]):
        print("⚠️ GitHub Secrets 未完全配置，跳过邮件发送。日志已在控制台输出。")
        return

    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = receiver
    
    date_str = datetime.now().strftime('%Y-%m-%d')
    msg['Subject'] = f"🚀 游资雷达：A股主升浪突破阵型 ({date_str})" if IS_MARKET_GOOD else f"🚨 警报：大盘破位！防御报告 ({date_str})"
    
    body = f"主人您好，今日的《V3.0 Pro 主升浪猎杀名单》已生成。\n\n{email_body_summary}\n—— 自动量化机器人 敬上\n"
    msg.attach(MIMEText(body, 'plain', 'utf-8'))
    
    if os.path.exists(file_path):
        try:
            with open(file_path, "rb") as f:
                part = MIMEApplication(f.read(), Name=os.path.basename(file_path))
            part['Content-Disposition'] = f'attachment; filename="{os.path.basename(file_path)}"'
            msg.attach(part)
        except Exception as e:
            pass
        
    try:
        server = smtplib.SMTP_SSL(smtp_server, smtp_port)
        server.login(sender, password)
        server.send_message(msg)
        server.quit()
        print("✅ 股票动能邮件发送成功！")
    except Exception as e:
        print(f"❌ 邮件发送失败: {e}")

send_excel_via_email(excel_filename, summary_text)
