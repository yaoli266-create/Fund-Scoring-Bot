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

print("🚀 启动全维量化大脑 V3.0 Pro (本地极速版 + 自适应防封锁装甲)...")

# ==========================================
# 🛡️ 核心强化 1：自适应指数退避重试引擎
# ==========================================
def robust_akshare_call(func, *args, max_retries=5, **kwargs):
    """
    不仅捕获网络崩溃，还能精准识别东方财富返回的“幽灵空数据”并触发重试。
    采用指数级退避算法 (Exponential Backoff)，越失败等得越久，保护本地 IP。
    """
    for attempt in range(max_retries):
        try:
            res = func(*args, **kwargs)
            # 核心防御：识别接口返回的空数据（假成功，真限流）
            if res is None or (isinstance(res, pd.DataFrame) and res.empty):
                raise ValueError("⚠️ 接口返回空数据，疑似触发反爬限流阈值")
            return res
        except Exception as e:
            if attempt < max_retries - 1:
                # 指数退避：重试间隔为 1.5s, 3.25s, 6.6s... 迅速降温
                sleep_time = (1.5 ** (attempt + 1)) + random.uniform(0.1, 0.5)
                # print(f"  [引擎降温] 触发限流，{sleep_time:.1f}秒后发起第 {attempt+2} 次突围...")
                time.sleep(sleep_time)
            else:
                print(f"❌ 接口彻底熔断: {e}")
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
# 🔍 1. 构建轻量化名称字典 & 读取本地股票池
# ==========================================
print("正在拉取A股基础名称字典 (轻量级请求)...")
try:
    name_df = robust_akshare_call(ak.stock_info_a_code_name)
    name_dict = dict(zip(name_df['code'], name_df['name']))
except Exception:
    name_dict = {}

def get_custom_stock_pool():
    print("正在读取自定义股票池...")
    try:
        df_input = pd.read_excel('我的股票池.xlsx', dtype={'股票代码': str})
        stock_list = df_input['股票代码'].dropna().str.strip().tolist()
        print(f"✅ 成功读取 {len(stock_list)} 只目标股票！准备开启量价诊断...")
        return stock_list
    except Exception as e:
        print(f"⚠️ 未找到 '我的股票池.xlsx'，进入核心兜底票池模式...")
        fallback_list = ['600519', '000858', '002594', '300750', '300059', '600030', '000333', '601318', '000001', '600036']
        return fallback_list

stock_list = get_custom_stock_pool()

# ==========================================
# ⚡️ 2. 核心算力：V3.0 Pro 游资主升浪打分引擎
# ==========================================
def process_single_stock(code):
    # 🛡️ 核心强化 2：移除长休眠，改为毫秒级微小抖动，防止并发瞬间撕裂接口
    time.sleep(random.uniform(0.05, 0.15)) 
    
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

        latest_price = today['close']
        pct_change = today['pct_change']
        turnover = today.get('换手率', 0.0) 
        industry = "自选标的" 

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
# 🚀 3. 多线程并发扫描 (火力全开模式)
# ==========================================
results = []
# 🛡️ 核心强化 3：本地运行解禁，并发从 3 提升至 12，速度提升近 4 倍
MAX_WORKERS = 12  
print(f"⚡ 正在分配火力，开启 {MAX_WORKERS} 个并发线程执行标的狙击...")

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    future_to_fund = {executor.submit(process_single_stock, code): code for code in stock_list}
    for future in as_completed(future_to_fund):
        stock_code = future_to_fund[future]
        try:
            data = future.result()
            if data and data['动能得分'] >= 60: 
                results.append(data)
                print(f"✅ 捕获异动: {stock_code} ({data['名称']}) | 动能: {data['动能得分']}")
        except Exception:
            pass

# ==========================================
# 4. 结果汇总、板块共振统计与邮件推送
# ==========================================
summary_text = ""
excel_filename = "A股自选池动能突破名单.xlsx"

if IS_MARKET_GOOD:
    market_alert = f"🟢 【大盘环境安全】上证指数 ({SH_CLOSE:.2f}) 稳居20日线上方，情绪处于进攻期。\n"
else:
    market_alert = f"🚨 【系统风险警告】上证指数 ({SH_CLOSE:.2f}) 跌破20日线！开启「防御模式」！\n"

summary_text += market_alert + "=" * 45 + "\n"

if len(results) > 0:
    df_output = pd.DataFrame(results)
    df_output.sort_values(by=["动能得分"], ascending=[False], inplace=True)
    top10 = df_output.head(10)
    
    summary_text += "🎯 【V3.0 Pro 自选池起爆点 Top 10】\n"
    summary_text += "-" * 45 + "\n"
    for idx, row in top10.iterrows():
        gene_str = "🔥有涨停基因" if row['近20日涨停'] > 0 else "无涨停基因"
        summary_text += f"▪️ {row['名称']} ({row['代码']})\n"
        summary_text += f"   得分: {row['动能得分']} | {gene_str} | 状态: {row['行动策略']}\n"
        summary_text += f"   真实量比: {row['今日量比']}倍 | 偏离度: {row['20日乖离率']} | 换手率: {row['换手率']}\n"
        summary_text += "-" * 45 + "\n"
    
    df_output.to_excel(excel_filename, index=False, sheet_name='自选股猎手')
    print(f"\n🎉 演算完毕！自选池中共捕获 {len(df_output)} 只强势标的。")
else:
    print("\n⚠️ 今日自选池内无符合强势动能特征的标的。")
    summary_text += "📉 今日打分系统【交白卷】！您的自选池内没有股票通过严苛的动能审核，建议持币观望。\n"

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
    
    if IS_MARKET_GOOD:
        msg['Subject'] = f"🚀 游资雷达：自选池主升浪突破阵型 ({date_str})"
    else:
        msg['Subject'] = f"🚨 警报：大盘破位！自选池防御报告 ({date_str})"
    
    body = f"主人您好，今日基于您的股票池生成的《V3.0 Pro 动能突破名单》已生成。\n\n{email_body_summary}\n—— 自动量化机器人 敬上\n"
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
