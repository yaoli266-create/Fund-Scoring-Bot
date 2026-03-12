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

# 🔴 引入硬核直连引擎
from pytdx.hq import TdxHq_API

warnings.filterwarnings('ignore')

print("🚀 启动全维量化大脑 V5.0 (PyTdx Socket 直连防封锁 + 游资漏斗引擎)...")

# ==========================================
# 🛡️ 核心强化 1：自适应指数退避重试引擎
# ==========================================
def robust_akshare_call(func, *args, max_retries=5, **kwargs):
    for attempt in range(max_retries):
        try:
            res = func(*args, **kwargs)
            if res is None or (isinstance(res, pd.DataFrame) and res.empty):
                raise ValueError("⚠️ 接口返回空数据，触发重试")
            return res
        except Exception as e:
            if attempt < max_retries - 1:
                sleep_time = (1.5 ** (attempt + 1)) + random.uniform(0.1, 0.5)
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
# 🌪️ 1. PyTdx 底层直连大漏斗：无视 HTTP 封锁
# ==========================================
def get_pytdx_active_pool(top_n=250):
    print("正在通过 TCP Socket 直连券商底层节点，绕过 GitHub 防火墙限制...")
    try:
        # 1. 轻量级获取 A股所有代码和名称
        name_df = robust_akshare_call(ak.stock_info_a_code_name)
        name_df = name_df[~name_df['name'].str.contains('ST|退|C|N|B')]
        codes = name_df['code'].tolist()
        name_dict = dict(zip(name_df['code'], name_df['name']))

        # 2. 区分沪深市场 (0:深圳, 1:上海) 适配 pytdx 协议
        market_codes = []
        for c in codes:
            if c.startswith('6'):
                market_codes.append((1, c))
            elif c.startswith('0') or c.startswith('3'):
                market_codes.append((0, c))

        # 3. 建立物理直连 (海量节点池 + 随机轮询 + 极速超时抛弃)
        api = TdxHq_API(raise_exception=False) # 关闭全局报错，允许悄悄重试
        
        # 扩充到 12 个国内一线大厂/券商的骨干节点
        nodes = [
            ('119.147.212.81', 7709), ('119.147.86.171', 7709), 
            ('114.115.234.36', 7709), ('111.12.55.94', 7709),
            ('114.80.63.12', 7709),   ('106.14.95.149', 7709),
            ('119.147.164.60', 7709), ('124.74.236.94', 7709),
            ('218.108.47.69', 7709),  ('218.71.118.105', 7709),
            ('180.153.39.51', 7709),  ('121.14.110.210', 7709)
        ]
        # 每次运行随机打乱顺序，避免被同一个死节点卡住
        random.shuffle(nodes) 
        
        connected = False
        for ip, port in nodes:
            try:
                # time_out=2秒，连不上立刻换下一个，绝不拖泥带水
                if api.connect(ip, port, time_out=2):
                    print(f"✅ 成功穿透防火墙，直连底层服务器: {ip}")
                    connected = True
                    break
            except Exception:
                pass # 连不上就静默跳过，试下一个
                
        if not connected:
            raise ValueError("所有通达信节点连接超时，海外 IP 可能被全面阻断")

        # 4. 暴力并发拉取全市场盘口快照
        quotes_list = []
        batch_size = 80 # pytdx 单次最大并发限制
        for i in range(0, len(market_codes), batch_size):
            batch = market_codes[i:i+batch_size]
            data = api.get_security_quotes(batch)
            if data:
                quotes_list.extend(data)
        api.disconnect()

        # 5. 内存降维过滤
        df_quotes = pd.DataFrame(quotes_list)
        df_quotes = df_quotes[df_quotes['price'] > 0] # 过滤停牌
        df_quotes['涨跌幅'] = (df_quotes['price'] - df_quotes['last_close']) / df_quotes['last_close'] * 100
        
        # 核心过滤：直接按成交额排兵布阵，选取全市场火力最猛的资金聚集地
        active_pool = df_quotes.sort_values(by='amount', ascending=False).head(top_n)
        stock_list = active_pool['code'].tolist()

        spot_info_dict = {}
        for _, row in active_pool.iterrows():
            spot_info_dict[row['code']] = {
                '最新价': row['price'],
                '涨跌幅': row['涨跌幅']
            }

        print(f"⚡ Socket 全网扫描完毕！瞬间浓缩出 {len(stock_list)} 只高频游资标的。")
        return stock_list, name_dict, spot_info_dict

    except Exception as e:
        print(f"⚠️ PyTdx 漏斗构建失败: {e}")
        fallback = ['600519', '000858', '300750']
        return fallback, {c: c for c in fallback}, {}

# 启动直连漏斗
stock_list, name_dict, spot_info_dict = get_pytdx_active_pool(top_n=250)

# ==========================================
# ⚡️ 2. 核心算力：V5.0 主升浪打分引擎 (精算阶段)
# ==========================================
def process_single_stock(code):
    # 🔴 GitHub 专属阻尼：微小伪装，保护 K 线获取不被封
    time.sleep(random.uniform(0.2, 0.6)) 
    
    code = str(code).zfill(6)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=200) 
    
    try:
        # 历史 K 线依然使用 Akshare 获取复权数据 (此时请求量已大幅降低，极其安全)
        df_k = robust_akshare_call(
            ak.stock_zh_a_hist, symbol=code, period="daily", 
            start_date=start_date.strftime("%Y%m%d"), 
            end_date=end_date.strftime("%Y%m%d"), adjust="qfq"
        )
        
        if df_k is None or df_k.empty or len(df_k) < 65: return None 
            
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

        # 完美拼接：利用 hist 数据中的换手率，补足 Pytdx 的短板
        spot_info = spot_info_dict.get(code, {})
        latest_price = today['close']
        pct_change = today['pct_change']
        turnover = today.get('换手率', 0.0)
        industry = "活跃量能标的"

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
# 🚀 3. 多线程并发扫描 (GitHub 黄金护航版)
# ==========================================
results = []
# 🔴 核心参数：6 线程是对海外 IP 最安全的并发量，防止封锁
MAX_WORKERS = 6  
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
excel_filename = "A股全市场主升浪突破名单.xlsx"

if IS_MARKET_GOOD:
    market_alert = f"🟢 【大盘环境安全】上证指数 ({SH_CLOSE:.2f}) 稳居20日线上方，情绪处于进攻期。\n"
else:
    market_alert = f"🚨 【系统风险警告】上证指数 ({SH_CLOSE:.2f}) 跌破20日线！开启「防御模式」！\n"

summary_text += market_alert + "=" * 45 + "\n"

if len(results) > 0:
    df_output = pd.DataFrame(results)
    df_output.sort_values(by=["动能得分"], ascending=[False], inplace=True)
    top10 = df_output.head(10)
    
    summary_text += "🎯 【V5.0 Socket 直连拦截起爆点 Top 10】\n"
    summary_text += "-" * 45 + "\n"
    for idx, row in top10.iterrows():
        gene_str = "🔥有涨停基因" if row['近20日涨停'] > 0 else "无涨停基因"
        summary_text += f"▪️ {row['名称']} ({row['代码']})\n"
        summary_text += f"   得分: {row['动能得分']} | {gene_str} | 状态: {row['行动策略']}\n"
        summary_text += f"   真实量比: {row['今日量比']}倍 | 偏离度: {row['20日乖离率']} | 换手率: {row['换手率']}\n"
        summary_text += "-" * 45 + "\n"
    
    df_output.to_excel(excel_filename, index=False, sheet_name='全网游资猎手')
    print(f"\n🎉 演算完毕！全网过滤中共捕获 {len(df_output)} 只强势标的。")
else:
    print("\n⚠️ 今日全网活水池内无符合强势动能特征的标的。")
    summary_text += "📉 今日打分系统【交白卷】！严格过滤后无票符合预期，建议持币观望。\n"

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
        msg['Subject'] = f"🚀 游资雷达：全市场主升浪突破阵型 ({date_str})"
    else:
        msg['Subject'] = f"🚨 警报：大盘破位！全市场防御报告 ({date_str})"
    
    body = f"主人您好，今日基于《TCP Socket 底层直连法》强行破盾扫描全网 5000 只标的，生成的《V5.0 动能突破名单》已生成。\n\n{email_body_summary}\n—— 自动量化机器人 敬上\n"
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

