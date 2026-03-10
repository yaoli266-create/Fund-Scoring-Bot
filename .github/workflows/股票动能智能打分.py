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

warnings.filterwarnings('ignore')

print("🚀 启动全维量化大脑 (A股个股专属 - 右侧动能突破打分模型)...")

# ==========================================
# 🔍 1. 实时粗筛：截取全市场最活跃资金池
# ==========================================
def get_active_stock_pool(top_n=500):
    print(f"正在扫描全市场，锁定今日成交最活跃的 Top {top_n} 只标的...")
    try:
        spot_df = ak.stock_zh_a_spot_em()
        
        # 基础风控：过滤 ST、退市股
        spot_df = spot_df[~spot_df['名称'].str.contains('ST|退|C|N')]
        
        # 形态过滤：剔除一字涨停（涨幅>9.8%且最高价=最低价=开盘价），散户买不进，毫无意义
        # 剔除一字跌停（跌幅<-9.8%）
        spot_df = spot_df[(spot_df['涨跌幅'] < 9.8) & (spot_df['涨跌幅'] > -9.8)]
        
        # 按“成交额”降序排列，资金去哪我们就去哪
        active_pool = spot_df.sort_values(by='成交额', ascending=False).head(top_n)
        
        stock_list = active_pool['代码'].tolist()
        name_dict = dict(zip(active_pool['代码'], active_pool['名称']))
        spot_info_dict = active_pool.set_index('代码').to_dict('index')
        
        print(f"✅ 成功锁定 {len(stock_list)} 只高流动性标的，开始深度量价诊断...")
        return stock_list, name_dict, spot_info_dict
    except Exception as e:
        print(f"⚠️ 粗筛失败: {e}")
        # 兜底测试池（包含部分科技主线、红利与消费代表）
        fallback_list = ['600519', '000858', '603993', '000063', '601899', '600362']
        return fallback_list, {c: c for c in fallback_list}, {}

stock_list, name_dict, spot_info_dict = get_active_stock_pool(top_n=500)

# ==========================================
# ⚡️ 2. 核心算力：个股专属动能打分逻辑
# ==========================================
def process_single_stock(code):
    code = str(code).zfill(6)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=120) # 股票算短期动能，半年数据足矣
    
    try:
        df_k = ak.stock_zh_a_hist(
            symbol=code, period="daily", 
            start_date=start_date.strftime("%Y%m%d"), 
            end_date=end_date.strftime("%Y%m%d"), adjust="qfq"
        )
        
        if df_k is None or df_k.empty or len(df_k) < 20:
            return None 
            
        df_k['date'] = pd.to_datetime(df_k['日期'])
        df_k['close'] = df_k['收盘'].astype(float)
        df_k['open'] = df_k['开盘'].astype(float)
        df_k['vol'] = df_k['成交量'].astype(float)
        
        df_k['MA5'] = df_k['close'].rolling(window=5).mean()
        df_k['MA10'] = df_k['close'].rolling(window=10).mean()
        df_k['MA20'] = df_k['close'].rolling(window=20).mean()
        df_k['Vol_MA5'] = df_k['vol'].rolling(window=5).mean()
        df_k['BIAS20'] = (df_k['close'] - df_k['MA20']) / df_k['MA20']
        
        exp1 = df_k['close'].ewm(span=12, adjust=False).mean()
        exp2 = df_k['close'].ewm(span=26, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        df_k['MACD_hist'] = macd_line - signal_line

        today = df_k.iloc[-1]
        yesterday = df_k.iloc[-2]
        
        # ---------------------------------------------------------
        # 🧠 股票动能专属打分模型 (Momentum & Breakout)
        # ---------------------------------------------------------
        BASE_SCORE = 50
        final_score = BASE_SCORE

        # 1. 趋势动能因子 (均线)
        if today['close'] > today['MA5'] and today['MA5'] > today['MA10'] and today['MA10'] > today['MA20']:
            final_score += 20  # 完美多头排列
        elif today['MA5'] < today['MA10']:
            final_score -= 30  # 空头排列，坚决不碰

        # 2. MACD 加速因子
        if today['MACD_hist'] > yesterday['MACD_hist'] and today['MACD_hist'] > 0:
            final_score += 10  # 零轴上方加速上扬
        
        # 3. 量能突破因子
        vol_ratio = today['vol'] / yesterday['Vol_MA5'] if yesterday['Vol_MA5'] > 0 else 1
        if 1.5 <= vol_ratio <= 3.5 and today['close'] > today['open']:
            final_score += 15  # 良性放量收阳
        elif vol_ratio > 4.0:
            final_score -= 15  # 天量滞涨或爆量，存在游资诱多风险
        elif vol_ratio < 0.6 and today['close'] < today['open']:
            final_score -= 10  # 缩量阴跌，人气涣散

        # 4. 乖离率防追高因子
        bias_val = today['BIAS20']
        if bias_val > 0.15:
            final_score -= 20  # 偏离20日线超过15%，极易短线回调
        elif -0.05 <= bias_val <= 0.05:
            final_score += 5   # 贴近均线起爆，盈亏比极佳

        # 获取今日实时表现
        spot_info = spot_info_dict.get(code, {})
        latest_price = spot_info.get('最新价', today['close'])
        pct_change = spot_info.get('涨跌幅', 0.0)
        industry = spot_info.get('所属行业', '未知')

        if final_score >= 85: signal_text = "🔥 强势主升-右侧伺机打板/半路"
        elif final_score >= 70: signal_text = "📈 趋势起爆-均线附近低吸"
        elif final_score >= 50: signal_text = "⏳ 多空博弈-等待方向选择"
        else: signal_text = "☢️ 破位/滞涨-坚决规避"
        
        return {
            "代码": code,
            "名称": name_dict.get(code, code),
            "所属板块": industry,
            "动能得分": round(final_score, 1),
            "行动策略": signal_text,
            "最新价": round(latest_price, 2),
            "今日涨幅": f"{pct_change}%",
            "今日量比": round(vol_ratio, 2),
            "20日乖离率": f"{bias_val*100:.2f}%"
        }
    except Exception as e:
        return None

# ==========================================
# 🚀 3. 多线程并发扫描
# ==========================================
results = []
MAX_WORKERS = 15 

print(f"⚡ 正在分配火力，开启 {MAX_WORKERS} 个并发线程执行动能量化...")

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    future_to_fund = {executor.submit(process_single_stock, code): code for code in stock_list}
    for future in as_completed(future_to_fund):
        try:
            data = future.result()
            if data and data['动能得分'] >= 60: # 只保留及格以上的股票，精简结果
                results.append(data)
        except Exception:
            pass

# ==========================================
# 4. 结果汇总与邮件推送
# ==========================================
summary_text = ""
excel_filename = "A股高潜动能突破名单.xlsx"

if len(results) > 0:
    df_output = pd.DataFrame(results)
    df_output.sort_values(by=["动能得分", "今日量比"], ascending=[False, False], inplace=True)
    
    # 提取 Top 10
    top10 = df_output.head(10)
    summary_text += "🎯 【今日A股游资/机构合力爆破 Top 10】\n"
    summary_text += "(*专注右侧动能，切勿左侧死扛。若次日跌破5日线需严格止损*)\n"
    summary_text += "-" * 40 + "\n"
    for idx, row in top10.iterrows():
        summary_text += f"▪️ {row['名称']} ({row['代码']}) - {row['所属板块']}\n"
        summary_text += f"   动能得分: {row['动能得分']} | 状态: {row['行动策略']}\n"
        summary_text += f"   今日涨幅: {row['今日涨幅']} | 放量倍数: {row['今日量比']}倍 | 偏离度: {row['20日乖离率']}\n"
        summary_text += "-" * 40 + "\n"
    
    df_output.to_excel(excel_filename, index=False, sheet_name='动能猎手')
    print(f"\n🎉 个股动能矩阵演算完毕！共捕获 {len(df_output)} 只起爆标的。")
else:
    print("\n⚠️ 今日无符合强势动能特征的标的（或行情数据获取失败）。")
    summary_text += "📉 今日无强势动能标的，市场极度萎靡或处于全面退潮期，管住手！\n"

# 邮件发送
def send_excel_via_email(file_path, email_body_summary):
    sender = os.getenv("EMAIL_SENDER")
    password = os.getenv("EMAIL_PASSWORD") 
    receiver = os.getenv("EMAIL_RECEIVER")
    smtp_server = os.getenv("SMTP_SERVER", "smtp.qq.com")
    smtp_port = int(os.getenv("SMTP_PORT", 465))
    
    if not all([sender, password, receiver]):
        print("⚠️ 未配置邮箱环境变量，跳过邮件发送。")
        return

    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = receiver
    msg['Subject'] = f"🚀 量化大脑：A股个股动能突破雷达 ({datetime.now().strftime('%Y-%m-%d')})"
    
    body = f"主人您好，今日的《A股高潜动能突破名单》已生成。\n\n{email_body_summary}\n—— 自动量化机器人 敬上\n"
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
