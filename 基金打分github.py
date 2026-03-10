import requests
import pandas as pd
import numpy as np
import re
import json
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import os
from datetime import datetime

print("🚀 启动全维量化大脑 (动态网格步长 + 夏普/索提诺动态加权 + 邮件全自动交付)...")

# 1. 读取基金池
try:
    df_input = pd.read_excel('我的基金池.xlsx', dtype={'基金代码': str})
    fund_list = df_input['基金代码'].dropna().str.strip().tolist()
    print(f"✅ 成功读取 {len(fund_list)} 只基金！")
except:
    print("⚠️ 未找到 '我的基金池.xlsx'，进入测试模式...")
    fund_list = ['005827', '110011', '510300', '159915'] 

results = []
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
current_year = datetime.now().year
RISK_FREE_RATE = 0.02

# 2. 核心演算循环
for code in fund_list:
    code = str(code).zfill(6)
    print(f"正在深度演算: {code} ...", end=" ")
    
    url = f"http://fund.eastmoney.com/pingzhongdata/{code}.js"
    try:
        res = requests.get(url, headers=headers, timeout=15)
        res.encoding = 'utf-8'
        text = res.text
        
        if "Data_netWorthTrend" in text:
            name_match = re.search(r'var fS_name = "(.*?)";', text)
            fund_name = name_match.group(1) if name_match else ""
            
            trend_match = re.search(r'var Data_netWorthTrend = (\[.*?\]);', text)
            if trend_match:
                trend_data = json.loads(trend_match.group(1))
                if len(trend_data) > 0:
                    df_k = pd.DataFrame(trend_data)
                    df_k['date'] = pd.to_datetime(df_k['x'], unit='ms')
                    df_k['close_raw'] = df_k['y'].astype(float) 
                    
                    df_k['ret'] = df_k['equityReturn'].fillna(0).astype(float) / 100.0
                    df_k['cum_growth'] = (1 + df_k['ret']).cumprod()
                    factor = df_k.iloc[-1]['close_raw'] / df_k.iloc[-1]['cum_growth']
                    df_k['close'] = df_k['cum_growth'] * factor
                    
                    df_k['MA20'] = df_k['close_raw'].rolling(window=20).mean()
                    df_k['STD20'] = df_k['close_raw'].rolling(window=20).std()
                    df_k['BIAS'] = (df_k['close_raw'] - df_k['MA20']) / df_k['MA20']
                    
                    exp1 = df_k['close'].ewm(span=12, adjust=False).mean()
                    exp2 = df_k['close'].ewm(span=26, adjust=False).mean()
                    macd_line = exp1 - exp2
                    signal_line = macd_line.ewm(span=9, adjust=False).mean()
                    df_k['MACD_hist'] = macd_line - signal_line
                    
                    macd_hist_trend = 'neutral'
                    if len(df_k) >= 2:
                        if df_k.iloc[-1]['MACD_hist'] > df_k.iloc[-2]['MACD_hist']:
                            macd_hist_trend = 'improving'
                        elif df_k.iloc[-1]['MACD_hist'] < df_k.iloc[-2]['MACD_hist']:
                            macd_hist_trend = 'deteriorating'

                    latest = df_k.iloc[-1]
                    cur_price_display = latest['close_raw'] 
                    cur_price_adj = latest['close'] 
                    daily_pct = latest['ret']
                    bias_val = latest['BIAS']
                    
                    df_1y = df_k.tail(250)
                    df_3y = df_k.tail(750)
                    
                    max_1y, min_1y = df_1y['close'].max(), df_1y['close'].min()
                    max_3y, min_3y = df_3y['close'].max(), df_3y['close'].min()
                    pct_1y = (cur_price_adj - min_1y) / (max_1y - min_1y) if max_1y > min_1y else None
                    pct_3y = (cur_price_adj - min_3y) / (max_3y - min_3y) if max_3y > min_3y else None
                    
                    if len(df_1y) > 10:
                        ret_ann_1y = df_1y['ret'].mean() * 250
                        vol_1y = df_1y['ret'].std() * np.sqrt(250)
                        rolling_max = df_1y['close'].cummax()
                        max_dd_1y = ((df_1y['close'] - rolling_max) / rolling_max).min()
                        sharpe_1y = (ret_ann_1y - RISK_FREE_RATE) / vol_1y if vol_1y > 0 else None
                        
                        downside_returns = df_1y[df_1y['ret'] < 0]['ret']
                        downside_vol_1y = downside_returns.std() * np.sqrt(250) if len(downside_returns) > 0 else None
                        sortino_1y = (ret_ann_1y - RISK_FREE_RATE) / downside_vol_1y if downside_vol_1y and downside_vol_1y > 0 else None
                    else:
                        vol_1y = max_dd_1y = sharpe_1y = sortino_1y = None

                    grid_step = max(0.015, min(0.05, vol_1y / 10.0)) if vol_1y else None

                    name_str = str(fund_name)
                    fund_type = '债券型' if any(k in name_str for k in ['债', '理财']) else '权益类'
                    is_sector = any(k in name_str for k in ['芯片', '半导体', '医药', '新能源', '酒', '军工', '医疗', '光伏', '煤炭', '消费'])
                    
                    BASE_SCORE = 50
                    final_score = BASE_SCORE

                    if fund_type == '权益类':
                        if pct_1y is not None:
                            if pct_1y < 0.10: final_score += 35  
                            elif pct_1y < 0.20: final_score += 25
                            elif pct_1y < 0.30: final_score += 15
                            elif pct_1y < 0.50: final_score += 5   
                            elif pct_1y > 0.90: final_score -= 35  
                            elif pct_1y > 0.80: final_score -= 25
                            elif pct_1y > 0.70: final_score -= 15
                            elif pct_1y > 0.50: final_score -= 5   
                        
                        if pct_3y is not None and pct_1y is not None:
                            if pct_3y < 0.20 and pct_1y < 0.30: final_score += 15

                        if bias_val is not None:
                            bias_multiplier = 1.5 if is_sector else 1.0 
                            if bias_val < -0.08 * bias_multiplier: final_score += 20
                            elif bias_val < -0.05 * bias_multiplier: final_score += 10
                            elif bias_val > 0.10 * bias_multiplier: final_score -= 20
                            elif bias_val > 0.07 * bias_multiplier: final_score -= 10

                        if vol_1y is not None:
                            if vol_1y < 0.10: final_score -= 10      
                            elif 0.25 <= vol_1y <= 0.45: final_score += 10 
                        
                        if max_dd_1y is not None and max_dd_1y < -0.45: final_score -= 10 

                        if sharpe_1y is not None:
                            final_score += max(-15, min(15, (sharpe_1y - 0.5) * 10))
                        if sortino_1y is not None:
                            final_score += max(-10, min(15, (sortino_1y - 0.8) * 8))

                        if macd_hist_trend == 'improving': final_score += 5
                        elif macd_hist_trend == 'deteriorating': final_score -= 5
                    else:
                        if max_dd_1y is not None:
                            if max_dd_1y > -0.01: final_score += 20
                            elif max_dd_1y < -0.04: final_score -= 30
                        if sharpe_1y is not None:
                            final_score += max(-15, min(20, (sharpe_1y - 1.0) * 15))
                        if pct_1y is not None:
                            if pct_1y < 0.30: final_score += 10 
                            elif pct_1y > 0.85: final_score -= 15

                    if final_score >= 110: signal_text = "☄️ 绝对冰点-砸锅卖铁" 
                    elif final_score >= 85: signal_text = "🔥 极度低估-强烈买入"
                    elif final_score >= 70: signal_text = "💰 优质低估-分批建仓"
                    elif final_score <= 0: signal_text = "☢️ 泡沫破灭-无脑清仓"
                    elif final_score <= 20: signal_text = "⚠️ 极度危险-强制止盈"
                    elif final_score <= 40: signal_text = "📉 高位风险-逢高减磅"
                    else: signal_text = "⏳ 估值合理-底仓持有"
                    
                    def get_return(trading_days):
                        if len(df_k) > trading_days:
                            base_price = df_k.iloc[-1 - trading_days]['close']
                            return (cur_price_adj - base_price) / base_price
                        return None
                    
                    last_year_df = df_k[df_k['date'].dt.year < current_year]
                    ytd_ret = (cur_price_adj - last_year_df.iloc[-1]['close']) / last_year_df.iloc[-1]['close'] if not last_year_df.empty else None
                    
                    fund_data = {
                        "代码": code,
                        "名称": fund_name,
                        "综合得分": round(final_score, 1),
                        "操作评级": signal_text,
                        "最佳网格步长": grid_step,
                        "夏普比率(1年)": round(sharpe_1y, 2) if pd.notnull(sharpe_1y) else None,
                        "索提诺比率(1年)": round(sortino_1y, 2) if pd.notnull(sortino_1y) else None,
                        "最新净值": round(cur_price_display, 4),
                        "日波动(涨跌)": daily_pct,
                        "1年百分位": pct_1y,
                        "3年百分位": pct_3y,
                        "乖离率(20日)": bias_val,
                        "年化波动率": vol_1y,
                        "最大回撤": max_dd_1y,
                        "资产类型": fund_type,
                        "今年以来": ytd_ret,
                        "近1月": get_return(20),
                        "近3月": get_return(60),
                        "近1年": get_return(250),
                        "近3年": get_return(750)
                    }
                    results.append(fund_data)
                    print(f"✅ 得分: {final_score:.1f}分")
                else:
                    print("❌ 数组为空")
            else:
                print("❌ 解析失败")
        else:
            print("❌ CDN拦截")
    except Exception as e:
        print(f"❌ 异常: {e}")

# 3. 生成 Excel 并提取邮件摘要
summary_text = ""
if len(results) > 0:
    df_output = pd.DataFrame(results)
    df_output.sort_values(by="综合得分", ascending=False, inplace=True)
    
    # 提取排名前三的基金作为邮件摘要
    top3 = df_output.head(3)
    summary_text += "🏆 【今日得分 Top 3 基金】\n"
    summary_text += "-" * 30 + "\n"
    for idx, row in top3.iterrows():
        summary_text += f"▪️ {row['名称']} ({row['代码']})\n"
        summary_text += f"   得分: {row['综合得分']} | 评级: {row['操作评级']}\n"
        summary_text += f"   1年百分位: {row['1年百分位']*100:.1f}% | 回撤: {row['最大回撤']*100:.1f}%\n"
        summary_text += "-" * 30 + "\n"
    
    pct_cols = ["日波动(涨跌)", "最佳网格步长", "1年百分位", "3年百分位", "乖离率(20日)", "年化波动率", "最大回撤", "今年以来", "近1月", "近3月", "近1年", "近3年"]
    
    excel_filename = "全量基金智能打分表.xlsx"
    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        df_output.to_excel(writer, index=False, sheet_name='量化复盘')
        worksheet = writer.sheets['量化复盘']
        for col_idx, col_name in enumerate(df_output.columns, 1):
            if col_name in pct_cols:
                for row_idx in range(2, len(df_output) + 2):
                    cell = worksheet.cell(row=row_idx, column=col_idx)
                    if isinstance(cell.value, (int, float)):
                        cell.number_format = '0.00%'
                        
    print(f"\n🎉 完美收官！全息动态加权矩阵已生成！")
else:
    print("\n⚠️ 抓取失败。")

# 4. 邮件自动推送模块
def send_excel_via_email(file_path, email_body_summary):
    sender = os.getenv("EMAIL_SENDER")
    password = os.getenv("EMAIL_PASSWORD") 
    receiver = os.getenv("EMAIL_RECEIVER")
    smtp_server = os.getenv("SMTP_SERVER", "smtp.qq.com")
    smtp_port = int(os.getenv("SMTP_PORT", 465))
    
    if not all([sender, password, receiver]):
        print("\n⚠️ 未配置邮箱环境变量 (Secrets)，跳过邮件发送步骤。")
        return

    print(f"\n📧 正在打包表格，准备发送至邮箱: {receiver} ...")
    
    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = receiver
    msg['Subject'] = f"📊 量化大脑：全量基金智能打分复盘报告 ({datetime.now().strftime('%Y-%m-%d')})"
    
    body = (
        f"主人您好，今日的《全量基金智能打分表》演算完毕，请查收附件。\n\n"
        f"{email_body_summary}\n"
        f"—— 自动量化机器人 敬上\n"
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    msg.attach(MIMEText(body, 'plain', 'utf-8'))
    
    try:
        with open(file_path, "rb") as f:
            part = MIMEApplication(f.read(), Name=os.path.basename(file_path))
        part['Content-Disposition'] = f'attachment; filename="{os.path.basename(file_path)}"'
        msg.attach(part)
    except Exception as e:
        print(f"❌ 读取附件失败: {e}")
        return
        
    try:
        server = smtplib.SMTP_SSL(smtp_server, smtp_port)
        server.login(sender, password)
        server.send_message(msg)
        server.quit()
        print("✅ 邮件发送成功！请检查您的收件箱。")
    except Exception as e:
        print(f"❌ 邮件发送失败: {e}")

if len(results) > 0 and os.path.exists(excel_filename):
    send_excel_via_email(excel_filename, summary_text)
