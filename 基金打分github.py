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

print("🚀 启动全维量化大脑 (多线程并发版 + 全局名称映射 + 绝对格式化)...")

# ==========================================
# 🔍 0. 构建全市场 ETF 名称映射字典 (极速查名)
# ==========================================
print("正在拉取全市场 ETF 名称字典...")
try:
    # 获取场内基金实时行情（包含代码和名称）
    spot_df = ak.fund_etf_spot_em()
    # 建立 { '510300': '沪深300ETF', ... } 的字典
    name_dict = dict(zip(spot_df['代码'], spot_df['名称']))
    print(f"✅ 成功构建名称映射字典，共收录 {len(name_dict)} 只场内基金。")
except Exception as e:
    print(f"⚠️ 名称映射字典获取失败，将以代码兜底: {e}")
    name_dict = {}

# ==========================================
# 1. 读取基金池
# ==========================================
try:
    df_input = pd.read_excel('我的基金池.xlsx', dtype={'基金代码': str})
    fund_list = df_input['基金代码'].dropna().str.strip().tolist()
    print(f"✅ 成功读取 {len(fund_list)} 只目标基金！准备开启多线程并发拉取...")
except:
    print("⚠️ 未找到 '我的基金池.xlsx'，进入测试模式...")
    fund_list = ['510300', '159915', '512890', '512170'] 

current_year = datetime.now().year
RISK_FREE_RATE = 0.02

# ==========================================
# ⚡️ 核心算力函数：单独处理每一只基金
# ==========================================
def process_single_fund(code):
    code = str(code).zfill(6)
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 3)
    
    try:
        df_k = ak.fund_etf_hist_em(
            symbol=code, 
            period="daily", 
            start_date=start_date.strftime("%Y%m%d"), 
            end_date=end_date.strftime("%Y%m%d"), 
            adjust="qfq"
        )
        
        if df_k is None or df_k.empty or len(df_k) < 20:
            return None 
            
        df_k['date'] = pd.to_datetime(df_k['日期'])
        df_k['close_raw'] = df_k['收盘'].astype(float)
        df_k['ret'] = df_k['涨跌幅'].astype(float) / 100.0 
        df_k['close'] = df_k['close_raw'] 
        
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

        # 💡 从全局字典中极速查名字，查不到就用代码顶替
        fund_name = name_dict.get(code, code) 
        
        fund_type = '债券/货币ETF' if str(code).startswith(('511', '1590')) else '权益ETF'
        is_sector = True 
        
        BASE_SCORE = 50
        final_score = BASE_SCORE

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
        
        return {
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
    except Exception as e:
        return None

# ==========================================
# 🚀 2. 开启多线程引擎，狂暴并发获取
# ==========================================
results = []
MAX_WORKERS = 10 

print(f"⚡ 正在分配火力，开启 {MAX_WORKERS} 个并发线程执行...")

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    future_to_fund = {executor.submit(process_single_fund, code): code for code in fund_list}
    for future in as_completed(future_to_fund):
        fund_code = future_to_fund[future]
        try:
            data = future.result()
            if data:
                results.append(data)
                print(f"✅ 完成运算: {fund_code} ({data['名称']}) | 得分: {data['综合得分']}")
            else:
                print(f"❌ 获取失败或跳过: {fund_code}")
        except Exception as exc:
            print(f"❌ 基金 {fund_code} 发生致命错误: {exc}")

# ==========================================
# 3. 结果汇总与绝对格式化
# ==========================================
summary_text = ""
if len(results) > 0:
    df_output = pd.DataFrame(results)
    df_output.sort_values(by="综合得分", ascending=False, inplace=True)
    
    top3 = df_output.head(3)
    summary_text += "🏆 【今日得分 Top 3 基金】\n"
    summary_text += "-" * 30 + "\n"
    for idx, row in top3.iterrows():
        summary_text += f"▪️ {row['名称']} ({row['代码']})\n"
        summary_text += f"   得分: {row['综合得分']} | 评级: {row['操作评级']}\n"
        # 考虑到如果某个基金还没满1年，百分位可能是空值，加入容错处理
        pct_1y_str = f"{row['1年百分位']*100:.1f}%" if pd.notnull(row['1年百分位']) else "无数据"
        max_dd_str = f"{row['最大回撤']*100:.1f}%" if pd.notnull(row['最大回撤']) else "无数据"
        summary_text += f"   1年百分位: {pct_1y_str} | 回撤: {max_dd_str}\n"
        summary_text += "-" * 30 + "\n"
    
    # 💡 强力格式化模块：将所有需要百分比显示的列，在 Pandas 层面彻底写死为带 % 的字符串
    def format_to_pct(val):
        if pd.isna(val) or val is None:
            return "-"
        return f"{val * 100:.2f}%"

    pct_cols = [
        "日波动(涨跌)", "最佳网格步长", "1年百分位", "3年百分位", 
        "乖离率(20日)", "年化波动率", "最大回撤", "今年以来", 
        "近1月", "近3月", "近1年", "近3年"
    ]
    
    for col in pct_cols:
        if col in df_output.columns:
            df_output[col] = df_output[col].apply(format_to_pct)
    
    # 导出到 Excel
    excel_filename = "全量基金智能打分表.xlsx"
    df_output.to_excel(excel_filename, index=False, sheet_name='量化复盘')
                        
    print(f"\n🎉 完美收官！全息动态加权矩阵已生成！")
else:
    print("\n⚠️ 抓取失败。")

# ==========================================
# 4. 邮件自动推送模块
# ==========================================
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
