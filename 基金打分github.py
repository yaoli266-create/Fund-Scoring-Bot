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

print("🚀 启动全维量化大脑 (ETF专属网格版 + 极简报表 + 自动算价)...")

# ==========================================
# 🔍 0. 构建全市场 ETF 名称映射字典
# ==========================================
print("正在拉取全市场 ETF 名称字典...")
try:
    spot_df = ak.fund_etf_spot_em()
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
    print(f"✅ 成功读取 {len(fund_list)} 只目标ETF！准备开启多线程并发拉取...")
except:
    print("⚠️ 未找到 '我的基金池.xlsx'，进入测试模式...")
    fund_list = ['510300', '159915', '512890', '512170'] 

current_year = datetime.now().year
RISK_FREE_RATE = 0.02

# ==========================================
# ⚡️ 核心算力函数：场内ETF专属打分逻辑
# ==========================================
def process_single_fund(code):
    code = str(code).zfill(6)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 3)
    
    try:
        df_k = ak.fund_etf_hist_em(
            symbol=code, period="daily", 
            start_date=start_date.strftime("%Y%m%d"), 
            end_date=end_date.strftime("%Y%m%d"), adjust="qfq"
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

        # 网格步长：依旧采用波动率的1/10，上下限锁定为 1.5% 到 5.0%
        grid_step = max(0.015, min(0.05, vol_1y / 10.0)) if vol_1y else 0.02
        
        # 💡 新增：实战买卖挂单价计算
        buy_trigger = round(cur_price_display * (1 - grid_step), 3)
        sell_trigger = round(cur_price_display * (1 + grid_step), 3)

        fund_name = name_dict.get(code, code) 
        
        # ---------------------------------------------------------
        # 🧠 场内 ETF 专属网格打分模型 (Mean Reversion & Grid Optim)
        # ---------------------------------------------------------
        BASE_SCORE = 50
        final_score = BASE_SCORE

        # 1. 均值回归因子 (ETF网格核心：重仓超跌，规避超买)
        if bias_val is not None:
            if bias_val < -0.08: final_score += 35      # 极度超跌，绝佳捡筹码区间
            elif bias_val < -0.05: final_score += 20    # 轻度超跌
            elif bias_val > 0.08: final_score -= 35     # 极度超买，马上要回调
            elif bias_val > 0.05: final_score -= 20     # 轻度超买

        # 2. 长期水位因子 (决胜于买入成本)
        if pct_1y is not None:
            if pct_1y < 0.10: final_score += 30         # 跌至1年内冰点
            elif pct_1y < 0.30: final_score += 15       # 底部区域
            elif pct_1y > 0.90: final_score -= 40       # 历史天际线，接盘预警
            elif pct_1y > 0.75: final_score -= 20       # 高位区域
            
        # 3. 趋势动能辅助 (左侧接刀也需要确认资金介入)
        if macd_hist_trend == 'improving': final_score += 5
        elif macd_hist_trend == 'deteriorating': final_score -= 5

        # 4. 盈亏比验证 (夏普/索提诺)
        if sharpe_1y is not None:
            final_score += max(-10, min(10, (sharpe_1y - 0.5) * 8))
        if sortino_1y is not None:
            final_score += max(-10, min(15, (sortino_1y - 0.8) * 8))

        # 评级文本微调以贴合网格交易
        if final_score >= 100: signal_text = "☄️ 绝对冰点-加大网格买入" 
        elif final_score >= 80: signal_text = "🔥 极度低估-启动网格建仓"
        elif final_score >= 65: signal_text = "💰 优质低估-逢低接底仓"
        elif final_score <= 10: signal_text = "☢️ 泡沫破灭-清空所有网格"
        elif final_score <= 30: signal_text = "⚠️ 极度危险-缩小网格卖出"
        elif final_score <= 45: signal_text = "📉 高位风险-暂停买入只卖"
        else: signal_text = "⏳ 估值合理-保持网格运行"
        
        def get_return(trading_days):
            if len(df_k) > trading_days:
                base_price = df_k.iloc[-1 - trading_days]['close']
                return (cur_price_adj - base_price) / base_price
            return None
        
        last_year_df = df_k[df_k['date'].dt.year < current_year]
        ytd_ret = (cur_price_adj - last_year_df.iloc[-1]['close']) / last_year_df.iloc[-1]['close'] if not last_year_df.empty else None
        
        # 返回装载好的单行数据字典 (💡 删除了资产类型、年化波动率，新增买卖挂单价)
        return {
            "代码": code,
            "名称": fund_name,
            "综合得分": round(final_score, 1),
            "操作评级": signal_text,
            "最新净值": round(cur_price_display, 4),
            "买入挂单价": buy_trigger,
            "卖出挂单价": sell_trigger,
            "最佳网格步长": grid_step,
            "日波动(涨跌)": daily_pct,
            "1年百分位": pct_1y,
            "3年百分位": pct_3y,
            "乖离率(20日)": bias_val,
            "夏普比率(1年)": round(sharpe_1y, 2) if pd.notnull(sharpe_1y) else None,
            "索提诺比率(1年)": round(sortino_1y, 2) if pd.notnull(sortino_1y) else None,
            "最大回撤": max_dd_1y,
            "今年以来": ytd_ret,
            "近1月": get_return(20),
            "近3月": get_return(60),
            "近1年": get_return(250),
            "近3年": get_return(750)
        }
    except Exception as e:
        return None

# ==========================================
# 🚀 2. 开启多线程引擎
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
    summary_text += "🏆 【今日网格绝佳买入标的 Top 3】\n"
    summary_text += "-" * 30 + "\n"
    for idx, row in top3.iterrows():
        summary_text += f"▪️ {row['名称']} ({row['代码']})\n"
        summary_text += f"   得分: {row['综合得分']} | 建议: {row['操作评级']}\n"
        # 直接把挂单价打印在邮件摘要里
        summary_text += f"   🎯 下跌买入价: {row['买入挂单价']} | 上涨卖出价: {row['卖出挂单价']}\n"
        summary_text += "-" * 30 + "\n"
    
    def format_to_pct(val):
        if pd.isna(val) or val is None:
            return "-"
        return f"{val * 100:.2f}%"

    # 更新了需格式化为百分比的列名
    pct_cols = [
        "日波动(涨跌)", "最佳网格步长", "1年百分位", "3年百分位", 
        "乖离率(20日)", "最大回撤", "今年以来", 
        "近1月", "近3月", "近1年", "近3年"
    ]
    
    for col in pct_cols:
        if col in df_output.columns:
            df_output[col] = df_output[col].apply(format_to_pct)
    
    excel_filename = "全量基金智能打分表.xlsx"
    df_output.to_excel(excel_filename, index=False, sheet_name='网格复盘')
                        
    print(f"\n🎉 完美收官！ETF专属网格参数矩阵已生成！")
else:
    print("\n⚠️ 抓取失败。")

# ==========================================
# 4. 邮件自动推送模块 (完全复用)
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
    msg['Subject'] = f"📊 量化大脑：ETF网格智能参数与实战复盘 ({datetime.now().strftime('%Y-%m-%d')})"
    
    body = (
        f"主人您好，今日的《ETF网格交易参数矩阵》演算完毕，请查收附件并参考挂单。\n\n"
        f"{email_body_summary}\n"
        f"—— 自动量化机器人 敬上\n"
    )
    msg.attach(MIMEText(body, 'plain', 'utf-8'))
    
    try:
        with open(file_path, "rb") as f:
            part = MIMEApplication(f.read(), Name=os.path.basename(file_path))
        part['Content-Disposition'] = f'attachment; filename="{os.path.basename(file_path)}"'
        msg.attach(part)
        server = smtplib.SMTP_SSL(smtp_server, smtp_port)
        server.login(sender, password)
        server.send_message(msg)
        server.quit()
        print("✅ 邮件发送成功！请检查您的收件箱。")
    except Exception as e:
        print(f"❌ 邮件发送失败: {e}")

if len(results) > 0 and os.path.exists(excel_filename):
    send_excel_via_email(excel_filename, summary_text)
