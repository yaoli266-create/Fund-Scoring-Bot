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

print("🚀 启动全维量化大脑 (多线程并发版 + ETF专线 + 动态加权)...")

# 1. 读取基金池 (无缝兼容你的逻辑)
try:
    df_input = pd.read_excel('我的基金池.xlsx', dtype={'基金代码': str})
    fund_list = df_input['基金代码'].dropna().str.strip().tolist()
    print(f"✅ 成功读取 {len(fund_list)} 只场内基金！准备开启多线程并发拉取...")
except:
    print("⚠️ 未找到 '我的基金池.xlsx'，进入测试模式...")
    fund_list = ['510300', '159915', '512890', '512170'] # 替换为常见的场内ETF代码

current_year = datetime.now().year
RISK_FREE_RATE = 0.02

# ==========================================
# ⚡️ 核心算力函数：单独处理每一只基金
# ==========================================
def process_single_fund(code):
    """
    专门为单只场内基金（ETF）设计的计算核心
    使用多线程时，这个函数会被并发调用
    """
    code = str(code).zfill(6)
    
    # 获取过去3年(约750个交易日)的数据即可，减少网络带宽占用
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 3)
    
    try:
        # 【重大升级】: 使用 akshare 的场内基金(ETF)历史接口，直接获取前复权收盘价
        df_k = ak.fund_etf_hist_em(
            symbol=code, 
            period="daily", 
            start_date=start_date.strftime("%Y%m%d"), 
            end_date=end_date.strftime("%Y%m%d"), 
            adjust="qfq"
        )
        
        if df_k is None or df_k.empty or len(df_k) < 20:
            return None # 数据不足，跳过
            
        # 统一列名映射，无缝对接你原来的算法
        df_k['date'] = pd.to_datetime(df_k['日期'])
        df_k['close_raw'] = df_k['收盘'].astype(float)
        df_k['ret'] = df_k['涨跌幅'].astype(float) / 100.0 # ETF接口直接提供精准涨跌幅
        df_k['close'] = df_k['close_raw'] # 场内前复权价格即为真实调仓基准
        
        # --- 下面的算法部分完全保留你的智慧结晶 ---
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
        
        # 风控指标演算
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

        # 由于 akshare 场内接口不返回基金中文名，如果是纯代码字典，这里用代码做兜底
        # 实际运行中 akshare 获取 ETF 列表很快，也可以提前做个映射。这里简化处理。
        fund_name = code 
        
        # 针对场内 ETF 的简单分类 (可根据您的需求扩展)
        fund_type = '债券/货币ETF' if str(code).startswith(('511', '1590')) else '权益ETF'
        is_sector = True # 场内大部都是行业主题ETF，默认加上行业乘数增强波动容忍度
        
        BASE_SCORE = 50
        final_score = BASE_SCORE

        # 打分逻辑 (保留您的原版权益类逻辑)
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

        # 评级信号
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
        
        # 返回装载好的单行数据字典
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
        # 并发环境下，千万不要让单个基金的报错阻塞全局
        return None

# ==========================================
# 🚀 2. 开启多线程引擎，狂暴并发获取
# ==========================================
results = []
# 设定并发线程数。对于 200 只基金，设定 10-15 个线程是最优解（过高依然会被墙）
MAX_WORKERS = 10 

print(f"⚡ 正在分配火力，开启 {MAX_WORKERS} 个并发线程执行...")

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    # 提交所有任务到线程池
    future_to_fund = {executor.submit(process_single_fund, code): code for code in fund_list}
    
    # 只要有任何一个线程完成，就立即收集结果
    for future in as_completed(future_to_fund):
        fund_code = future_to_fund[future]
        try:
            data = future.result()
            if data:
                results.append(data)
                print(f"✅ 完成运算: {fund_code} | 得分: {data['综合得分']}")
            else:
                print(f"❌ 获取失败或跳过: {fund_code}")
        except Exception as exc:
            print(f"❌ 基金 {fund_code} 发生致命错误: {exc}")


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
