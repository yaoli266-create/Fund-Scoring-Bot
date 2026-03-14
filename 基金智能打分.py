import requests
import pandas as pd
import numpy as np
import re
import json
import time
import random
import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

warnings.filterwarnings('ignore')

print("🚀 启动全维量化大脑 v11.0 (双擎 Z-Score 终极进化版)...")

# =========================
# 全局参数与系统配置
# =========================
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
MAX_WORKERS = 10  
RISK_FREE = 0.02
MIN_TRADING_DAYS = 252 

# 宏观基准与行业风向标 ETF 配置
BENCHMARK_CODE = "510300"  
SECTOR_ETFS = {
    "半导体": "512760",
    "新能源": "516160",
    "医药": "512010",
    "消费": "159928",
    "军工": "512660",
    "金融": "512880",
    "科技": "515000",
    "红利": "510880"
}

SECTOR_KEYWORDS = {
    "半导体": ["芯片", "半导体"],
    "新能源": ["新能源", "光伏", "锂电"],
    "医药": ["医药", "医疗"],
    "消费": ["消费", "白酒", "食品"],
    "军工": ["军工"],
    "金融": ["银行", "证券", "非银"],
    "科技": ["科技", "互联网", "通信"],
    "红利": ["红利", "高股息", "价值"]
}

def get_sector_by_name(name):
    for sector, keys in SECTOR_KEYWORDS.items():
        if any(k in name for k in keys):
            return sector
    return "宽基/其他"

# =========================
# 核心数据拉取与清洗 (植入双维自适应基因)
# =========================
def fetch_and_clean_data(code, retries=3):
    code = str(code).zfill(6)
    url = f"http://fund.eastmoney.com/pingzhongdata/{code}.js"
    
    for attempt in range(retries):
        try:
            time.sleep(random.uniform(0.5, 1.5))
            
            r = requests.get(url, headers=HEADERS, timeout=10)
            r.raise_for_status()
            text = r.text

            name_match = re.search(r'var fS_name = "(.*?)";', text)
            if not name_match: return None
            name = name_match.group(1)
            if any(x in name for x in ["货币", "纯债", "理财"]): return None

            trend_match = re.search(r'var Data_netWorthTrend = (\[.*?\]);', text)
            if not trend_match: return None
            data = json.loads(trend_match.group(1))
            
            df = pd.DataFrame(data)
            if len(df) < MIN_TRADING_DAYS: return None

            df["close_raw"] = df["y"].astype(float)
            df["ret"] = df["equityReturn"].fillna(0).astype(float) / 100.0
            
            df["cum_growth"] = (1 + df["ret"]).cumprod()
            factor = df.iloc[-1]["close_raw"] / df.iloc[-1]["cum_growth"] if df.iloc[-1]["cum_growth"] != 0 else 1
            df["close_adj"] = df["cum_growth"] * factor

            # 基础技术与乖离指标
            df["MA20"] = df["close_adj"].rolling(20).mean()
            df["MA60"] = df["close_adj"].rolling(60).mean()
            df["MA120"] = df["close_adj"].rolling(120).mean()
            df["MA200"] = df["close_adj"].rolling(200).mean()
            df["BIAS20"] = (df["close_adj"] - df["MA20"]) / df["MA20"]

            # 左侧引擎核心：乖离率 Z-Score (恐慌/贪婪度)
            df["ROLL_BIAS_MEAN"] = df["BIAS20"].rolling(252).mean()
            df["ROLL_BIAS_STD"] = df["BIAS20"].rolling(252).std()
            df["BIAS_ZSCORE"] = (df["BIAS20"] - df["ROLL_BIAS_MEAN"]) / df["ROLL_BIAS_STD"]

            # 右侧引擎核心：动量 Z-Score (相对强势度)
            df["MOM60"] = df["close_adj"] / df["close_adj"].shift(60) - 1
            df["ROLL_MOM_MEAN"] = df["MOM60"].rolling(252).mean()
            df["ROLL_MOM_STD"] = df["MOM60"].rolling(252).std()
            df["MOM_ZSCORE"] = (df["MOM60"] - df["ROLL_MOM_MEAN"]) / df["ROLL_MOM_STD"]

            return name, df
            
        except requests.exceptions.RequestException:
            time.sleep(2)
        except json.JSONDecodeError:
            time.sleep(2)
        except Exception:
            time.sleep(2) 
            continue
    return None

# =========================
# 宏观与中观前置探测引擎
# =========================
def analyze_macro_and_sectors():
    print("📡 正在扫描宏观大盘与中观行业状态...")
    
    market_state = "震荡"
    hs300_data = fetch_and_clean_data(BENCHMARK_CODE)
    if hs300_data:
        _, df = hs300_data
        latest = df.iloc[-1]
        if pd.notnull(latest["MA200"]):
            if latest["close_adj"] > latest["MA200"]:
                market_state = "牛市"
            else:
                market_state = "熊市"
    print(f"📊 宏观气候判定: 【{market_state}】环境")

    sector_momentums = {}
    for sector, code in SECTOR_ETFS.items():
        data = fetch_and_clean_data(code)
        if data:
            _, df = data
            latest = df.iloc[-1]
            if pd.notnull(latest["MOM_ZSCORE"]):
                # 使用相对动量进行赛道PK，更显公平
                sector_momentums[sector] = latest["MOM_ZSCORE"]
                
    top_sectors = sorted(sector_momentums, key=sector_momentums.get, reverse=True)[:3]
    print(f"🏆 当前标准动量最强 TOP 3 赛道: {top_sectors}\n")
    
    return market_state, top_sectors

# =========================
# 策略双引擎核算核心逻辑
# =========================
def process_fund(code, market_state, top_sectors):
    data = fetch_and_clean_data(code)
    if not data: return None
    
    name, df = data
    sector = get_sector_by_name(name)
    df_1y = df.tail(250)
    latest = df.iloc[-1]
    
    roll_max = df_1y["close_adj"].cummax()
    max_dd = ((df_1y["close_adj"] - roll_max) / roll_max).min()
    vol = df_1y["ret"].std() * np.sqrt(250)
    ret_ann = df_1y["ret"].mean() * 250
    sharpe = (ret_ann - RISK_FREE) / vol if vol > 0 else 0
    
    max_1y, min_1y = df_1y["close_adj"].max(), df_1y["close_adj"].min()
    pct1 = (latest["close_adj"] - min_1y) / (max_1y - min_1y) if max_1y > min_1y else 1.0
    
    bias_zscore = latest["BIAS_ZSCORE"]
    mom_zscore = latest["MOM_ZSCORE"]

    # =========================
    # 策略A：左侧抄底自适应引擎 (越跌越买)
    # =========================
    score_L = 50
    if pct1 < 0.1: score_L += 30
    elif pct1 < 0.2: score_L += 20
    elif pct1 > 0.8: score_L -= 20
    
    if pd.notnull(bias_zscore):
        if bias_zscore < -2.0: score_L += 25       # 极度恐慌
        elif bias_zscore < -1.5: score_L += 15     # 较度恐慌
        elif bias_zscore > 1.5: score_L -= 15      # 阶段超买
        
    if market_state == "牛市": score_L -= 10
    elif market_state == "熊市": score_L += 10

    sig_L = "☄️ 砸锅卖铁" if score_L >= 85 else "💰 分批建仓" if score_L >= 65 else "⏳ 观望"

    # =========================
    # 策略B：右侧趋势自适应引擎 (突破追击)
    # =========================
    score_R = 50
    if pd.notnull(latest["MA120"]):
        if latest["close_adj"] > latest["MA20"] > latest["MA60"] > latest["MA120"]:
            score_R += 20 
        elif latest["close_adj"] < latest["MA60"] < latest["MA120"]:
            score_R -= 20 
            
    if pd.notnull(mom_zscore):
        if mom_zscore > 1.5: score_R += 20     # 突破自身强势极值
        elif mom_zscore > 1.0: score_R += 10   # 走势转强
        elif mom_zscore < -1.0: score_R -= 15  # 动量溃散
        
    # 🛡️ 致命防守：若偏离均线过远(极度贪婪)，剥夺买入资格防接盘
    if pd.notnull(bias_zscore) and bias_zscore > 2.0:
        score_R -= 30

    score_R += max(-10, min(15, sharpe * 10))
    
    if sector in top_sectors:
        score_R += 25
        
    if market_state == "熊市": 
        score_R -= 30

    sig_R = "🔥 强势主升" if score_R >= 85 else "📈 顺势持有" if score_R >= 65 else "⏳ 观望"

    return {
        "代码": code, "名称": name, "所属赛道": sector,
        "左侧_反转得分": score_L, "左侧_操作建议": sig_L,
        "右侧_动量得分": score_R, "右侧_操作建议": sig_R,
        "1年百分位": pct1, "极度偏差(Z)": bias_zscore, 
        "相对动量(Z)": mom_zscore, "夏普比率": sharpe,
        "最大回撤": max_dd, "年化波动率": vol
    }

# =========================
# 自动化邮件推送模块
# =========================
def send_email_with_excel(file_path, df_left, df_right, market_state, top_sectors):
    print("\n📧 正在打包发送投研报表邮件...")
    
    sender = os.getenv('EMAIL_SENDER')
    pwd = os.getenv('EMAIL_PASSWORD')
    receiver = os.getenv('EMAIL_RECEIVER')
    
    if not all([sender, pwd, receiver]):
        print("⚠️ 提醒: 未检测到完整的邮件配置，跳过发送邮件，文件将通过 GitHub Artifacts 留存。")
        return

    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = receiver
    today_str = datetime.now().strftime('%Y-%m-%d')
    msg['Subject'] = f"量化大脑双擎 v11.0 - {today_str}"

    left_top10 = df_left.head(10)
    right_top10 = df_right.head(10)

    body = f"全维量化大脑运行完成，请查收今日【自适应双引擎】投研报表。\n\n"
    body += f"📊 宏观气候: 【{market_state}】\n"
    body += f"🏆 标准动量 Top 3 赛道: {', '.join(top_sectors)}\n\n"

    body += "====================================\n"
    body += "🟢 策略A：左侧网格 (恐慌抄底) TOP 10\n"
    body += "====================================\n"
    for i, (_, row) in enumerate(left_top10.iterrows(), 1):
        body += f"{i}. {row['代码']} {row['名称']} | 得分: {row['左侧_反转得分']} | 建议: {row['左侧_操作建议']} | 恐慌度: {row['极度偏差(Z)']} Sigma\n"

    body += "\n====================================\n"
    body += "🔴 策略B：右侧趋势 (动量突破) TOP 10\n"
    body += "====================================\n"
    for i, (_, row) in enumerate(right_top10.iterrows(), 1):
        body += f"{i}. {row['代码']} {row['名称']} | 得分: {row['右侧_动量得分']} | 建议: {row['右侧_操作建议']} | 突破度: {row['相对动量(Z)']} Sigma\n"

    msg.attach(MIMEText(body, 'plain', 'utf-8'))

    try:
        with open(file_path, "rb") as f:
            part = MIMEApplication(f.read(), Name=os.path.basename(file_path))
            part['Content-Disposition'] = f'attachment; filename="{os.path.basename(file_path)}"'
            msg.attach(part)
    except Exception as e:
        print(f"❌ 附件读取失败: {e}")
        return

    try:
        server = smtplib.SMTP_SSL("smtp.qq.com", 465) 
        server.login(sender, pwd)
        server.sendmail(sender, receiver, msg.as_string())
        server.quit()
        print("✅ 邮件发送成功！策略报表已触达。")
    except Exception as e:
        print(f"❌ 邮件发送失败: {e}")

# =========================
# 系统执行主进程
# =========================
if __name__ == "__main__":
    market_state, top_sectors = analyze_macro_and_sectors()
    
    try:
        df_input = pd.read_excel("我的基金池.xlsx", dtype={"基金代码": str})
        fund_list = df_input["基金代码"].dropna().astype(str).str.strip().str.zfill(6).unique().tolist()
        print(f"📥 成功挂载《我的基金池》，锁定 {len(fund_list)} 只标的进行多线程演算...")
    except:
        print("⚠️ 未找到本地 Excel，启动防御性测试池...")
        fund_list = ['510300', '159915', '512880', '512690', '110011', '005827', '159928', '512760']

    results = []
    with ThreadPoolExecutor(MAX_WORKERS) as executor:
        futures = {executor.submit(process_fund, code, market_state, top_sectors): code for code in fund_list}
        for i, f in enumerate(as_completed(futures), 1):
            res = f.result()
            if res: results.append(res)
            print(f"\r⏳ 矩阵推演进度: {i}/{len(fund_list)}", end="")

    if results:
        df_out = pd.DataFrame(results)
        pct_cols = ["1年百分位", "最大回撤", "年化波动率"]
        for col in pct_cols:
            df_out[col] = df_out[col].astype(float) * 100
            
        df_out = df_out.round({"左侧_反转得分": 1, "右侧_动量得分": 1, "夏普比率": 2, 
                               "1年百分位": 1, "极度偏差(Z)": 2, "相对动量(Z)": 2, 
                               "最大回撤": 1, "年化波动率": 1})
        
        df_left = df_out.sort_values("左侧_反转得分", ascending=False).drop(columns=["右侧_动量得分", "右侧_操作建议"])
        df_right = df_out.sort_values("右侧_动量得分", ascending=False).drop(columns=["左侧_反转得分", "左侧_操作建议"])
        
        output_filename = "量化大脑选基矩阵_v11.xlsx"
        with pd.ExcelWriter(output_filename) as writer:
            df_left.to_excel(writer, index=False, sheet_name=f"左侧网格_{market_state}")
            df_right.to_excel(writer, index=False, sheet_name=f"右侧趋势_{market_state}")
            
        print(f"\n✅ 演算完美收官！已生成底层物理文件《{output_filename}》。")
        
        send_email_with_excel(output_filename, df_left, df_right, market_state, top_sectors)
        
    else:
        print("\n\n❌ 未萃取到有效信号，请检查网络或接口状态。")
