import requests
import pandas as pd
import numpy as np
import re
import json
import time
import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

warnings.filterwarnings('ignore')

print("🚀 启动全维量化大脑 v9.0 (云端自动化部署版)...")

# =========================
# 全局参数与系统配置
# =========================
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
MAX_WORKERS = 15  # 并发线程数，控制在15-20避免触发反爬策略
RISK_FREE = 0.02
MIN_TRADING_DAYS = 120 # 剔除成立不足半年的次新基

# 宏观基准与行业风向标 ETF 配置
BENCHMARK_CODE = "510300"  # 沪深300 ETF
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

# 基金名称文本映射 (用于给持仓基金归类)
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
# 核心数据拉取与清洗
# =========================
def fetch_and_clean_data(code, retries=3):
    """拉取数据并使用累计复权净值重构核心技术指标"""
    code = str(code).zfill(6)
    url = f"http://fund.eastmoney.com/pingzhongdata/{code}.js"
    
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=8)
            text = r.text

            name_match = re.search(r'var fS_name = "(.*?)";', text)
            if not name_match: return None
            name = name_match.group(1)
            # 过滤不需要的基金类型
            if any(x in name for x in ["货币", "纯债", "理财"]): return None

            trend_match = re.search(r'var Data_netWorthTrend = (\[.*?\]);', text)
            if not trend_match: return None
            data = json.loads(trend_match.group(1))
            
            df = pd.DataFrame(data)
            if len(df) < MIN_TRADING_DAYS: return None

            df["close_raw"] = df["y"].astype(float)
            df["ret"] = df["equityReturn"].fillna(0).astype(float) / 100.0
            
            # 🛡️ 核心修复：基于每日收益率连乘，计算真实的“累计复权净值”
            # 确保 BIAS 与均线演算不受基金现金分红或折算的影响
            df["cum_growth"] = (1 + df["ret"]).cumprod()
            factor = df.iloc[-1]["close_raw"] / df.iloc[-1]["cum_growth"] if df.iloc[-1]["cum_growth"] != 0 else 1
            df["close_adj"] = df["cum_growth"] * factor

            # 基于复权净值计算技术指标
            df["MA20"] = df["close_adj"].rolling(20).mean()
            df["MA60"] = df["close_adj"].rolling(60).mean()
            df["MA120"] = df["close_adj"].rolling(120).mean()
            df["MA200"] = df["close_adj"].rolling(200).mean()
            
            # 使用累计复权净值推演乖离率
            df["BIAS20"] = (df["close_adj"] - df["MA20"]) / df["MA20"]

            return name, df
        except Exception:
            time.sleep(1) # 遇错冷却重试
            continue
    return None

# =========================
# 宏观与中观前置探测引擎
# =========================
def analyze_macro_and_sectors():
    print("📡 正在扫描宏观大盘与中观行业状态...")
    
    # 1. 市场状态识别 (沪深300 MA200 判定牛熊)
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

    # 2. 行业轮动动量扫描 (测算 60 日动量)
    sector_momentums = {}
    for sector, code in SECTOR_ETFS.items():
        data = fetch_and_clean_data(code)
        if data:
            _, df = data
            if len(df) >= 60:
                mom60 = df.iloc[-1]["close_adj"] / df.iloc[-60]["close_adj"] - 1
                sector_momentums[sector] = mom60
                
    # 提取动量最强的 TOP 3 行业
    top_sectors = sorted(sector_momentums, key=sector_momentums.get, reverse=True)[:3]
    print(f"🏆 当前动量最强 TOP 3 赛道: {top_sectors}\n")
    
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
    
    # 基础风控指标推演
    roll_max = df_1y["close_adj"].cummax()
    max_dd = ((df_1y["close_adj"] - roll_max) / roll_max).min()
    vol = df_1y["ret"].std() * np.sqrt(250)
    ret_ann = df_1y["ret"].mean() * 250
    sharpe = (ret_ann - RISK_FREE) / vol if vol > 0 else 0
    mom60 = latest["close_adj"] / df.iloc[-60]["close_adj"] - 1 if len(df) >= 60 else 0
    
    # 1年期高低位推演
    max_1y, min_1y = df_1y["close_adj"].max(), df_1y["close_adj"].min()
    pct1 = (latest["close_adj"] - min_1y) / (max_1y - min_1y) if max_1y > min_1y else 1.0
    bias20 = latest["BIAS20"]

    # -------------------------
    # 策略A：左侧抄底网格引擎 (低估反转)
    # -------------------------
    score_L = 50
    if pct1 < 0.1: score_L += 30
    elif pct1 < 0.2: score_L += 20
    elif pct1 > 0.8: score_L -= 20
    
    if pd.notnull(bias20):
        if bias20 < -0.08: score_L += 25
        elif bias20 > 0.05: score_L -= 15
        
    # 宏观干预调节阀
    if market_state == "牛市": score_L -= 10
    elif market_state == "熊市": score_L += 10

    sig_L = "☄️ 砸锅卖铁" if score_L >= 85 else "💰 分批建仓" if score_L >= 65 else "⏳ 观望"

    # -------------------------
    # 策略B：右侧趋势轮动引擎 (动量突破)
    # -------------------------
    score_R = 50
    if pd.notnull(latest["MA120"]):
        if latest["close_adj"] > latest["MA20"] > latest["MA60"] > latest["MA120"]:
            score_R += 20 
        elif latest["close_adj"] < latest["MA60"] < latest["MA120"]:
            score_R -= 20 
            
    if mom60 > 0.15: score_R += 20
    score_R += max(-10, min(15, sharpe * 10))
    
    # 行业景气度干预 (命中 TOP3 强势主线加分)
    if sector in top_sectors:
        score_R += 25
        
    # 宏观干预调节阀
    if market_state == "熊市": 
        score_R -= 30

    sig_R = "🔥 强势主升" if score_R >= 85 else "📈 顺势持有" if score_R >= 65 else "⏳ 观望"

    return {
        "代码": code, "名称": name, "所属赛道": sector,
        "左侧_反转得分": score_L, "左侧_操作建议": sig_L,
        "右侧_动量得分": score_R, "右侧_操作建议": sig_R,
        "1年百分位": pct1, "20日乖离率": bias20, 
        "60日动量": mom60, "夏普比率": sharpe,
        "最大回撤": max_dd, "年化波动率": vol
    }

# =========================
# 自动化邮件推送模块 (已升级 Top10 摘要)
# =========================
def send_email_with_excel(file_path, df_left, df_right, market_state, top_sectors):
    print("\n📧 正在打包发送投研报表邮件 (包含 Top10 摘要)...")
    
    sender = os.getenv('EMAIL_SENDER')
    pwd = os.getenv('EMAIL_PASSWORD')
    receiver = os.getenv('EMAIL_RECEIVER')
    
    if not all([sender, pwd, receiver]):
        print("⚠️ 提醒: 未检测到完整的邮件配置，跳过发送。")
        return

    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = receiver
    today_str = datetime.now().strftime('%Y-%m-%d')
    msg['Subject'] = f"量化大脑矩阵 v9.0 - {today_str}"

    # 提取左右侧 Top 10 数据
    left_top10 = df_left.head(10)
    right_top10 = df_right.head(10)

    # 构建邮件正文
    body = f"自动运行完成，请查收今日的【双引擎】量化评分报表。\n\n"
    body += f"📊 宏观状态: 【{market_state}】\n"
    body += f"🏆 强势赛道 Top 3: {', '.join(top_sectors)}\n\n"

    body += "====================================\n"
    body += "🟢 策略A：左侧网格 (低估反转) TOP 10\n"
    body += "====================================\n"
    for i, (_, row) in enumerate(left_top10.iterrows(), 1):
        body += f"{i}. {row['代码']} {row['名称']} | 得分: {row['左侧_反转得分']} | 建议: {row['左侧_操作建议']} | 1年位置: {row['1年百分位']}%\n"

    body += "\n====================================\n"
    body += "🔴 策略B：右侧趋势 (动量轮动) TOP 10\n"
    body += "====================================\n"
    for i, (_, row) in enumerate(right_top10.iterrows(), 1):
        body += f"{i}. {row['代码']} {row['名称']} | 得分: {row['右侧_动量得分']} | 建议: {row['右侧_操作建议']} | 60日动量: {row['60日动量']}%\n"

    body += "\n\n* 详细的完整指标（夏普、回撤、乖离等）及其他基金评分，请查阅附件 Excel。\n"
    body += "* 技术指标及BIAS均已严格运用累计复权净值推演，剔除分红干扰。"

    msg.attach(MIMEText(body, 'plain', 'utf-8'))

    # 挂载 Excel 附件
    try:
        with open(file_path, "rb") as f:
            part = MIMEApplication(f.read(), Name=os.path.basename(file_path))
            part['Content-Disposition'] = f'attachment; filename="{os.path.basename(file_path)}"'
            msg.attach(part)
    except Exception as e:
        print(f"❌ 附件读取失败: {e}")
        return

    # 发送请求
    try:
        server = smtplib.SMTP_SSL("smtp.qq.com", 465) 
        server.login(sender, pwd)
        server.sendmail(sender, receiver, msg.as_string())
        server.quit()
        print("✅ 邮件发送成功！策略报表及摘要已触达。")
    except Exception as e:
        print(f"❌ 邮件发送失败: {e}")
# =========================
# 系统执行主进程
# =========================
if __name__ == "__main__":
    # 1. 宏观与中观前置雷达探测
    market_state, top_sectors = analyze_macro_and_sectors()
    
    # 2. 加载目标基金池
    try:
        df_input = pd.read_excel("我的基金池.xlsx", dtype={"基金代码": str})
        fund_list = df_input["基金代码"].dropna().astype(str).str.strip().str.zfill(6).unique().tolist()
        print(f"📥 成功挂载《我的基金池》，共锁定 {len(fund_list)} 只标的，准备多线程并发测试...")
    except:
        print("⚠️ 未找到本地 Excel 配置文件，启动默认防御性测试池...")
        fund_list = ['510300', '159915', '512880', '512690', '110011', '005827', '159928', '512760']

    # 3. 高并发多线程演算
    results = []
    with ThreadPoolExecutor(MAX_WORKERS) as executor:
        futures = {executor.submit(process_fund, code, market_state, top_sectors): code for code in fund_list}
        for i, f in enumerate(as_completed(futures), 1):
            res = f.result()
            if res: results.append(res)
            print(f"\r⏳ 矩阵推演进度: {i}/{len(fund_list)}", end="")

    # 4. 数据报表清洗与持久化输出
    if results:
        df_out = pd.DataFrame(results)
        pct_cols = ["1年百分位", "20日乖离率", "60日动量", "最大回撤", "年化波动率"]
        for col in pct_cols:
            df_out[col] = df_out[col].astype(float) * 100
            
        df_out = df_out.round({"左侧_反转得分": 1, "右侧_动量得分": 1, "夏普比率": 2, 
                               "1年百分位": 1, "20日乖离率": 2, "60日动量": 1, 
                               "最大回撤": 1, "年化波动率": 1})
        
        df_left = df_out.sort_values("左侧_反转得分", ascending=False).drop(columns=["右侧_动量得分", "右侧_操作建议"])
        df_right = df_out.sort_values("右侧_动量得分", ascending=False).drop(columns=["左侧_反转得分", "左侧_操作建议"])
        
        output_filename = "量化大脑选基矩阵_v9.xlsx"
        with pd.ExcelWriter(output_filename) as writer:
            df_left.to_excel(writer, index=False, sheet_name=f"左侧网格_{market_state}")
            df_right.to_excel(writer, index=False, sheet_name=f"右侧趋势_{market_state}")
            
        print(f"\n✅ 演算完美收官！底层文件已生成《{output_filename}》。")
        
        # 5. 触发云端邮件下发协议
        send_email_with_excel(output_filename)
        
    else:
        print("\n\n❌ 未能萃取到有效信号，请检查上游接口状态或网络配置。")


