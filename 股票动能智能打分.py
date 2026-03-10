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

print("🚀 启动全维量化大脑 (A股实战终极版 - 大盘熔断 + 流动性甄别 + 右侧动能)...")

# ==========================================
# 🛑 0. 顶层风控：大盘趋势嗅探 (决定今日仓位)
# ==========================================
def check_market_environment():
    """
    检查上证指数是否站上20日均线。
    量化铁律：大盘破位，所有突破皆为诱多。
    """
    print("正在嗅探系统性风险 (上证指数趋势)...")
    try:
        # 获取上证指数历史日线
        df_index = ak.stock_zh_index_daily_em(symbol="sh000001")
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
# 🔍 1. 实时粗筛：截取最符合游资审美的海选池
# ==========================================
def get_active_stock_pool(top_n=500):
    print(f"正在扫描全市场，执行基本面与流动性双重过滤...")
    try:
        spot_df = ak.stock_zh_a_spot_em()
        
        # 1. 基础过滤：剔除 ST、退市、次新股
        spot_df = spot_df[~spot_df['名称'].str.contains('ST|退|C|N')]
        
        # 2. 形态过滤：剔除一字涨停和一字跌停
        spot_df = spot_df[(spot_df['涨跌幅'] < 9.8) & (spot_df['涨跌幅'] > -9.8)]
        
        # 3. ⭐️ 活跃度过滤：剔除换手率 < 3% 的死水股 (无短线资金关注)
        spot_df = spot_df[spot_df['换手率'] >= 3.0]
        
        # 4. ⭐️ 市值过滤：剔除流通市值 > 500亿 的大盘股 (短线拉不动)
        # 注意：akshare中市值单位通常为元，500亿 = 500 * 10^8
        spot_df = spot_df[spot_df['流通市值'] <= 50000000000]
        
        # 按“成交额”降序排列，截取头部高流动性标的
        active_pool = spot_df.sort_values(by='成交额', ascending=False).head(top_n)
        
        stock_list = active_pool['代码'].tolist()
        name_dict = dict(zip(active_pool['代码'], active_pool['名称']))
        spot_info_dict = active_pool.set_index('代码').to_dict('index')
        
        print(f"✅ 成功锁定 {len(stock_list)} 只 [高换手+中小盘] 标的，开始深度量价诊断...")
        return stock_list, name_dict, spot_info_dict
    except Exception as e:
        print(f"⚠️ 粗筛失败: {e}")
        fallback_list = ['600519', '000858', '603993']
        return fallback_list, {c: c for c in fallback_list}, {}

stock_list, name_dict, spot_info_dict = get_active_stock_pool(top_n=500)

# ==========================================
# ⚡️ 2. 核心算力：个股专属动能打分逻辑
# ==========================================
def process_single_stock(code):
    code = str(code).zfill(6)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=120)
    
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
        df_k['Vol_MA20'] = df_k['vol'].rolling(window=20).mean() # 新增20日均量
        df_k['BIAS20'] = (df_k['close'] - df_k['MA20']) / df_k['MA20']
        
        exp1 = df_k['close'].ewm(span=12, adjust=False).mean()
        exp2 = df_k['close'].ewm(span=26, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        df_k['MACD_hist'] = macd_line - signal_line

        today = df_k.iloc[-1]
        yesterday = df_k.iloc[-2]
        
        # ---------------------------------------------------------
        # 🧠 动能评分引擎 (融合大盘环境判定)
        # ---------------------------------------------------------
        BASE_SCORE = 50
        final_score = BASE_SCORE

        # ⭐️ 顶层环境制裁：如果大盘破位，所有个股强行扣除30分，几乎不可能及格
        if not IS_MARKET_GOOD:
            final_score -= 30

        # 1. 趋势动能因子
        if today['close'] > today['MA5'] and today['MA5'] > today['MA10'] and today['MA10'] > today['MA20']:
            final_score += 20  
        elif today['MA5'] < today['MA10']:
            final_score -= 30  

        # 2. MACD 加速因子
        if today['MACD_hist'] > yesterday['MACD_hist'] and today['MACD_hist'] > 0:
            final_score += 10  
        
        # 3. 量能突破因子
        vol_ratio = today['vol'] / yesterday['Vol_MA5'] if yesterday['Vol_MA5'] > 0 else 1
        if 1.5 <= vol_ratio <= 3.5 and today['close'] > today['open']:
            final_score += 15  
        elif vol_ratio > 4.0:
            final_score -= 15  # 爆量滞涨防范
        elif vol_ratio < 0.6 and today['close'] < today['open']:
            final_score -= 10  
            
        # ⭐️ 4. 量能连续性 (真金白银验证)
        if today['vol'] > today['Vol_MA20'] and yesterday['vol'] > yesterday['Vol_MA20']:
            final_score += 10  # 连续两日成交量大于20日均量，说明大资金进场持续性好

        # 5. 乖离率防追高因子
        bias_val = today['BIAS20']
        if bias_val > 0.15:
            final_score -= 20  # 短线涨幅过大，防止站岗
        elif -0.02 <= bias_val <= 0.08:
            final_score += 10  # 贴近均线起爆，最安全的上车点

        # 提取实时现价和换手率
        spot_info = spot_info_dict.get(code, {})
        latest_price = spot_info.get('最新价', today['close'])
        pct_change = spot_info.get('涨跌幅', 0.0)
        industry = spot_info.get('所属行业', '未知')
        turnover = spot_info.get('换手率', 0.0)

        if final_score >= 85: signal_text = "🔥 强势主升-竞价确认后上车"
        elif final_score >= 70: signal_text = "📈 趋势起爆-依托分时均线低吸"
        elif final_score >= 50: signal_text = "⏳ 震荡蓄势-观望"
        else: signal_text = "☢️ 破位/诱多-坚决拉黑"
        
        return {
            "代码": code,
            "名称": name_dict.get(code, code),
            "所属板块": industry,
            "动能得分": round(final_score, 1),
            "行动策略": signal_text,
            "最新价": round(latest_price, 2),
            "今日涨幅": f"{pct_change}%",
            "换手率": f"{turnover}%",
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
            # 只有得分及格的股票才有资格进入最终报表
            if data and data['动能得分'] >= 60: 
                results.append(data)
        except Exception:
            pass

# ==========================================
# 4. 结果汇总与邮件推送
# ==========================================
summary_text = ""
excel_filename = "A股高潜动能突破名单.xlsx"

# ⭐️ 构建大盘环境警报头
if IS_MARKET_GOOD:
    market_alert = f"🟢 【大盘环境安全】上证指数 ({SH_CLOSE:.2f}) 稳居20日线上方，可积极做多。\n"
else:
    market_alert = f"🚨 【系统性风险警告】上证指数 ({SH_CLOSE:.2f}) 已跌破20日线生命线！\n⚠️ 覆巢之下无完卵，量化系统已强制下调所有个股评分，建议空仓或极小仓位试错！\n"

summary_text += market_alert + "=" * 40 + "\n"

if len(results) > 0:
    df_output = pd.DataFrame(results)
    df_output.sort_values(by=["动能得分", "今日量比"], ascending=[False, False], inplace=True)
    
    top10 = df_output.head(10)
    summary_text += "🎯 【游资合力爆破 Top 10】(已剔除大盘股与死水股)\n"
    summary_text += "-" * 40 + "\n"
    for idx, row in top10.iterrows():
        summary_text += f"▪️ {row['名称']} ({row['代码']}) - {row['所属板块']}\n"
        summary_text += f"   得分: {row['动能得分']} | 状态: {row['行动策略']}\n"
        summary_text += f"   涨幅: {row['今日涨幅']} | 换手率: {row['换手率']} | 量比: {row['今日量比']}倍\n"
        summary_text += "-" * 40 + "\n"
    
    df_output.to_excel(excel_filename, index=False, sheet_name='动能猎手')
    print(f"\n🎉 个股动能矩阵演算完毕！共捕获 {len(df_output)} 只起爆标的。")
else:
    print("\n⚠️ 今日无符合强势动能特征的标的。")
    summary_text += "📉 今日打分系统【交白卷】！无任何标的满足起爆标准，请严格管住手，持币观望。\n"

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
    
    # 标题根据大盘状态动态变化
    if IS_MARKET_GOOD:
        msg['Subject'] = f"🚀 量化大脑：A股个股动能突破雷达 ({datetime.now().strftime('%Y-%m-%d')})"
    else:
        msg['Subject'] = f"🚨 警报：大盘破位！量化大脑复盘报告 ({datetime.now().strftime('%Y-%m-%d')})"
    
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
