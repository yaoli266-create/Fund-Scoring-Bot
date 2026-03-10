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

print("🚀 启动全维量化大脑 V3.0 (主升浪起爆点 + 涨停基因 + 平台突破)...")

# ==========================================
# 🛑 0. 顶层风控：大盘趋势嗅探 (决定今日仓位)
# ==========================================
def check_market_environment():
    print("正在嗅探系统性风险 (上证指数趋势)...")
    try:
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
        
        # 基础过滤：剔除 ST、退市、次新股、一字涨跌停
        spot_df = spot_df[~spot_df['名称'].str.contains('ST|退|C|N')]
        spot_df = spot_df[(spot_df['涨跌幅'] < 9.8) & (spot_df['涨跌幅'] > -9.8)]
        
        # 活跃度与市值过滤：只看换手率>=3% 且 流通市值<=500亿 的活跃中小盘
        spot_df = spot_df[spot_df['换手率'] >= 3.0]
        spot_df = spot_df[spot_df['流通市值'] <= 50000000000]
        
        # 按成交额降序锁定头部资金聚集地
        active_pool = spot_df.sort_values(by='成交额', ascending=False).head(top_n)
        
        stock_list = active_pool['代码'].tolist()
        name_dict = dict(zip(active_pool['代码'], active_pool['名称']))
        spot_info_dict = active_pool.set_index('代码').to_dict('index')
        
        print(f"✅ 成功锁定 {len(stock_list)} 只 [高换手+中小盘] 标的，开启游资级量价诊断...")
        return stock_list, name_dict, spot_info_dict
    except Exception as e:
        print(f"⚠️ 粗筛失败: {e}")
        fallback_list = ['600519', '000858']
        return fallback_list, {c: c for c in fallback_list}, {}

stock_list, name_dict, spot_info_dict = get_active_stock_pool(top_n=500)

# ==========================================
# ⚡️ 2. 核心算力：V3.0 游资主升浪打分引擎
# ==========================================
def process_single_stock(code):
    code = str(code).zfill(6)
    end_date = datetime.now()
    # 扩大数据获取范围至200天，确保MA60和20日极值的计算数据充足
    start_date = end_date - timedelta(days=200) 
    
    try:
        df_k = ak.stock_zh_a_hist(
            symbol=code, period="daily", 
            start_date=start_date.strftime("%Y%m%d"), 
            end_date=end_date.strftime("%Y%m%d"), adjust="qfq"
        )
        
        if df_k is None or df_k.empty or len(df_k) < 65: # 确保至少有60天以上数据
            return None 
            
        df_k['date'] = pd.to_datetime(df_k['日期'])
        df_k['close'] = df_k['收盘'].astype(float)
        df_k['open'] = df_k['开盘'].astype(float)
        df_k['high'] = df_k['最高'].astype(float)
        df_k['vol'] = df_k['成交量'].astype(float)
        df_k['pct_change'] = df_k['涨跌幅'].astype(float)
        
        # 均线系统 (引入MA60牛熊分界线)
        df_k['MA5'] = df_k['close'].rolling(window=5).mean()
        df_k['MA10'] = df_k['close'].rolling(window=10).mean()
        df_k['MA20'] = df_k['close'].rolling(window=20).mean()
        df_k['MA60'] = df_k['close'].rolling(window=60).mean() 
        
        # 量能系统 (切换为更稳定的MA20_vol基准)
        df_k['Vol_MA20'] = df_k['vol'].rolling(window=20).mean()
        
        # 乖离率与MACD
        df_k['BIAS20'] = (df_k['close'] - df_k['MA20']) / df_k['MA20']
        exp1 = df_k['close'].ewm(span=12, adjust=False).mean()
        exp2 = df_k['close'].ewm(span=26, adjust=False).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        df_k['MACD_hist'] = macd_line - signal_line

        # ⭐️ 新增因子 1：涨停基因 (近20日内是否有过涨幅 >= 9.5% 的阳线)
        df_k['is_limit_up'] = (df_k['pct_change'] >= 9.5).astype(int)
        df_k['recent_limit_up'] = df_k['is_limit_up'].rolling(window=20).sum()
        
        # ⭐️ 新增因子 2：20日平台突破 (昨日之前的20日最高价)
        df_k['rolling_high_20'] = df_k['high'].rolling(window=20).max().shift(1)

        today = df_k.iloc[-1]
        yesterday = df_k.iloc[-2]
        
        # ---------------------------------------------------------
        # 🧠 V3.0 动能打分体系 (满分优化，权重向主升浪倾斜)
        # ---------------------------------------------------------
        BASE_SCORE = 50
        final_score = BASE_SCORE

        # 🛑 顶层环境制裁
        if not IS_MARKET_GOOD: final_score -= 30

        # 📈 1. 趋势周期判断 (MA60护航)
        if today['close'] > today['MA20'] and today['MA20'] > today['MA60']:
            final_score += 15  # 长线多头主升浪形态
        elif today['close'] < today['MA60']:
            final_score -= 30  # 跌破60日线，坚决认定为反弹诱多，一票否决
            
        if today['close'] > today['MA5'] and today['MA5'] > today['MA10'] and today['MA10'] > today['MA20']:
            final_score += 15  # 短线极度强势

        # 🧬 2. 涨停基因识别 (股性活跃度)
        if today['recent_limit_up'] >= 1:
            final_score += 10  # 股性已被激活，容易连板
        else:
            final_score -= 5   # 死鱼股，拉升极其费力

        # 🚀 3. 趋势突破识别 (创20日新高)
        if today['close'] > today['rolling_high_20']:
            final_score += 15  # 平台突破，极高权重进攻信号

        # 📊 4. 真实量能比对 (对比MA20)
        vol_ratio = today['vol'] / yesterday['Vol_MA20'] if yesterday['Vol_MA20'] > 0 else 1
        if 1.5 <= vol_ratio <= 4.0 and today['close'] > today['open']:
            final_score += 15  # 底部拔桩放量
        elif vol_ratio > 5.0:
            final_score -= 15  # 放量过大，提防爆量见顶
        elif vol_ratio < 0.6 and today['close'] < today['open']:
            final_score -= 10  # 缩量阴跌

        # 🎯 5. 其他辅助验证 (MACD与乖离防追高)
        if today['MACD_hist'] > yesterday['MACD_hist'] and today['MACD_hist'] > 0:
            final_score += 10  
            
        bias_val = today['BIAS20']
        if bias_val > 0.15:
            final_score -= 20  # 偏离20日线过高，回调风险剧增
        elif -0.02 <= bias_val <= 0.08:
            final_score += 10  # 均线附近起爆最佳买点

        # 提取实时现价和换手率
        spot_info = spot_info_dict.get(code, {})
        latest_price = spot_info.get('最新价', today['close'])
        pct_change = spot_info.get('涨跌幅', 0.0)
        industry = spot_info.get('所属行业', '未知')
        turnover = spot_info.get('换手率', 0.0)

        if final_score >= 100: signal_text = "🔥 龙头上车-绝佳的主升浪突破口"
        elif final_score >= 80: signal_text = "📈 强势共振-均线附近低吸待涨"
        elif final_score >= 60: signal_text = "⏳ 多空博弈-等待更强信号"
        else: signal_text = "☢️ 诱多陷阱-坚决规避不碰"
        
        return {
            "代码": code,
            "名称": name_dict.get(code, code),
            "所属板块": industry,
            "动能得分": round(final_score, 1),
            "行动策略": signal_text,
            "最新价": round(latest_price, 2),
            "今日涨幅": f"{pct_change}%",
            "换手率": f"{turnover}%",
            "今日量比": round(vol_ratio, 2), # 此时的量比已是对比 MA20 的真实量比
            "近20日涨停": int(today['recent_limit_up']),
            "20日乖离率": f"{bias_val*100:.2f}%"
        }
    except Exception as e:
        return None

# ==========================================
# 🚀 3. 多线程并发扫描
# ==========================================
results = []
MAX_WORKERS = 15 

print(f"⚡ 正在分配火力，开启 {MAX_WORKERS} 个并发线程执行 V3.0 动能量化...")

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    future_to_fund = {executor.submit(process_single_stock, code): code for code in stock_list}
    for future in as_completed(future_to_fund):
        try:
            data = future.result()
            # 门槛提高：V3.0要求得分至少 >= 70 才算及格（满分变高了，要求也更严厉）
            if data and data['动能得分'] >= 70: 
                results.append(data)
        except Exception:
            pass

# ==========================================
# 4. 结果汇总、板块共振统计与邮件推送
# ==========================================
summary_text = ""
excel_filename = "A股主升浪动能突破名单.xlsx"

# ⭐️ 构建大盘环境警报头
if IS_MARKET_GOOD:
    market_alert = f"🟢 【大盘环境安全】上证指数 ({SH_CLOSE:.2f}) 稳居20日线上方，情绪处于进攻期。\n"
else:
    market_alert = f"🚨 【系统性风险警告】上证指数 ({SH_CLOSE:.2f}) 跌破20日线！\n⚠️ 系统已强行开启「防御模式」并下调所有个股评分，极易吃面，务必空仓观望！\n"

summary_text += market_alert + "=" * 45 + "\n"

if len(results) > 0:
    df_output = pd.DataFrame(results)
    
    # 💡 解决问题三：板块强度。虽然不遍历全市场，但我们可以统计筛选出的及格股票中，哪个板块出现的频率最高！
    sector_counts = df_output['所属板块'].value_counts()
    hot_sectors = sector_counts[sector_counts >= 2].index.tolist()
    
    if hot_sectors:
        summary_text += f"🌪️ 【系统侦测到今日最强主线板块】：{', '.join(hot_sectors)}\n"
        summary_text += "(*实战建议：优先买入属于这些板块的标的，享受题材共振溢价*)\n"
        summary_text += "=" * 45 + "\n"
    
    # 优先按板块排序，其次按得分排序，方便直观看到板块效应
    df_output.sort_values(by=["所属板块", "动能得分"], ascending=[True, False], inplace=True)
    
    top10 = df_output.head(10)
    summary_text += "🎯 【V3.0 游资起爆点猎杀名单 Top 10】\n"
    summary_text += "(*特征：MA60护航 + 携带涨停基因 + 放量突破20日平台*)\n"
    summary_text += "-" * 45 + "\n"
    for idx, row in top10.iterrows():
        # 凸显带有涨停基因的股票
        gene_str = "🔥有涨停基因" if row['近20日涨停'] > 0 else "无涨停基因"
        summary_text += f"▪️ {row['名称']} ({row['代码']}) - 【{row['所属板块']}】\n"
        summary_text += f"   得分: {row['动能得分']} | {gene_str} | 状态: {row['行动策略']}\n"
        summary_text += f"   真实量比: {row['今日量比']}倍 | 偏离度: {row['20日乖离率']} | 换手率: {row['换手率']}\n"
        summary_text += "-" * 45 + "\n"
    
    df_output.to_excel(excel_filename, index=False, sheet_name='主升浪猎手')
    print(f"\n🎉 个股动能矩阵演算完毕！共捕获 {len(df_output)} 只起爆标的。")
else:
    print("\n⚠️ 今日无符合强势动能特征的标的。")
    summary_text += "📉 今日打分系统【交白卷】！没有一只股票能通过 V3.0 的严苛审核，请严格管住手，持币观望。\n"

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
    
    if IS_MARKET_GOOD:
        msg['Subject'] = f"🚀 游资雷达：A股主升浪突破阵型 ({datetime.now().strftime('%Y-%m-%d')})"
    else:
        msg['Subject'] = f"🚨 警报：大盘破位！量化大脑防御报告 ({datetime.now().strftime('%Y-%m-%d')})"
    
    body = f"主人您好，今日的《V3.0 主升浪起爆点猎杀名单》已生成。\n\n{email_body_summary}\n—— 自动量化机器人 敬上\n"
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
