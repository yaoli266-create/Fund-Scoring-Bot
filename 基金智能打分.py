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

print("🚀 启动全维量化大脑 V4.0 (大类资产轮动 + 宏观风控 + Smart Money 资金流)...")

# ==========================================
# 🛑 0. 顶层风控：沪深300 宏观风险嗅探 (决定网格密度)
# ==========================================
def check_macro_risk():
    """
    检查沪深300是否跌破60日均线 (牛熊分水岭)
    """
    print("正在嗅探宏观系统性风险 (沪深300趋势)...")
    try:
        df_index = ak.stock_zh_index_daily_em(symbol="sh000300")
        df_index['MA60'] = df_index['close'].rolling(window=60).mean()
        today_close = df_index.iloc[-1]['close']
        ma60 = df_index.iloc[-1]['MA60']
        
        is_bear_market = today_close < ma60
        status_text = "🔴 熊市预警 (跌破季线，防守模式)" if is_bear_market else "🟢 结构牛市 (站上季线，进攻模式)"
        print(f"沪深300状态: {status_text} | 最新点位: {today_close:.2f} | MA60: {ma60:.2f}")
        return is_bear_market, today_close, ma60
    except Exception as e:
        print(f"⚠️ 宏观数据获取失败，默认安全: {e}")
        return False, 0, 0

IS_BEAR_MARKET, HS300_CLOSE, HS300_MA60 = check_macro_risk()

# ==========================================
# 🧬 1. 资产分类器：打破“一刀切”的刻舟求剑
# ==========================================
def classify_etf(name):
    """根据ETF名称划定底层大类资产属性"""
    if any(k in name for k in ['纳斯达克', '标普', '日经', '恒生', '港股', '亚太', '德国', '法国', '道琼斯', '海外']):
        return '海外跨境'
    elif any(k in name for k in ['黄金', '白银', '豆粕', '有色', '能源', '大宗']):
        return '大宗商品'
    elif any(k in name for k in ['300', '500', '1000', '2000', '创业板', '科创', '上证50', '红利', 'A50', '深证']):
        return '宽基指数'
    else:
        return '行业主题'

# 获取 ETF 名称映射字典
print("正在拉取全市场 ETF 名称字典...")
try:
    spot_df = ak.fund_etf_spot_em()
    name_dict = dict(zip(spot_df['代码'], spot_df['名称']))
except Exception:
    name_dict = {}

try:
    df_input = pd.read_excel('我的基金池.xlsx', dtype={'基金代码': str})
    fund_list = df_input['基金代码'].dropna().str.strip().tolist()
    print(f"✅ 成功读取 {len(fund_list)} 只目标ETF，准备启动并发演算...")
except:
    print("⚠️ 未找到 '我的基金池.xlsx'，进入测试模式...")
    fund_list = ['510300', '159915', '512890', '518880', '513100'] # 包含宽基、行业、商品、海外

current_year = datetime.now().year
RISK_FREE_RATE = 0.02

# ==========================================
# ⚡️ 2. 核心算力：多策略异构打分引擎
# ==========================================
def process_single_etf(code):
    code = str(code).zfill(6)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * 3)
    
    try:
        df_k = ak.fund_etf_hist_em(
            symbol=code, period="daily", 
            start_date=start_date.strftime("%Y%m%d"), 
            end_date=end_date.strftime("%Y%m%d"), adjust="qfq"
        )
        if df_k is None or df_k.empty or len(df_k) < 65: return None 
            
        df_k['date'] = pd.to_datetime(df_k['日期'])
        df_k['close'] = df_k['收盘'].astype(float)
        df_k['vol'] = df_k['成交量'].astype(float)
        df_k['ret'] = df_k['涨跌幅'].astype(float) / 100.0 
        
        # 指标计算
        df_k['MA20'] = df_k['close'].rolling(window=20).mean()
        df_k['MA60'] = df_k['close'].rolling(window=60).mean()
        df_k['Vol_MA20'] = df_k['vol'].rolling(window=20).mean()
        df_k['BIAS20'] = (df_k['close'] - df_k['MA20']) / df_k['MA20']
        
        today = df_k.iloc[-1]
        yesterday = df_k.iloc[-2]
        
        df_1y = df_k.tail(250)
        max_1y, min_1y = df_1y['close'].max(), df_1y['close'].min()
        pct_1y = (today['close'] - min_1y) / (max_1y - min_1y) if max_1y > min_1y else None
        vol_1y = df_1y['ret'].std() * np.sqrt(250) if len(df_1y) > 10 else None

        fund_name = name_dict.get(code, code) 
        asset_class = classify_etf(fund_name)
        
        # 基础网格步长 (波动率的 1/10)
        grid_step = max(0.015, min(0.06, vol_1y / 10.0)) if vol_1y else 0.025
        
        # ---------------------------------------------------------
        # 🧠 V4.0 多维度异构打分体系 
        # ---------------------------------------------------------
        final_score = 50
        
        # 💡 升级一：资金流 (Smart Money) 监测
        # 机构左侧建仓特征：价格下跌或处于低位，但成交量极其异常地放大
        is_smart_money_inflow = (today['ret'] < 0.01) and (today['vol'] > today['Vol_MA20'] * 1.8)
        if is_smart_money_inflow:
            final_score += 25  # 极其强烈的底部主力托底信号
            
        # 💡 升级二：宏观系统性风险过滤 (沪深300联动)
        if IS_BEAR_MARKET:
            if asset_class in ['宽基指数', '行业主题']:
                final_score -= 20      # 压制A股相关资产的买入冲动
                grid_step *= 1.5       # 【核心】拉大网格间距，例如原来跌3%买，现在强行跌4.5%才买，防止资金打光
            elif asset_class == '大宗商品':
                final_score += 15      # 黄金等商品在A股熊市具有避险对冲价值，逆势加分
        else:
            if asset_class in ['宽基指数', '行业主题']:
                final_score += 10      # 大盘安全时，鼓励参与A股网格
        
        # 💡 升级三：大类资产异构打分
        bias_val = today['BIAS20']
        
        if asset_class == '宽基指数':
            # 宽基看重均值回归，跌出坑即可买
            if bias_val < -0.05: final_score += 20
            elif bias_val > 0.05: final_score -= 20
            
        elif asset_class == '行业主题':
            # 行业波动剧烈，必须深跌（戴维斯双杀）才能重仓
            if bias_val > -0.04: final_score -= 15  # 没跌透坚决不买
            elif bias_val < -0.08: final_score += 30 # 跌透了大力出奇迹
            if pct_1y is not None and pct_1y > 0.8: final_score -= 30 # 行业绝对不接高位盘
            
        elif asset_class == '海外跨境':
            # 海外（如纳指）是长牛趋势属性，不能等深跌
            if today['close'] > today['MA60']: final_score += 15 # 趋势向上顺势加分
            if bias_val < -0.03: final_score += 15 # 稍微回踩就是绝佳买点
            
        elif asset_class == '大宗商品':
            # 商品独立行情，参考均线支撑
            if -0.03 <= bias_val <= 0.02: final_score += 10
        
        # 价格挂单测算
        buy_trigger = round(today['close'] * (1 - grid_step), 3)
        sell_trigger = round(today['close'] * (1 + grid_step), 3)

        if final_score >= 90: signal_text = "🔥 绝对冰点-加大网格买入" 
        elif final_score >= 70: signal_text = "💰 优质低估-启动网格建仓"
        elif final_score >= 50: signal_text = "⏳ 估值合理-保持网格底仓"
        elif final_score <= 30: signal_text = "⚠️ 极度危险-缩小网格卖出"
        else: signal_text = "☢️ 高位泡沫-清空所有网格"
        
        return {
            "代码": code,
            "名称": fund_name,
            "资产类别": asset_class,
            "综合得分": round(final_score, 1),
            "操作建议": signal_text,
            "最新净值": round(today['close'], 4),
            "动态网格步长": grid_step,
            "建议买入价": buy_trigger,
            "建议卖出价": sell_trigger,
            "资金逆势流入": "✅ 主力潜伏" if is_smart_money_inflow else "-",
            "20日乖离率": bias_val,
            "1年百分位": pct_1y
        }
    except Exception as e:
        return None

# ==========================================
# 🚀 3. 多线程并发扫描
# ==========================================
results = []
MAX_WORKERS = 10 

print(f"⚡ 正在分配火力，开启 {MAX_WORKERS} 个并发线程执行大类资产演算...")

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    future_to_fund = {executor.submit(process_single_etf, code): code for code in fund_list}
    for future in as_completed(future_to_fund):
        try:
            data = future.result()
            if data: results.append(data)
        except Exception:
            pass

# ==========================================
# 4. 结果汇总与邮件推送
# ==========================================
summary_text = ""
excel_filename = "ETF全天候网格战略报表.xlsx"

# ⭐️ 宏观预警头部
if IS_BEAR_MARKET:
    summary_text += f"🚨 【宏观风控警报】沪深300 ({HS300_CLOSE:.2f}) 跌破季线！\n"
    summary_text += "🛡️ 系统已自动压制A股类ETF评分，并将网格步长拉大1.5倍以扩大防守纵深。建议重点关注黄金/海外对冲！\n"
else:
    summary_text += f"🟢 【宏观环境安全】沪深300 ({HS300_CLOSE:.2f}) 稳居季线上方，处于可进攻周期。\n"

summary_text += "=" * 45 + "\n"

if len(results) > 0:
    df_output = pd.DataFrame(results)
    
    # 格式化百分比
    def format_to_pct(val):
        if pd.isna(val) or val is None: return "-"
        return f"{val * 100:.2f}%"
    
    for col in ["动态网格步长", "20日乖离率", "1年百分位"]:
        df_output[col] = df_output[col].apply(format_to_pct)
        
    df_output.sort_values(by=["综合得分", "资产类别"], ascending=[False, True], inplace=True)
    
    top10 = df_output.head(10)
    summary_text += "🎯 【V4.0 大类资产网格优选 Top 10】\n"
    summary_text += "-" * 45 + "\n"
    for idx, row in top10.iterrows():
        # 若检测到异常放量的资金流入，重点标红
        money_flow_flag = " 💸[主力左侧流入!]" if row['资金逆势流入'] != "-" else ""
        
        summary_text += f"▪️ {row['名称']} ({row['代码']}) - 【{row['资产类别']}】{money_flow_flag}\n"
        summary_text += f"   得分: {row['综合得分']} | {row['操作建议']}\n"
        summary_text += f"   🎯 步长: {row['动态网格步长']} | 下跌买入价: {row['建议买入价']} | 上涨卖出价: {row['建议卖出价']}\n"
        summary_text += "-" * 45 + "\n"
    
    df_output.to_excel(excel_filename, index=False, sheet_name='大类资产网格')
    print(f"\n🎉 V4.0 全息资产矩阵演算完毕！")
else:
    print("\n⚠️ 抓取失败。")

def send_excel_via_email(file_path, email_body_summary):
    sender = os.getenv("EMAIL_SENDER")
    password = os.getenv("EMAIL_PASSWORD") 
    receiver = os.getenv("EMAIL_RECEIVER")
    smtp_server = os.getenv("SMTP_SERVER", "smtp.qq.com")
    smtp_port = int(os.getenv("SMTP_PORT", 465))
    
    if not all([sender, password, receiver]):
        return

    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = receiver
    
    if IS_BEAR_MARKET:
        msg['Subject'] = f"🚨 宏观防守：ETF网格战略风控报表 ({datetime.now().strftime('%Y-%m-%d')})"
    else:
        msg['Subject'] = f"🚀 宏观进攻：ETF全天候资产轮动雷达 ({datetime.now().strftime('%Y-%m-%d')})"
    
    body = f"主人您好，今日的《V4.0 ETF全天候大类资产轮动与网格报表》已生成。\n\n{email_body_summary}\n—— 自动量化大脑 敬上\n"
    msg.attach(MIMEText(body, 'plain', 'utf-8'))
    
    if os.path.exists(file_path):
        try:
            with open(file_path, "rb") as f:
                part = MIMEApplication(f.read(), Name=os.path.basename(file_path))
            part['Content-Disposition'] = f'attachment; filename="{os.path.basename(file_path)}"'
            msg.attach(part)
        except Exception: pass
        
    try:
        server = smtplib.SMTP_SSL(smtp_server, smtp_port)
        server.login(sender, password)
        server.send_message(msg)
        server.quit()
        print("✅ ETF战略邮件发送成功！")
    except Exception as e:
        print(f"❌ 邮件发送失败: {e}")

send_excel_via_email(excel_filename, summary_text)
