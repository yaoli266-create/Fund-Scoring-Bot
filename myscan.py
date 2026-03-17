import akshare as ak
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import time
import os
import smtplib
import random
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.header import Header
import warnings

# 忽略警告信息
warnings.filterwarnings('ignore')
console = Console()

# ==========================================
# 1. 全局配置与请求会话优化
# ==========================================
RUN_MODE = os.getenv("RUN_MODE", "realtime")
BACKTEST_DATE = "2024-11-20"
HOLD_DAYS = 5
MAX_WORKERS = 15  # 降低并发数以提高 GitHub Actions 环境下的稳定性

# 配置全局 Session 增加连接重试能力
session = requests.Session()
retry_strategy = Retry(
    total=3,
    backoff_factor=1,
    status_forcelist=[429, 500, 502, 503, 504],
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session.mount("http://", adapter)
session.mount("https://", adapter)

class SequoiaUltimate:
    def __init__(self):
        self.min_volume = 150000000  # 1.5亿门槛
        self.market_breadth = 0

    # ------------------------------------------
    # 核心评分算法
    # ------------------------------------------
    def calculate_score(self, row, hist_df):
        try:
            score = 0
            close = hist_df['收盘'].values
            vol = hist_df['成交量'].values
            
            ma5 = pd.Series(close).rolling(5).mean().values[-1]
            ma10 = pd.Series(close).rolling(10).mean().values[-1]
            ma20 = pd.Series(close).rolling(20).mean().values[-1]
            
            if ma5 > ma10 > ma20:
                score += 30
                if close[-1] > ma5: score += 10

            avg_vol_5 = np.mean(vol[-6:-1])
            vol_ratio = vol[-1] / (avg_vol_5 + 1)
            if 1.5 <= vol_ratio <= 3.0:
                score += 25
            elif vol_ratio > 4.5:
                score -= 15

            turnover = row.get('换手率', 0)
            if 3 <= turnover <= 10:
                score += 20
            elif 1 <= turnover < 3 or 10 < turnover <= 15:
                score += 10

            bias_20 = (close[-1] - ma20) / ma20
            if 0 < bias_20 <= 0.05:
                score += 15
            elif 0.05 < bias_20 <= 0.12:
                score += 5

            return score
        except:
            return 0

    # ------------------------------------------
    # 健壮的数据抓取引擎 (带重试逻辑)
    # ------------------------------------------
    def fetch_with_retry(self, row, target_date=None):
        symbol = row['代码']
        # 针对 GitHub Actions 环境增加随机延迟，避免瞬间高并发
        time.sleep(random.uniform(0.2, 1.0))
        
        for attempt in range(3):  # 最多重试 3 次
            try:
                hist = ak.stock_zh_a_hist(symbol=symbol, period="daily", adjust="qfq")
                if hist.empty: return None
                
                hist['日期'] = pd.to_datetime(hist['日期'])
                
                if target_date:
                    target_dt = pd.to_datetime(target_date)
                    hist_snapshot = hist[hist['日期'] <= target_dt].tail(30)
                    future_data = hist[hist['日期'] > target_dt].head(HOLD_DAYS)
                    if len(future_data) < HOLD_DAYS: return None
                    profit = (future_data.iloc[-1]['收盘'] - hist_snapshot.iloc[-1]['收盘']) / hist_snapshot.iloc[-1]['收盘']
                else:
                    hist_snapshot = hist.tail(30)
                    profit = 0

                if len(hist_snapshot) < 20: return None
                score = self.calculate_score(row, hist_snapshot)
                
                if score >= 60:
                    return {
                        "代码": symbol, "名称": row['名称'], "最新价": row['最新价'],
                        "涨跌幅": row['涨跌幅'], "换手率": row.get('换手率', 0),
                        "综合评分": score, "回测收益": profit
                    }
                return None # 评分不足直接返回
            except Exception:
                if attempt < 2:
                    time.sleep(2 ** attempt) # 指数级退避等待
                    continue
                return None

    # ------------------------------------------
    # 邮件附件发送逻辑
    # ------------------------------------------
    def send_email_with_excel(self, df_results, excel_path):
        sender = os.getenv('EMAIL_USER')
        password = os.getenv('EMAIL_PASS')
        receiver = os.getenv('EMAIL_RECEIVER')
        
        if not all([sender, password, receiver]):
            console.print("[yellow]未检测到完整邮箱配置，跳过邮件发送。[/yellow]")
            return

        date_str = datetime.now().strftime('%Y-%m-%d')
        msg = MIMEMultipart()
        msg['From'] = sender
        msg['To'] = receiver
        msg['Subject'] = Header(f"Sequoia-X 选股报告 - {date_str}", 'utf-8')

        body = f"🚀 Sequoia-X 今日选股报告 ({date_str})\n"
        body += f"当前全市场多头占比: {self.market_breadth:.1f}%\n"
        body += "详细名单见附件 Excel 文件。\n"
        msg.attach(MIMEText(body, 'plain', 'utf-8'))

        try:
            with open(excel_path, 'rb') as f:
                part = MIMEApplication(f.read())
                part.add_header('Content-Disposition', 'attachment', filename=os.path.basename(excel_path))
                msg.attach(part)
        except Exception as e:
            console.print(f"[red]附件添加失败: {e}[/red]")

        try:
            smtp_server = "smtp.qq.com" 
            with smtplib.SMTP_SSL(smtp_server, 465) as server:
                server.login(sender, password)
                server.sendmail(sender, [receiver], msg.as_string())
            console.print("[bold green]邮件报告及附件已成功发送！[/bold green]")
        except Exception as e:
            console.print(f"[bold red]邮件发送失败: {e}[/bold red]")

    # ------------------------------------------
    # 运行主逻辑
    # ------------------------------------------
    def run(self):
        console.print(f"[bold cyan]▶ Sequoia-X 极速提速版启动...[/bold cyan]")
        
        # 1. 获取全市场快照
        try:
            df_spot = ak.stock_zh_a_spot_em()
            df_spot = df_spot[~df_spot['名称'].str.contains("ST|退")]
            df_spot = df_spot[df_spot['成交额'] > self.min_volume]
        except Exception as e:
            console.print(f"[bold red]获取市场快照失败: {e}[/bold red]")
            return

        # 2. 计算市场宽度 (抽样)
        sample = df_spot.head(min(100, len(df_spot)))
        self.market_breadth = (sample['最新价'] > sample['最新价'].rolling(20).mean().fillna(0)).sum() / len(sample) * 100
        
        # 3. 并发评分
        results = []
        target = BACKTEST_DATE if RUN_MODE == "backtest" else None
        
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(self.fetch_with_retry, row, target) for _, row in df_spot.iterrows()]
            for future in as_completed(futures):
                res = future.result()
                if res: results.append(res)
        
        final_df = pd.DataFrame(results)
        if not final_df.empty:
            final_df = final_df.sort_values("综合评分", ascending=False)
            
            # 生成 Excel
            date_str = datetime.now().strftime('%Y%m%d')
            file_name = f"Sequoia_Report_{date_str}.xlsx"
            final_df.to_excel(file_name, index=False)
            
            # 打印表格
            table = Table(title="Sequoia-X 精选结果")
            table.add_column("评分", style="red")
            table.add_column("代码", style="cyan")
            table.add_column("名称")
            for _, row in final_df.head(15).iterrows():
                table.add_row(str(row['综合评分']), row['代码'], row['名称'])
            console.print(table)

            if RUN_MODE == "realtime":
                self.send_email_with_excel(final_df, file_name)
        else:
            console.print("[yellow]今日未筛选出符合条件的个股。[/yellow]")

if __name__ == "__main__":
    start_time = time.time()
    SequoiaUltimate().run()
    console.print(f"任务完成，总耗时: {time.time() - start_time:.2f}s")
