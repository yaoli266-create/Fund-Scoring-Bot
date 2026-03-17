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
from email.mime.text import MIMEText
from email.header import Header
import warnings

# 忽略警告信息
warnings.filterwarnings('ignore')
console = Console()

# ==========================================
# 1. 全局配置
# ==========================================
RUN_MODE = os.getenv("RUN_MODE", "realtime")  # 默认实时模式
BACKTEST_DATE = "2024-11-20"                 # 回测日期
HOLD_DAYS = 5                                # 回测持股周期
MAX_WORKERS = 30                             # 并发线程数

class SequoiaUltimate:
    def __init__(self):
        self.min_volume = 150000000  # 成交额门槛 1.5亿
        self.market_breadth = 0      # 市场多头占比

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
            if 1.5 <= vol_ratio <= 3.0: score += 25
            elif vol_ratio > 4.5: score -= 15
            turnover = row.get('换手率', 0)
            if 3 <= turnover <= 10: score += 20
            elif 1 <= turnover < 3 or 10 < turnover <= 15: score += 10
            bias_20 = (close[-1] - ma20) / ma20
            if 0 < bias_20 <= 0.05: score += 15
            elif 0.05 < bias_20 <= 0.12: score += 5
            return score
        except: return 0

    def fetch_and_score(self, row, target_date=None):
        symbol = row['代码']
        try:
            hist = ak.stock_zh_a_hist(symbol=symbol, period="daily", adjust="qfq")
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
        except: return None

    # --- 修正后的邮件发送逻辑 ---
        def send_email_report(self, df_top10):
        sender = os.getenv('EMAIL_USER')
        password = os.getenv('EMAIL_PASS')
        receiver = os.getenv('EMAIL_RECEIVER')
        if not all([sender, password, receiver]):
            console.print("[yellow]未检测到完整邮箱配置，跳过邮件发送。[/yellow]")
            return

        date_str = datetime.now().strftime('%Y-%m-%d')
        
        # 1. 先定义头部内容
        head = f"🚀 Sequoia-X 今日选股报告 ({date_str})\n"
        breadth = f"当前全市场多头占比: {self.market_breadth:.1f}%\n"
        line_sep = "-" * 40 + "\n"
        
        content = head + breadth + line_sep

        # 2. 逐行添加数据，避免超长字符串
        for _, row in df_top10.iterrows():
            s = str(row['综合评分'])
            c = str(row['代码'])
            n = str(row['名称'])
            p = str(row['最新价'])
            z = str(row['涨跌幅'])
            h = str(row['换手率'])
            
            # 拆分拼接，确保万无一失
            item = "【" + s + "分】" + c + " " + n + " | 现价:" + p + " | 涨幅:" + z + "% | 换手:" + h + "%\n"
            content += item
        
        msg = MIMEText(content, 'plain', 'utf-8')
        msg['From'] = sender
        msg['To'] = receiver
        msg['Subject'] = Header("Sequoia-X 选股报告 - " + date_str, 'utf-8')

        try:
            smtp_server = "smtp.qq.com" 
            with smtplib.SMTP_SSL(smtp_server, 465) as server:
                server.login(sender, password)
                server.sendmail(sender, [receiver], msg.as_string())
            console.print("[bold green]邮件报告已成功发送！[/bold green]")
        except Exception as e:
            console.print(f"[bold red]邮件发送失败: {e}[/bold red]")


    def run(self):
        console.print(f"[bold cyan]▶ Sequoia-X 启动模式: {RUN_MODE}[/bold cyan]")
        df_spot = ak.stock_zh_a_spot_em()
        df_spot = df_spot[~df_spot['名称'].str.contains("ST|退")]
        df_spot = df_spot[df_spot['成交额'] > self.min_volume]
        
        sample = df_spot.head(100)
        above_count = (sample['最新价'] > sample['最新价'].rolling(20).mean().fillna(0)).sum()
        self.market_breadth = (above_count / 100) * 100
        
        results = []
        target = BACKTEST_DATE if RUN_MODE == "backtest" else None
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [executor.submit(self.fetch_and_score, row, target) for _, row in df_spot.iterrows()]
            for future in as_completed(futures):
                res = future.result()
                if res: results.append(res)
        
        final_df = pd.DataFrame(results)
        if not final_df.empty:
            final_df = final_df.sort_values("综合评分", ascending=False)
            # 打印表格
            table = Table(title="Sequoia-X 选股结果")
            table.add_column("评分", style="red")
            table.add_column("代码", style="cyan")
            table.add_column("名称")
            table.add_column("最新价")
            for _, row in final_df.head(15).iterrows():
                table.add_row(str(row['综合评分']), row['代码'], row['名称'], str(row['最新价']))
            console.print(table)

            if RUN_MODE == "realtime":
                self.send_email_report(final_df.head(10))
        else:
            console.print("[yellow]今日无符合条件的选股结果。[/yellow]")

if __name__ == "__main__":
    start = time.time()
    SequoiaUltimate().run()
    print(f"任务完成，总耗时: {time.time() - start:.2f}s")
