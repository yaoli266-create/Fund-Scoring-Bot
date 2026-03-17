import os
import sys

# 【关键】必须在所有 import 之前强制禁用代理，防止 v2rayN 干扰
os.environ['NO_PROXY'] = '*'
os.environ['no_proxy'] = '*'
os.environ['http_proxy'] = ''
os.environ['https_proxy'] = ''

import pandas as pd
import numpy as np
import baostock as bs
import akshare as ak
import requests
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

# --- 配置区域 ---
MIN_VOLUME = 150_000_000  # 成交额门槛：1.5亿
SCORE_THRESHOLD = 60      # 评分入选门槛
MAX_THREADS = 10          # Baostock 并发限制
RUN_MODE = "realtime"     # 运行模式

# 邮件配置 (从环境变量读取，安全性更高)
EMAIL_HOST = "smtp.qq.com"
EMAIL_PORT = 465
EMAIL_USER = os.getenv("EMAIL_USER", "你的邮箱@qq.com")
EMAIL_PASS = os.getenv("EMAIL_PASS", "你的授权码")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER", "接收邮箱@qq.com")

console = Console()

class SequoiaUltimateV3:
    def __init__(self):
        self.results = []
        # 自动计算回溯日期
        self.end_date = datetime.now().strftime("%Y-%m-%d")
        self.start_date = (datetime.now() - timedelta(days=100)).strftime("%Y-%m-%d")

    def get_fast_snapshot(self):
        """核心改进：使用国内极速接口获取全市场快照"""
        # 东方财富/腾讯后端通用接口，支持 5000+ 股票，极速且不封IP
        url = "https://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=6000&po=1&np=1&ut=bd1d9ddb04089700cf9c27f6f7426281&fltt=2&invt=2&fid=f3&fs=m:0+t:6,m:0+t:80,m:1+t:2,m:1+t:23&fields=f12,f14,f6"
        try:
            resp = requests.get(url, timeout=5)
            data = resp.json()
            if data and 'data' in data and 'diff' in data['data']:
                df = pd.DataFrame(data['data']['diff'])
                df.columns = ['代码', '名称', '成交额']
                return df
        except Exception as e:
            console.print(f"[yellow]极速接口失效: {e}，尝试备用接口...[/yellow]")
        return pd.DataFrame()

    def calculate_score(self, df):
        """四维度量化评分系统"""
        if len(df) < 20: return 0
        
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)
        df['amount'] = df['amount'].astype(float)
        
        last_close = df['close'].iloc[-1]
        ma20 = df['close'].rolling(20).mean().iloc[-1]
        ma60 = df['close'].rolling(60).mean().iloc[-1]
        
        # 1. 趋势得分 (40分): 价格在均线之上且均线多头
        score_trend = 0
        if last_close > ma20: score_trend += 20
        if ma20 > ma60: score_trend += 20
        
        # 2. 量能得分 (25分): 近3日放量
        vol_ma5 = df['volume'].rolling(5).mean().iloc[-1]
        score_vol = 25 if df['volume'].iloc[-1] > vol_ma5 * 1.2 else 10
        
        # 3. 换手得分 (20分): 活跃度
        avg_turn = df['turn'].astype(float).tail(5).mean()
        score_turn = 20 if 3 < avg_turn < 10 else 10
        
        # 4. 安全边际 (15分): 距离20日线不要太远（防止追高）
        dist = (last_close - ma20) / ma20
        score_safety = 15 if dist < 0.1 else 5
        
        return score_trend + score_vol + score_turn + score_safety

    def fetch_and_score(self, stock):
        symbol = f"{'sh' if stock['代码'].startswith('6') else 'sz'}.{stock['代码']}"
        try:
            rs = bs.query_history_k_data_plus(
                symbol, "date,code,close,volume,amount,turn",
                start_date=self.start_date, end_date=self.end_date,
                frequency="d", adjustflag="3"
            )
            data = []
            while (rs.error_code == '0') & rs.next():
                data.append(rs.get_row_data())
            
            if not data: return None
            
            df = pd.DataFrame(data, columns=rs.fields)
            score = self.calculate_score(df)
            
            if score >= SCORE_THRESHOLD:
                return {
                    "代码": stock['代码'],
                    "名称": stock['名称'],
                    "现价": df['close'].iloc[-1],
                    "成交额": f"{float(stock['成交额'])/1e8:.2f}亿",
                    "综合评分": score
                }
        except:
            pass
        return None

    def send_report(self, df):
        """发送邮件及Excel附件"""
        if df.empty: return
        filename = f"Sequoia_Report_{datetime.now().strftime('%Y%m%d')}.xlsx"
        df.to_excel(filename, index=False)
        
        msg = MIMEMultipart()
        msg['Subject'] = f"A股量化选股报告 - {self.end_date}"
        msg['From'] = EMAIL_USER
        msg['To'] = EMAIL_RECEIVER
        
        body = f"今日筛选完成，共有 {len(df)} 只股票入选。\n详情见附件表格。"
        msg.attach(MIMEText(body, 'plain'))
        
        with open(filename, "rb") as f:
            part = MIMEApplication(f.read(), Name=filename)
            part['Content-Disposition'] = f'attachment; filename="{filename}"'
            msg.attach(part)
            
        try:
            with smtplib.SMTP_SSL(EMAIL_HOST, EMAIL_PORT) as server:
                server.login(EMAIL_USER, EMAIL_PASS)
                server.send_message(msg)
            console.print("[green]邮件报告发送成功！[/green]")
        except Exception as e:
            console.print(f"[red]邮件发送失败: {e}[/red]")

    def run(self):
        console.print(f"[bold cyan]▶ Sequoia-X V3.3 启动 (模式: {RUN_MODE})[/bold cyan]")
        bs.login()
        
        # 1. 获取快照
        df_spot = pd.DataFrame()
        with console.status("[bold yellow]正在获取市场快照..."):
            df_spot = self.get_fast_snapshot()
            if df_spot.empty:
                try: df_spot = ak.stock_zh_a_spot_em()[['代码', '名称', '成交额']]
                except: pass
        
        if df_spot.empty:
            console.print("[red]无法获取快照，请检查网络。[/red]")
            return

        # 2. 初筛
        df_spot = df_spot[~df_spot['名称'].str.contains("ST|退")]
        df_spot = df_spot[df_spot['成交额'].astype(float) > MIN_VOLUME]
        candidates = df_spot.to_dict('records')
        
        # 3. 深度多线程评分
        final_list = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]正在进行深度分析...", total=len(candidates))
            with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
                futures = [executor.submit(self.fetch_and_score, s) for s in candidates]
                for future in as_completed(futures):
                    res = future.result()
                    if res: final_list.append(res)
                    progress.update(task, advance=1)

        # 4. 展示与输出
        if final_list:
            df_final = pd.DataFrame(final_list).sort_values(by="综合评分", ascending=False)
            table = Table(title=f"Sequoia-X 选股结果 ({self.end_date})")
            for col in df_final.columns: table.add_column(col, style="magenta")
            for _, row in df_final.iterrows():
                table.add_row(*[str(v) for v in row.values])
            console.print(table)
            
            # 发送邮件
            self.send_report(df_final)
        else:
            console.print("[yellow]今日无入选标的。[/yellow]")
            
        bs.logout()

if __name__ == "__main__":
    start_time = datetime.now()
    app = SequoiaUltimateV3()
    app.run()
    console.print(f"任务耗时: {datetime.now() - start_time}")
