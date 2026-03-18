import os
import time
import pandas as pd
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication

# 引入通达信底层接口
from pytdx.hq import TdxHq_API

# --- 配置区域 ---
MIN_VOLUME = 150_000_000  # 成交额门槛：1.5亿
SCORE_THRESHOLD = 60      # 评分入选门槛
MAX_THREADS = 8           # 多线程并发数（通达信节点建议不要太高，8-10最佳）
RUN_MODE = "pytdx-realtime"

# 通达信行情服务器 (推荐招商证券或华泰证券的节点，稳定且速度快)
TDX_IP = '119.147.212.81' 
TDX_PORT = 7709

# 邮件配置
EMAIL_HOST = "smtp.qq.com"
EMAIL_PORT = 465
EMAIL_USER = os.getenv("EMAIL_USER", "你的邮箱@qq.com")
EMAIL_PASS = os.getenv("EMAIL_PASS", "你的授权码")
EMAIL_RECEIVER = os.getenv("EMAIL_RECEIVER", "接收邮箱@qq.com")

console = Console()

class SequoiaTDXV4:
    def __init__(self):
        self.end_date = datetime.now().strftime("%Y-%m-%d")

    def get_a_share_list(self, api):
        """遍历获取沪深两市所有 A 股的基础信息 (代码与名称)"""
        stock_list = []
        # 0: 深圳, 1: 上海
        for market in [0, 1]:
            # 通达信接口每次最多返回 1000 条，循环拉取直到为空
            for i in range(0, 6000, 1000):
                data = api.get_security_list(market, i)
                if not data:
                    break
                for row in data:
                    code = row['code']
                    # 只保留 A 股核心主板、创业板、科创板代码
                    if code.startswith(('00', '30', '60')):
                        # 过滤 ST 和退市股
                        if 'ST' not in row['name'] and '退' not in row['name']:
                            stock_list.append({
                                'market': market,
                                'code': code,
                                'name': row['name']
                            })
        return pd.DataFrame(stock_list)

    def get_fast_snapshot(self, api, df_stocks):
        """批量获取全市场实时盘口快照 (替代原有的 HTTP 接口)"""
        quotes = []
        stock_tuples = [(row['market'], row['code']) for _, row in df_stocks.iterrows()]
        
        # 通达信限制每次最多查询 80 只股票的快照
        chunk_size = 80
        for i in range(0, len(stock_tuples), chunk_size):
            chunk = stock_tuples[i:i + chunk_size]
            data = api.get_security_quotes(chunk)
            if data:
                quotes.extend(data)
                
        df_quotes = pd.DataFrame(quotes)
        if df_quotes.empty:
            return pd.DataFrame()
            
        # 提取所需的现价和成交额数据
        # 通达信返回的 price 是真实价格，amount 是真实成交额(元)
        df_quotes = df_quotes[['code', 'price', 'amount']]
        df_quotes.rename(columns={'code': '代码', 'price': '现价', 'amount': '成交额'}, inplace=True)
        
        # 与名称合并
        df_spot = pd.merge(df_stocks, df_quotes, on='代码', how='inner')
        return df_spot

    def calculate_score(self, df):
        """四维度量化评分系统 (适配 pytdx 数据结构)"""
        if len(df) < 20: return 0
        
        for col in ['close', 'vol', 'amount']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        df = df.dropna(subset=['close', 'vol', 'amount'])
        if len(df) < 20: return 0
        
        last_close = df['close'].iloc[-1]
        ma20 = df['close'].rolling(20).mean().iloc[-1]
        ma60 = df['close'].rolling(60).mean().iloc[-1]
        
        # 1. 趋势得分 (40分)
        score_trend = 0
        if last_close > ma20: score_trend += 20
        if ma20 > ma60: score_trend += 20
        
        # 2. 量能得分 (25分) - 通达信中 vol 代表成交量(手)
        vol_ma5 = df['vol'].rolling(5).mean().iloc[-1]
        score_vol = 25 if df['vol'].iloc[-1] > vol_ma5 * 1.2 else 10
        
        # 3. 活跃度得分 (20分) - 由于 pytdx 基础 K 线不含直接的换手率，用成交额替代评估活跃度
        avg_amount = df['amount'].tail(5).mean()
        score_turn = 20 if avg_amount > 2_000_000_00 else 10 # 近5日均成交额大于2亿算高活跃
        
        # 4. 安全边际 (15分)
        if ma20 == 0: return 0
        dist = (last_close - ma20) / ma20
        score_safety = 15 if dist < 0.1 else 5
        
        return score_trend + score_vol + score_turn + score_safety

    def fetch_and_score(self, stock):
        """子线程工作函数：必须在线程内部独立创建 API 连接以保证线程安全"""
        sub_api = TdxHq_API(raise_exception=False, auto_retry=True)
        try:
            if not sub_api.connect(TDX_IP, TDX_PORT):
                return None
                
            # category: 9 代表日 K 线。拉取近 100 天数据计算均线
            # 注意：此处获取的是基础 K 线，量化实盘要求极高时，应进一步做前复权处理
            data = sub_api.get_security_bars(category=9, market=stock['market'], code=stock['代码'], start=0, count=100)
            
            if not data: 
                return None
                
            df = pd.DataFrame(data)
            score = self.calculate_score(df)
            
            if score >= SCORE_THRESHOLD:
                return {
                    "代码": stock['代码'],
                    "名称": stock['名称'],
                    "现价": stock['现价'],
                    "成交额": f"{float(stock['成交额'])/1e8:.2f}亿",
                    "综合评分": score
                }
        except Exception as e:
            pass
        finally:
            sub_api.disconnect()
            
        return None

    def send_report(self, df):
        if df.empty: return
        filename = f"Sequoia_Report_{datetime.now().strftime('%Y%m%d')}.xlsx"
        df.to_excel(filename, index=False)
        
        msg = MIMEMultipart()
        msg['Subject'] = f"A股量化选股报告 (TDX版) - {self.end_date}"
        msg['From'] = EMAIL_USER
        msg['To'] = EMAIL_RECEIVER
        
        body = f"今日基于通达信直连筛选完成，共有 {len(df)} 只股票入选。\n详情见附件表格。"
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
        console.print(f"[bold cyan]▶ Sequoia-X V4.0 启动 (底层: {RUN_MODE})[/bold cyan]")
        
        # 1. 建立主连接获取快照
        main_api = TdxHq_API()
        with console.status("[bold yellow]正在连接通达信主行情服务器..."):
            if not main_api.connect(TDX_IP, TDX_PORT):
                console.print("[red]通达信服务器连接失败，请检查网络或更换节点 IP。[/red]")
                return
                
        with console.status("[bold yellow]正在扫描全市场 A 股代码..."):
            df_stocks = self.get_a_share_list(main_api)
            
        with console.status(f"[bold yellow]正在获取 {len(df_stocks)} 只股票实时快照..."):
            df_spot = self.get_fast_snapshot(main_api, df_stocks)
            
        main_api.disconnect() # 获取完快照后，主连接断开释放资源

        if df_spot.empty:
            console.print("[red]获取市场快照失败。[/red]")
            return

        # 2. 初筛逻辑
        df_spot['成交额'] = pd.to_numeric(df_spot['成交额'], errors='coerce')
        df_spot = df_spot[df_spot['成交额'] > MIN_VOLUME]
        candidates = df_spot.to_dict('records')
        
        console.print(f"[green]初筛完成，{len(candidates)} 只标的进入深度多线程计算。[/green]")

        # 3. 深度多线程评分
        final_list = []
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:
            task = progress.add_task("[cyan]正在通过 TDX 专线获取 K 线并分析...", total=len(candidates))
            
            with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
                futures = [executor.submit(self.fetch_and_score, s) for s in candidates]
                for future in as_completed(futures):
                    res = future.result()
                    if res: final_list.append(res)
                    progress.update(task, advance=1)

        # 4. 展示与输出
        if final_list:
            df_final = pd.DataFrame(final_list).sort_values(by="综合评分", ascending=False)
            table = Table(title=f"Sequoia-X V4 选股结果 ({self.end_date})")
            for col in df_final.columns: table.add_column(col, style="magenta")
            for _, row in df_final.iterrows():
                table.add_row(*[str(v) for v in row.values])
            console.print(table)
            
            # 发送邮件
            self.send_report(df_final)
        else:
            console.print("[yellow]今日无入选标的。[/yellow]")

if __name__ == "__main__":
    start_time = time.time()
    app = SequoiaTDXV4()
    app.run()
    console.print(f"任务耗时: {time.time() - start_time:.2f} 秒")
