#!/usr/bin/env python3
"""
优化的批量下载脚本 - 修复卡顿问题
使用实际可用的交易对，优化重试机制
"""

import psutil
import sys
import os
import time
import pandas as pd
from datetime import datetime
sys.path.append('src')

def monitor_memory():
    """监控内存使用情况"""
    memory = psutil.virtual_memory()
    used_gb = memory.used / (1024**3)
    total_gb = memory.total / (1024**3)
    percent = memory.percent
    print(f'内存使用: {used_gb:.2f}GB / {total_gb:.2f}GB ({percent:.1f}%)')
    return used_gb < 8.0

def get_available_symbols():
    """获取实际可用的期货交易对"""
    import ccxt
    
    print('=== 获取可用的期货交易对 ===')
    exchange = ccxt.binance({'enableRateLimit': True})
    markets = exchange.load_markets()
    
    # 获取期货交易对
    futures_symbols = [symbol for symbol, market in markets.items() 
                      if market['type'] in ['future', 'swap'] and 'USDT' in symbol and 'PERP' not in symbol]
    
    print(f'总期货交易对数: {len(futures_symbols)}')
    
    # 优先选择的交易对（按成交量排序）
    preferred_symbols = [
        'BTC/USDT:USDT',
        'ETH/USDT:USDT', 
        'BNB/USDT:USDT',
        'XRP/USDT:USDT',
        'ADA/USDT:USDT',
        'SOL/USDT:USDT',
        'DOGE/USDT:USDT',
        'AVAX/USDT:USDT',
        'DOT/USDT:USDT',
        'LINK/USDT:USDT',
        'UNI/USDT:USDT',
        'LTC/USDT:USDT',
        'BCH/USDT:USDT',
        'ATOM/USDT:USDT',
        'NEAR/USDT:USDT'
    ]
    
    # 筛选可用的交易对
    available_symbols = []
    for symbol in preferred_symbols:
        if symbol in futures_symbols:
            available_symbols.append(symbol)
            print(f'✅ {symbol} - 可用')
        else:
            print(f'❌ {symbol} - 不可用')
    
    # 如果不足10个，补充其他可用的
    if len(available_symbols) < 10:
        print(f'\\n补充其他可用的交易对:')
        for symbol in futures_symbols:
            if symbol not in available_symbols and len(available_symbols) < 10:
                available_symbols.append(symbol)
                print(f'  + {symbol}')
    
    print(f'\\n最终选择 {len(available_symbols)} 个交易对:')
    for i, symbol in enumerate(available_symbols):
        print(f'  {i+1}. {symbol}')
    
    return available_symbols

def download_single_symbol(symbol, start_date, end_date):
    """下载单个交易对的数据"""
    from kline_fetcher import fetch_kline_data
    
    print(f'\\n=== 开始下载 {symbol} ===')
    print(f'时间范围: {start_date} 到 {end_date}')
    
    start_time = time.time()
    
    try:
        # 若已存在CSV则跳过下载，避免重复
        os.makedirs('data', exist_ok=True)
        filename = f'data/{symbol.replace("/", "_").replace(":", "_")}_5m.csv'
        if os.path.exists(filename):
            print(f'  - 发现已存在CSV，跳过下载: {filename}')
            try:
                df = pd.read_csv(filename)
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return True, len(df), 0.0
            except Exception:
                print('  - 本地CSV读取失败，将重新下载')

        # 下载数据
        df = fetch_kline_data(symbol, start_date, end_date)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f'✅ {symbol} 下载完成:')
        print(f'  - 数据量: {len(df)} 条记录')
        print(f'  - 耗时: {duration:.2f} 秒')
        print(f'  - 时间范围: {df["timestamp"].min()} 到 {df["timestamp"].max()}')
        
        # 保存到CSV
        # 文件名与上方一致
        df.to_csv(filename, index=False)
        print(f'  - 已保存到: {filename}')
        
        # 检查文件大小
        file_size = os.path.getsize(filename) / (1024 * 1024)  # MB
        print(f'  - 文件大小: {file_size:.2f} MB')
        
        return True, len(df), duration
        
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        print(f'❌ {symbol} 下载失败: {e}')
        print(f'  - 耗时: {duration:.2f} 秒')
        return False, 0, duration

def main():
    print('=== 优化的批量下载脚本 - 修复卡顿问题 ===')
    print(f'开始时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'初始内存状态:')
    monitor_memory()
    
    # 获取可用的交易对
    symbols = get_available_symbols()
    
    if not symbols:
        print('❌ 没有找到可用的交易对')
        return
    
    # 设置时间范围（两年）
    end_date = '2024-12-31'
    start_date = '2023-01-01'
    
    print(f'\\n=== 开始批量下载 ===')
    print(f'时间范围: {start_date} 到 {end_date}')
    print(f'预计总数据量: ~{len(symbols)} 个交易对 × 2年 × 365天 × 288条/天 ≈ {len(symbols) * 2 * 365 * 288:,} 条记录')
    
    # 统计变量
    total_records = 0
    total_duration = 0
    successful_downloads = 0
    failed_downloads = 0
    
    # 逐个下载
    for i, symbol in enumerate(symbols):
        print(f'\\n{"="*60}')
        print(f'进度: {i+1}/{len(symbols)} - {symbol}')
        print(f'{"="*60}')
        
        # 监控内存
        if not monitor_memory():
            print('⚠️ 内存使用过高，停止下载')
            break
        
        # 下载单个交易对
        success, records, duration = download_single_symbol(symbol, start_date, end_date)
        
        # 更新统计
        if success:
            successful_downloads += 1
            total_records += records
        else:
            failed_downloads += 1
        
        total_duration += duration
        
        # 显示进度
        print(f'\\n📊 当前进度统计:')
        print(f'  - 已完成: {i+1}/{len(symbols)} 个交易对')
        print(f'  - 成功: {successful_downloads} 个')
        print(f'  - 失败: {failed_downloads} 个')
        print(f'  - 总记录数: {total_records:,} 条')
        print(f'  - 总耗时: {total_duration:.2f} 秒')
        print(f'  - 平均耗时: {total_duration/(i+1):.2f} 秒/交易对')
        
        # 请求间隔（避免API限制）
        if i < len(symbols) - 1:
            print(f'  - 等待1秒后继续...')
            time.sleep(1)  # 减少等待时间
    
    # 最终统计
    print(f'\\n{"="*60}')
    print(f'🎉 批量下载完成!')
    print(f'{"="*60}')
    print(f'总交易对数: {len(symbols)}')
    print(f'成功下载: {successful_downloads} 个')
    print(f'下载失败: {failed_downloads} 个')
    print(f'总记录数: {total_records:,} 条')
    print(f'总耗时: {total_duration:.2f} 秒')
    print(f'平均耗时: {total_duration/len(symbols):.2f} 秒/交易对')
    print(f'结束时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    
    # 检查data文件夹
    print(f'\\n📁 数据文件检查:')
    data_files = [f for f in os.listdir('data') if f.endswith('.csv')]
    print(f'CSV文件数量: {len(data_files)}')
    for file in data_files:
        file_path = os.path.join('data', file)
        file_size = os.path.getsize(file_path) / (1024 * 1024)
        print(f'  - {file}: {file_size:.2f} MB')
    
    print(f'\\n最终内存状态:')
    monitor_memory()

if __name__ == "__main__":
    main()
