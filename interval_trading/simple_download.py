#!/usr/bin/env python3
"""
最简单的下载方案 - 直接下载，无复杂逻辑
"""

import ccxt
import pandas as pd
import numpy as np
import time
import os
from datetime import datetime

def download_simple(symbol, start_date, end_date):
    """最简单的下载方法"""
    print(f"开始下载 {symbol}...")
    
    # 创建交易所
    exchange = ccxt.binance({'enableRateLimit': True})
    
    # 转换时间
    start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
    
    all_data = []
    current_ts = start_ts
    batch_count = 0
    
    while current_ts < end_ts:
        try:
            batch_count += 1
            print(f"  批次 {batch_count}: {datetime.fromtimestamp(current_ts/1000)}")
            
            # 下载一批数据
            batch = exchange.fetch_ohlcv(symbol, '5m', since=current_ts, limit=1000)
            
            if not batch:
                print(f"  无更多数据，停止")
                break
            
            # 过滤时间范围
            filtered_batch = [kline for kline in batch if kline[0] <= end_ts]
            all_data.extend(filtered_batch)
            
            print(f"  获取 {len(filtered_batch)} 条，总计 {len(all_data)} 条")
            
            # 更新时间戳
            current_ts = batch[-1][0] + 1
            
            # 如果已经达到结束时间，停止
            if batch[-1][0] >= end_ts:
                break
                
            # 简单等待
            time.sleep(0.1)
            
        except Exception as e:
            print(f"  批次 {batch_count} 失败: {e}")
            current_ts += 1000 * 60 * 60 * 24  # 跳过1天
            time.sleep(1)
    
    if not all_data:
        print(f"❌ {symbol} 下载失败")
        return None
    
    # 转换为DataFrame
    df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    # 添加对数价格
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        df[f'log_{col}'] = np.log(df[col].clip(lower=1e-6))
    
    # 成交量Z-Score
    df['volume_zscore'] = (df['volume'] - df['volume'].mean()) / df['volume'].std()
    
    print(f"✅ {symbol} 完成: {len(df)} 条记录")
    return df

def main():
    print("=== 简单直接下载方案 ===")
    
    # 只下载前5个主要交易对，避免卡顿
    symbols = [
        'BTC/USDT:USDT',
        'ETH/USDT:USDT', 
        'BNB/USDT:USDT',
        'XRP/USDT:USDT',
        'ADA/USDT:USDT'
    ]
    
    # 只下载1年数据，减少下载量
    start_date = '2024-01-01'
    end_date = '2024-12-31'
    
    print(f"下载 {len(symbols)} 个交易对，时间范围: {start_date} 到 {end_date}")
    
    os.makedirs('data', exist_ok=True)
    
    for i, symbol in enumerate(symbols):
        print(f"\n{'='*50}")
        print(f"进度: {i+1}/{len(symbols)}")
        print(f"{'='*50}")
        
        try:
            df = download_simple(symbol, start_date, end_date)
            
            if df is not None:
                # 保存文件
                filename = f'data/{symbol.replace("/", "_").replace(":", "_")}_5m.csv'
                df.to_csv(filename, index=False)
                
                file_size = os.path.getsize(filename) / (1024 * 1024)
                print(f"保存到: {filename} ({file_size:.2f} MB)")
            else:
                print(f"跳过 {symbol}")
                
        except Exception as e:
            print(f"❌ {symbol} 处理失败: {e}")
        
        # 简单等待
        if i < len(symbols) - 1:
            print("等待2秒...")
            time.sleep(2)
    
    print(f"\n🎉 下载完成!")
    
    # 检查结果
    files = [f for f in os.listdir('data') if f.endswith('.csv')]
    print(f"生成文件: {len(files)} 个")
    for file in files:
        file_path = os.path.join('data', file)
        file_size = os.path.getsize(file_path) / (1024 * 1024)
        print(f"  - {file}: {file_size:.2f} MB")

if __name__ == "__main__":
    main()

