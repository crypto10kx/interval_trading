"""
K线数据获取模块 - 任务1.2: 拉取K线数据
为每个交易对拉取2023年6月1日-2025年6月30日的5分钟K线数据，保存为CSV
优化参数：单次下载1500根K线，请求间隔50ms
"""

import ccxt
import pandas as pd
import numpy as np
import time
import os
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def apply_log_scale(df: pd.DataFrame) -> pd.DataFrame:
    """
    全局对数转换，避免负值
    
    Args:
        df: K线数据DataFrame，必须包含['open', 'high', 'low', 'close', 'volume']列
        
    Returns:
        添加了log_列和volume_zscore列的DataFrame
    """
    logger.info("开始全局对数转换")
    
    # 创建结果DataFrame的副本
    result_df = df.copy()
    
    # 转换价格列为对数刻度，使用clip避免log(0)或负值
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        result_df[f'log_{col}'] = np.log(df[col].clip(lower=1e-6))
        logger.debug(f"转换 {col} 为 log_{col}")
    
    # 对成交量进行Z-Score归一化，突出异常值
    volume_mean = df['volume'].mean()
    volume_std = df['volume'].std()
    
    if volume_std > 0:
        result_df['volume_zscore'] = (df['volume'] - volume_mean) / volume_std
    else:
        # 如果标准差为0，所有值设为0
        result_df['volume_zscore'] = 0.0
        logger.warning("成交量标准差为0，volume_zscore设为0")
    
    logger.info("✓ 全局对数转换完成")
    return result_df


def validate_kline_data_integrity(df: pd.DataFrame, start_date: str, end_date: str) -> bool:
    """
    验证K线数据的完整性
    
    Args:
        df: K线数据DataFrame
        start_date: 期望开始日期
        end_date: 期望结束日期
        
    Returns:
        数据是否完整
    """
    try:
        if df.empty:
            logger.error("数据为空")
            return False
        
        # 检查时间范围
        start_expected = pd.to_datetime(start_date)
        end_expected = pd.to_datetime(end_date)
        
        actual_start = df['timestamp'].min()
        actual_end = df['timestamp'].max()
        
        # 允许1天的时间偏差（考虑到API限制）
        time_tolerance = pd.Timedelta(days=1)
        
        if actual_start > start_expected + time_tolerance:
            logger.warning(f"实际开始时间({actual_start})晚于期望时间({start_expected})")
            return False
        
        if actual_end < end_expected - time_tolerance:
            logger.warning(f"实际结束时间({actual_end})早于期望时间({end_expected})")
            return False
        
        # 检查数据量是否合理（2年约20万条5分钟K线）
        expected_min_records = 200000  # 约2年数据
        if len(df) < expected_min_records * 0.8:  # 允许20%的缺失
            logger.warning(f"数据量不足，期望至少{expected_min_records}条，实际{len(df)}条")
            return False
        
        # 检查价格数据合理性
        if (df['high'] < df['low']).any():
            logger.error("存在high < low的数据")
            return False
        
        if (df[['open', 'high', 'low', 'close', 'volume']] <= 0).any().any():
            logger.error("存在非正数的价格或成交量数据")
            return False
        
        # 检查时间间隔（5分钟K线应该间隔5分钟）
        time_diffs = df['timestamp'].diff().dropna()
        expected_interval = pd.Timedelta(minutes=5)
        tolerance = pd.Timedelta(minutes=1)  # 允许1分钟偏差
        
        irregular_intervals = time_diffs[(time_diffs < expected_interval - tolerance) | 
                                       (time_diffs > expected_interval + tolerance)]
        
        if len(irregular_intervals) > len(df) * 0.05:  # 允许5%的不规则间隔
            logger.warning(f"发现{len(irregular_intervals)}个不规则时间间隔")
            return False
        
        logger.info("数据完整性检查通过")
        return True
        
    except Exception as e:
        logger.error(f"数据完整性检查失败: {e}")
        return False


def fetch_kline_data(symbol: str, start_date: str, end_date: str, 
                    timeframe: str = '5m', batch_size: int = 1500, 
                    request_interval: float = 0.05, max_retries: int = 3) -> pd.DataFrame:
    """
    拉取单个交易对的K线数据
    
    Args:
        symbol: 交易对符号，如 'BTC/USDT'
        start_date: 开始日期，格式 'YYYY-MM-DD'
        end_date: 结束日期，格式 'YYYY-MM-DD'
        timeframe: K线周期，默认 '5m'
        batch_size: 单次请求的K线数量，默认1500
        request_interval: 请求间隔（秒），默认0.05（50ms）
        
    Returns:
        包含K线数据的DataFrame
    """
    try:
        # 创建Binance交易所实例
        exchange = ccxt.binance({
            'enableRateLimit': True,
            'timeout': 30000,  # 30秒超时
            'rateLimit': 50     # 50ms间隔
        })
        
        # 转换日期为时间戳
        start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)
        
        logger.info(f"开始拉取 {symbol} 的K线数据: {start_date} 到 {end_date}")
        
        all_data = []
        current_ts = start_ts
        batch_count = 0
        
        while current_ts <= end_ts:
            retry_count = 0
            batch_success = False
            
            while retry_count < max_retries and not batch_success:
                try:
                    batch_count += 1
                    logger.info(f"拉取第 {batch_count} 批数据，时间戳: {current_ts} (重试 {retry_count + 1}/{max_retries})")
                    
                    # 拉取一批K线数据
                    batch = exchange.fetch_ohlcv(
                        symbol, 
                        timeframe, 
                        since=current_ts, 
                        limit=batch_size
                    )
                    
                    if not batch:
                        logger.warning(f"第 {batch_count} 批数据为空，停止拉取")
                        break
                    
                    # 过滤超出结束时间的数据
                    filtered_batch = [kline for kline in batch if kline[0] <= end_ts]
                    all_data.extend(filtered_batch)
                    
                    # 更新当前时间戳为最后一条K线的时间戳+1
                    current_ts = batch[-1][0] + 1
                    
                    logger.info(f"第 {batch_count} 批完成，获取 {len(filtered_batch)} 条数据，总数据量: {len(all_data)}")
                    
                    batch_success = True
                    
                    # 如果最后一批数据的时间戳已经达到或超过结束时间，停止
                    if batch[-1][0] >= end_ts:
                        break
                        
                except Exception as e:
                    retry_count += 1
                    logger.error(f"拉取第 {batch_count} 批数据失败 (重试 {retry_count}/{max_retries}): {e}")
                    
                    if retry_count < max_retries:
                        # 异常重试间隔5秒
                        logger.info(f"等待5秒后重试...")
                        time.sleep(5)
                    else:
                        logger.error(f"第 {batch_count} 批数据拉取失败，已达到最大重试次数")
                        # 跳过这一批，继续下一批
                        current_ts += batch_size * 5 * 60 * 1000  # 跳过约batch_size个5分钟K线的时间
                        break
            
            if not batch_success:
                logger.warning(f"跳过第 {batch_count} 批数据，继续处理下一批")
            
            # 请求间隔
            time.sleep(request_interval)
        
        if not all_data:
            raise Exception(f"未获取到任何K线数据: {symbol}")
        
        # 转换为DataFrame
        df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        
        # 转换时间戳为datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # 按时间排序并去重
        df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp']).reset_index(drop=True)
        
        # 数据完整性检查
        if not validate_kline_data_integrity(df, start_date, end_date):
            logger.warning(f"{symbol} 数据完整性检查失败，但继续处理")
        
        # 立即应用对数转换
        df = apply_log_scale(df)
        
        logger.info(f"{symbol} K线数据拉取完成: {len(df)} 条记录")
        logger.info(f"时间范围: {df['timestamp'].min()} 到 {df['timestamp'].max()}")
        logger.info(f"已应用对数转换，新增列: {[col for col in df.columns if col.startswith('log_') or col == 'volume_zscore']}")
        
        return df
        
    except Exception as e:
        logger.error(f"拉取 {symbol} K线数据失败: {e}")
        raise


def save_kline_to_csv(df: pd.DataFrame, symbol: str, output_dir: str = 'data') -> str:
    """
    保存K线数据到CSV文件
    
    Args:
        df: K线数据DataFrame
        symbol: 交易对符号
        output_dir: 输出目录
        
    Returns:
        保存的文件路径
    """
    try:
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成文件名（替换/为_）
        filename = f"{symbol.replace('/', '_')}_5m.csv"
        filepath = os.path.join(output_dir, filename)
        
        # 保存到CSV
        df.to_csv(filepath, index=False)
        
        # 获取文件大小
        file_size = os.path.getsize(filepath) / (1024 * 1024)  # MB
        
        logger.info(f"K线数据已保存到: {filepath}")
        logger.info(f"文件大小: {file_size:.2f} MB")
        
        return filepath
        
    except Exception as e:
        logger.error(f"保存CSV文件失败: {e}")
        raise


def load_symbols_from_json(filepath: str = 'data/symbols.json') -> List[Dict[str, Any]]:
    """
    从JSON文件加载交易对列表
    
    Args:
        filepath: JSON文件路径
        
    Returns:
        交易对列表
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            symbols = json.load(f)
        
        logger.info(f"成功加载 {len(symbols)} 个交易对")
        return symbols
        
    except Exception as e:
        logger.error(f"加载交易对列表失败: {e}")
        raise


def fetch_all_symbols_klines(symbols: List[Dict[str, Any]], 
                           start_date: str = '2023-06-01',
                           end_date: str = '2025-06-30',
                           batch_size: int = 1500,
                           request_interval: float = 0.05) -> Dict[str, str]:
    """
    批量拉取所有交易对的K线数据
    
    Args:
        symbols: 交易对列表
        start_date: 开始日期
        end_date: 结束日期
        batch_size: 单次请求的K线数量
        request_interval: 请求间隔（秒）
        
    Returns:
        交易对到文件路径的映射
    """
    results = {}
    
    for i, symbol_info in enumerate(symbols, 1):
        symbol = symbol_info['symbol']
        
        try:
            logger.info(f"开始处理第 {i}/{len(symbols)} 个交易对: {symbol}")
            
            # 拉取K线数据
            df = fetch_kline_data(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                batch_size=batch_size,
                request_interval=request_interval
            )
            
            # 保存到CSV
            filepath = save_kline_to_csv(df, symbol)
            results[symbol] = filepath
            
            logger.info(f"✓ {symbol} 处理完成")
            
        except Exception as e:
            logger.error(f"✗ {symbol} 处理失败: {e}")
            results[symbol] = None
    
    return results


def main():
    """
    主函数：执行任务1.2的完整流程
    """
    try:
        logger.info("开始执行任务1.2：拉取K线数据")
        
        # 加载交易对列表
        symbols = load_symbols_from_json('data/symbols.json')
        
        if not symbols:
            raise Exception("未找到交易对列表，请先运行任务1.1")
        
        # 批量拉取K线数据
        results = fetch_all_symbols_klines(
            symbols=symbols,
            start_date='2023-06-01',
            end_date='2025-06-30',
            batch_size=1500,
            request_interval=0.05
        )
        
        # 统计结果
        success_count = sum(1 for path in results.values() if path is not None)
        total_count = len(results)
        
        logger.info("=" * 60)
        logger.info("任务1.2完成 - K线数据拉取结果:")
        logger.info("=" * 60)
        
        for symbol, filepath in results.items():
            if filepath:
                file_size = os.path.getsize(filepath) / (1024 * 1024)
                logger.info(f"✓ {symbol:<12} | {filepath} | {file_size:>8.2f} MB")
            else:
                logger.info(f"✗ {symbol:<12} | 拉取失败")
        
        logger.info("=" * 60)
        logger.info(f"成功: {success_count}/{total_count} 个交易对")
        
        return results
        
    except Exception as e:
        logger.error(f"任务1.2执行失败: {e}")
        raise


if __name__ == "__main__":
    # 执行主函数
    results = main()
    
    # 验证结果
    assert len(results) > 0, "未获取到任何K线数据"
    success_count = sum(1 for path in results.values() if path is not None)
    assert success_count > 0, "所有交易对K线数据拉取失败"
    
    print("✓ 任务1.2测试通过")
