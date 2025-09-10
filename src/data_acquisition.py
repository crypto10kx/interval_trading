"""
数据获取模块 - 任务1.1: 获取Binance排名前10的USDT期货交易对
从Binance获取当前24小时成交量排名前10的期货交易对，确保有2023年6月前5分钟K线数据
"""

import ccxt
import pandas as pd
import json
import time
from datetime import datetime
import os
from typing import List, Dict, Any
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_top_usdt_symbols(exchange: ccxt.Exchange, limit: int = 10) -> List[Dict[str, Any]]:
    """
    获取24小时成交量排名前10的USDT期货交易对
    
    Args:
        exchange: CCXT交易所实例
        limit: 返回的交易对数量限制
        
    Returns:
        包含symbol和24h成交量的字典列表
    """
    try:
        # 获取所有市场信息
        markets = exchange.load_markets()
        logger.info(f"成功加载 {len(markets)} 个交易对")
        
        # 筛选USDT期货交易对并获取24小时成交量
        usdt_symbols = []
        
        for symbol, market in markets.items():
            # 检查是否为USDT期货交易对且市场活跃（包括永续合约）
            # 永续合约通常以USDT结尾，且类型为swap或future
            if (market['quote'] == 'USDT' and 
                market['active'] and 
                market['type'] in ['future', 'swap'] and
                'PERP' not in symbol):  # 排除有到期日的合约
                
                try:
                    # 获取24小时成交量数据
                    ticker = exchange.fetch_ticker(symbol)
                    if ticker and ticker['quoteVolume']:  # quoteVolume为USDT成交量
                        usdt_symbols.append({
                            'symbol': symbol,
                            'volume_24h': ticker['quoteVolume'],
                            'base': market['base'],
                            'quote': market['quote']
                        })
                        logger.debug(f"添加交易对: {symbol}, 24h成交量: {ticker['quoteVolume']:.2f}")
                except Exception as e:
                    logger.warning(f"获取 {symbol} 成交量失败: {e}")
                    continue
        
        # 按24小时成交量降序排序
        usdt_symbols.sort(key=lambda x: x['volume_24h'], reverse=True)
        
        # 返回前limit个
        top_symbols = usdt_symbols[:limit]
        logger.info(f"成功筛选出前{len(top_symbols)}个USDT期货交易对")
        
        return top_symbols
        
    except Exception as e:
        logger.error(f"获取USDT期货交易对失败: {e}")
        raise


def check_kline_data_availability(exchange: ccxt.Exchange, symbol: str, 
                                required_start_date: str = '2023-06-01') -> Dict[str, Any]:
    """
    检查交易对的K线数据起始时间是否早于指定日期
    
    Args:
        exchange: CCXT交易所实例
        symbol: 交易对符号
        required_start_date: 要求的起始日期
        
    Returns:
        包含检查结果的字典
    """
    try:
        # 转换日期为时间戳
        required_timestamp = int(datetime.strptime(required_start_date, '%Y-%m-%d').timestamp() * 1000)
        
        # 尝试获取历史K线数据，从2023年1月开始
        start_2023 = int(datetime.strptime('2023-01-01', '%Y-%m-%d').timestamp() * 1000)
        
        # 获取2023年1月开始的K线数据（只获取少量数据用于验证）
        historical_klines = exchange.fetch_ohlcv(symbol, '5m', since=start_2023, limit=10)
        
        if not historical_klines:
            # 如果无法获取历史数据，尝试获取最早的可用数据
            earliest_klines = exchange.fetch_ohlcv(symbol, '5m', limit=1)
            if not earliest_klines:
                return {
                    'available': False,
                    'earliest_timestamp': None,
                    'earliest_date': None,
                    'reason': '无法获取K线数据'
                }
            earliest_timestamp = earliest_klines[0][0]
        else:
            # 使用历史数据中的第一个K线
            earliest_timestamp = historical_klines[0][0]
        
        earliest_date = datetime.fromtimestamp(earliest_timestamp / 1000).strftime('%Y-%m-%d')
        
        # 检查是否早于要求日期
        is_available = earliest_timestamp < required_timestamp
        
        return {
            'available': is_available,
            'earliest_timestamp': earliest_timestamp,
            'earliest_date': earliest_date,
            'reason': '数据可用' if is_available else f'数据起始时间 {earliest_date} 晚于要求时间 {required_start_date}'
        }
        
    except Exception as e:
        logger.warning(f"检查 {symbol} K线数据失败: {e}")
        return {
            'available': False,
            'earliest_timestamp': None,
            'earliest_date': None,
            'reason': f'检查失败: {str(e)}'
        }


def fetch_top_symbols_with_validation(required_start_date: str = '2023-06-01', 
                                    limit: int = 10, 
                                    max_retries: int = 3) -> List[Dict[str, Any]]:
    """
    获取并验证排名前10的USDT期货交易对，确保有足够的K线数据
    
    Args:
        required_start_date: 要求的K线数据起始日期
        limit: 返回的交易对数量
        max_retries: 最大重试次数
        
    Returns:
        验证通过的交易对列表
    """
    exchange = None
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # 创建Binance交易所实例
            exchange = ccxt.binance({
                'enableRateLimit': True,
                'timeout': 30000,  # 30秒超时
                'rateLimit': 1200   # 限制请求频率
            })
            
            logger.info(f"开始获取排名前{limit}的USDT期货交易对 (重试 {retry_count + 1}/{max_retries})")
            
            # 获取排名前10的USDT期货交易对
            top_symbols = get_top_usdt_symbols(exchange, limit * 2)  # 获取更多候选，用于筛选
            
            if not top_symbols:
                raise Exception("未找到任何USDT期货交易对")
            
            logger.info(f"开始验证K线数据可用性...")
            
            # 验证每个交易对的K线数据
            validated_symbols = []
            
            for symbol_info in top_symbols:
                symbol = symbol_info['symbol']
                logger.info(f"验证交易对: {symbol}")
                
                # 检查K线数据可用性
                kline_check = check_kline_data_availability(exchange, symbol, required_start_date)
                
                if kline_check['available']:
                    validated_symbols.append({
                        'symbol': symbol,
                        'start_time': kline_check['earliest_timestamp'],
                        'start_date': kline_check['earliest_date'],
                        'volume_24h': symbol_info['volume_24h'],
                        'base': symbol_info['base'],
                        'quote': symbol_info['quote']
                    })
                    logger.info(f"✓ {symbol} 验证通过，数据起始: {kline_check['earliest_date']}")
                else:
                    logger.warning(f"✗ {symbol} 验证失败: {kline_check['reason']}")
                
                # 如果已找到足够的交易对，停止验证
                if len(validated_symbols) >= limit:
                    break
                
                # 避免API限频，请求间隔50ms
                time.sleep(0.05)
            
            if len(validated_symbols) < limit:
                logger.warning(f"只找到 {len(validated_symbols)} 个符合条件的交易对，少于要求的 {limit} 个")
            
            logger.info(f"成功验证 {len(validated_symbols)} 个交易对")
            return validated_symbols[:limit]
            
        except Exception as e:
            retry_count += 1
            logger.error(f"第 {retry_count} 次尝试失败: {e}")
            
            if retry_count < max_retries:
                wait_time = 2 ** retry_count  # 指数退避
                logger.info(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
            else:
                logger.error(f"所有重试失败，无法获取交易对数据")
                raise
    
    return []


def save_symbols_to_json(symbols: List[Dict[str, Any]], filepath: str = 'data/symbols.json') -> None:
    """
    保存交易对信息到JSON文件
    
    Args:
        symbols: 交易对信息列表
        filepath: 保存路径
    """
    try:
        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # 保存到JSON文件
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(symbols, f, indent=2, ensure_ascii=False)
        
        logger.info(f"成功保存 {len(symbols)} 个交易对到 {filepath}")
        
    except Exception as e:
        logger.error(f"保存JSON文件失败: {e}")
        raise


def main():
    """
    主函数：执行任务1.1的完整流程
    """
    try:
        logger.info("开始执行任务1.1：获取Binance排名前10的USDT期货交易对")
        
        # 获取并验证交易对
        symbols = fetch_top_symbols_with_validation(
            required_start_date='2023-06-01',
            limit=10,
            max_retries=3
        )
        
        if not symbols:
            raise Exception("未找到任何符合条件的交易对")
        
        # 保存结果
        save_symbols_to_json(symbols, 'data/symbols.json')
        
        # 输出结果摘要
        logger.info("=" * 50)
        logger.info("任务1.1完成 - 期货交易对获取结果:")
        logger.info("=" * 50)
        
        for i, symbol_info in enumerate(symbols, 1):
            logger.info(f"{i:2d}. {symbol_info['symbol']:<12} | "
                       f"数据起始: {symbol_info['start_date']:<12} | "
                       f"24h成交量: {symbol_info['volume_24h']:>15,.2f} USDT")
        
        logger.info("=" * 50)
        logger.info(f"总计: {len(symbols)} 个交易对")
        logger.info("结果已保存到 data/symbols.json")
        
        return symbols
        
    except Exception as e:
        logger.error(f"任务1.1执行失败: {e}")
        raise


if __name__ == "__main__":
    # 执行主函数
    symbols = main()
    
    # 验证结果
    assert len(symbols) > 0, "未获取到任何交易对"
    assert all('symbol' in s and 'start_time' in s for s in symbols), "交易对数据格式错误"
    assert all(s['start_time'] < int(datetime.strptime('2023-06-01', '%Y-%m-%d').timestamp() * 1000) 
               for s in symbols), "存在交易对数据起始时间晚于2023-06-01"
    
    print("✓ 任务1.1测试通过")
