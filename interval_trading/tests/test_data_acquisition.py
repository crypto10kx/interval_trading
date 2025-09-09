"""
测试用例 - 数据获取模块
验证任务1.1的功能正确性
"""

import unittest
import sys
import os
import json
from datetime import datetime
from unittest.mock import patch, MagicMock

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_acquisition import (
    get_top_usdt_symbols,
    check_kline_data_availability,
    fetch_top_symbols_with_validation,
    save_symbols_to_json
)


class TestDataAcquisition(unittest.TestCase):
    """数据获取模块测试类"""
    
    def setUp(self):
        """测试前准备"""
        self.test_symbols = [
            {
                'symbol': 'BTC/USDT',
                'volume_24h': 1000000.0,
                'base': 'BTC',
                'quote': 'USDT'
            },
            {
                'symbol': 'ETH/USDT', 
                'volume_24h': 800000.0,
                'base': 'ETH',
                'quote': 'USDT'
            }
        ]
        
        self.test_kline_data = [
            [1640995200000, 47000, 48000, 46000, 47500, 1000]  # 2022-01-01的K线数据
        ]
    
    def test_get_top_usdt_symbols(self):
        """测试获取USDT交易对功能"""
        # 模拟CCXT交易所
        mock_exchange = MagicMock()
        mock_exchange.load_markets.return_value = {
            'BTC/USDT': {
                'quote': 'USDT',
                'active': True,
                'type': 'spot',
                'base': 'BTC'
            },
            'ETH/USDT': {
                'quote': 'USDT', 
                'active': True,
                'type': 'spot',
                'base': 'ETH'
            },
            'BTC/BTC': {  # 非USDT交易对
                'quote': 'BTC',
                'active': True,
                'type': 'spot',
                'base': 'BTC'
            }
        }
        
        # 模拟ticker数据
        mock_exchange.fetch_ticker.side_effect = [
            {'quoteVolume': 1000000.0},  # BTC/USDT
            {'quoteVolume': 800000.0},   # ETH/USDT
        ]
        
        # 执行测试
        result = get_top_usdt_symbols(mock_exchange, limit=2)
        
        # 验证结果
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['symbol'], 'BTC/USDT')
        self.assertEqual(result[0]['volume_24h'], 1000000.0)
        self.assertEqual(result[1]['symbol'], 'ETH/USDT')
        self.assertEqual(result[1]['volume_24h'], 800000.0)
    
    def test_check_kline_data_availability(self):
        """测试K线数据可用性检查"""
        # 模拟CCXT交易所
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.return_value = self.test_kline_data
        
        # 测试数据可用的情况
        result = check_kline_data_availability(
            mock_exchange, 
            'BTC/USDT', 
            '2023-06-01'
        )
        
        # 验证结果
        self.assertTrue(result['available'])
        self.assertEqual(result['earliest_timestamp'], 1640995200000)
        self.assertEqual(result['earliest_date'], '2022-01-01')
        self.assertEqual(result['reason'], '数据可用')
    
    def test_check_kline_data_availability_future_date(self):
        """测试K线数据起始时间晚于要求日期的情况"""
        # 模拟CCXT交易所
        mock_exchange = MagicMock()
        # 模拟2024年的数据（晚于2023-06-01）
        future_kline = [[1704067200000, 47000, 48000, 46000, 47500, 1000]]  # 2024-01-01
        mock_exchange.fetch_ohlcv.return_value = future_kline
        
        # 测试数据不可用的情况
        result = check_kline_data_availability(
            mock_exchange,
            'BTC/USDT',
            '2023-06-01'
        )
        
        # 验证结果
        self.assertFalse(result['available'])
        self.assertEqual(result['earliest_date'], '2024-01-01')
        self.assertIn('晚于要求时间', result['reason'])
    
    def test_check_kline_data_availability_no_data(self):
        """测试无法获取K线数据的情况"""
        # 模拟CCXT交易所
        mock_exchange = MagicMock()
        mock_exchange.fetch_ohlcv.return_value = []
        
        # 测试无数据的情况
        result = check_kline_data_availability(
            mock_exchange,
            'BTC/USDT',
            '2023-06-01'
        )
        
        # 验证结果
        self.assertFalse(result['available'])
        self.assertIsNone(result['earliest_timestamp'])
        self.assertEqual(result['reason'], '无法获取K线数据')
    
    def test_save_symbols_to_json(self):
        """测试保存交易对到JSON文件"""
        # 创建临时目录
        test_dir = 'test_data'
        os.makedirs(test_dir, exist_ok=True)
        
        try:
            # 测试保存
            filepath = os.path.join(test_dir, 'test_symbols.json')
            save_symbols_to_json(self.test_symbols, filepath)
            
            # 验证文件存在
            self.assertTrue(os.path.exists(filepath))
            
            # 验证文件内容
            with open(filepath, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
            
            self.assertEqual(len(saved_data), 2)
            self.assertEqual(saved_data[0]['symbol'], 'BTC/USDT')
            self.assertEqual(saved_data[1]['symbol'], 'ETH/USDT')
            
        finally:
            # 清理测试文件
            if os.path.exists(filepath):
                os.remove(filepath)
            if os.path.exists(test_dir):
                os.rmdir(test_dir)
    
    @patch('data_acquisition.ccxt.binance')
    def test_fetch_top_symbols_with_validation(self, mock_binance_class):
        """测试完整的交易对获取和验证流程"""
        # 模拟Binance交易所
        mock_exchange = MagicMock()
        mock_binance_class.return_value = mock_exchange
        
        # 模拟市场数据
        mock_exchange.load_markets.return_value = {
            'BTC/USDT': {'quote': 'USDT', 'active': True, 'type': 'spot', 'base': 'BTC'},
            'ETH/USDT': {'quote': 'USDT', 'active': True, 'type': 'spot', 'base': 'ETH'},
        }
        
        # 模拟ticker数据
        mock_exchange.fetch_ticker.side_effect = [
            {'quoteVolume': 1000000.0},  # BTC/USDT
            {'quoteVolume': 800000.0},   # ETH/USDT
        ]
        
        # 模拟K线数据（2022年的数据，早于2023-06-01）
        mock_exchange.fetch_ohlcv.return_value = self.test_kline_data
        
        # 执行测试
        result = fetch_top_symbols_with_validation(
            required_start_date='2023-06-01',
            limit=2,
            max_retries=1
        )
        
        # 验证结果
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['symbol'], 'BTC/USDT')
        self.assertEqual(result[1]['symbol'], 'ETH/USDT')
        
        # 验证所有交易对都有必要字段
        for symbol_info in result:
            self.assertIn('symbol', symbol_info)
            self.assertIn('start_time', symbol_info)
            self.assertIn('start_date', symbol_info)
            self.assertIn('volume_24h', symbol_info)
    
    def test_data_format_validation(self):
        """测试数据格式验证"""
        # 测试有效的交易对数据
        valid_symbol = {
            'symbol': 'BTC/USDT',
            'start_time': 1640995200000,
            'start_date': '2022-01-01',
            'volume_24h': 1000000.0,
            'base': 'BTC',
            'quote': 'USDT'
        }
        
        # 验证必要字段
        required_fields = ['symbol', 'start_time', 'start_date', 'volume_24h']
        for field in required_fields:
            self.assertIn(field, valid_symbol)
        
        # 验证时间戳格式
        self.assertIsInstance(valid_symbol['start_time'], int)
        self.assertGreater(valid_symbol['start_time'], 0)
        
        # 验证成交量格式
        self.assertIsInstance(valid_symbol['volume_24h'], (int, float))
        self.assertGreaterEqual(valid_symbol['volume_24h'], 0)


def run_performance_test():
    """性能测试：验证代码在大量数据下的运行时间"""
    import time
    
    print("开始性能测试...")
    start_time = time.time()
    
    try:
        # 模拟获取交易对（实际会调用API）
        with patch('data_acquisition.ccxt.binance') as mock_binance_class:
            mock_exchange = MagicMock()
            mock_binance_class.return_value = mock_exchange
            
            # 模拟大量交易对数据
            mock_markets = {}
            for i in range(100):  # 模拟100个USDT交易对
                symbol = f'TOKEN{i}/USDT'
                mock_markets[symbol] = {
                    'quote': 'USDT',
                    'active': True,
                    'type': 'spot',
                    'base': f'TOKEN{i}'
                }
            
            mock_exchange.load_markets.return_value = mock_markets
            mock_exchange.fetch_ticker.return_value = {'quoteVolume': 1000000.0}
            mock_exchange.fetch_ohlcv.return_value = self.test_kline_data
            
            # 执行测试
            result = fetch_top_symbols_with_validation(limit=10, max_retries=1)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            print(f"性能测试完成:")
            print(f"- 执行时间: {execution_time:.2f} 秒")
            print(f"- 获取交易对数量: {len(result)}")
            print(f"- 平均每个交易对耗时: {execution_time/len(result):.3f} 秒")
            
            # 验证性能要求（应该在合理时间内完成）
            self.assertLess(execution_time, 60, "执行时间超过60秒，性能不达标")
            
    except Exception as e:
        print(f"性能测试失败: {e}")


if __name__ == '__main__':
    # 运行单元测试
    print("运行数据获取模块测试...")
    unittest.main(verbosity=2)
    
    # 运行性能测试
    print("\n" + "="*50)
    run_performance_test()
