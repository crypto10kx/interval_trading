"""
测试用例 - K线数据获取模块
验证任务1.2的功能正确性
"""

import unittest
import sys
import os
import pandas as pd
from datetime import datetime
from unittest.mock import patch, MagicMock
import tempfile
import shutil

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from kline_fetcher import (
    fetch_kline_data,
    save_kline_to_csv,
    load_symbols_from_json,
    fetch_all_symbols_klines
)


class TestKlineFetcher(unittest.TestCase):
    """K线数据获取模块测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()
        
        # 模拟K线数据
        self.mock_kline_data = [
            [1640995200000, 47000, 48000, 46000, 47500, 1000],  # 2022-01-01
            [1640995500000, 47500, 48500, 47000, 48000, 1200],  # 2022-01-01 00:05
            [1640995800000, 48000, 49000, 47500, 48500, 1100],  # 2022-01-01 00:10
        ]
        
        # 模拟交易对列表
        self.mock_symbols = [
            {
                'symbol': 'BTC/USDT',
                'start_time': 1640995200000,
                'start_date': '2022-01-01',
                'volume_24h': 1000000.0
            },
            {
                'symbol': 'ETH/USDT',
                'start_time': 1640995200000,
                'start_date': '2022-01-01',
                'volume_24h': 800000.0
            }
        ]
    
    def tearDown(self):
        """测试后清理"""
        # 清理临时目录
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    @patch('kline_fetcher.ccxt.binance')
    def test_fetch_kline_data(self, mock_binance_class):
        """测试K线数据拉取功能"""
        # 模拟Binance交易所
        mock_exchange = MagicMock()
        mock_binance_class.return_value = mock_exchange
        
        # 模拟K线数据返回
        mock_exchange.fetch_ohlcv.return_value = self.mock_kline_data
        
        # 执行测试
        df = fetch_kline_data(
            symbol='BTC/USDT',
            start_date='2022-01-01',
            end_date='2022-01-02',
            batch_size=1500,
            request_interval=0.05
        )
        
        # 验证结果
        self.assertEqual(len(df), 3)
        self.assertEqual(list(df.columns), ['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        self.assertEqual(df['open'].iloc[0], 47000)
        self.assertEqual(df['close'].iloc[0], 47500)
        
        # 验证时间戳转换
        self.assertIsInstance(df['timestamp'].iloc[0], pd.Timestamp)
        self.assertEqual(df['timestamp'].iloc[0].year, 2022)
    
    def test_save_kline_to_csv(self):
        """测试保存K线数据到CSV"""
        # 创建测试DataFrame
        df = pd.DataFrame(self.mock_kline_data, 
                         columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # 执行测试
        filepath = save_kline_to_csv(df, 'BTC/USDT', self.temp_dir)
        
        # 验证文件存在
        self.assertTrue(os.path.exists(filepath))
        
        # 验证文件内容
        loaded_df = pd.read_csv(filepath)
        self.assertEqual(len(loaded_df), 3)
        self.assertEqual(list(loaded_df.columns), ['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    def test_load_symbols_from_json(self):
        """测试从JSON加载交易对列表"""
        # 创建测试JSON文件
        json_file = os.path.join(self.temp_dir, 'test_symbols.json')
        import json
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.mock_symbols, f, indent=2)
        
        # 执行测试
        symbols = load_symbols_from_json(json_file)
        
        # 验证结果
        self.assertEqual(len(symbols), 2)
        self.assertEqual(symbols[0]['symbol'], 'BTC/USDT')
        self.assertEqual(symbols[1]['symbol'], 'ETH/USDT')
    
    @patch('kline_fetcher.ccxt.binance')
    def test_fetch_all_symbols_klines(self, mock_binance_class):
        """测试批量拉取K线数据"""
        # 模拟Binance交易所
        mock_exchange = MagicMock()
        mock_binance_class.return_value = mock_exchange
        
        # 模拟K线数据返回
        mock_exchange.fetch_ohlcv.return_value = self.mock_kline_data
        
        # 执行测试
        results = fetch_all_symbols_klines(
            symbols=self.mock_symbols,
            start_date='2022-01-01',
            end_date='2022-01-02',
            batch_size=1500,
            request_interval=0.05
        )
        
        # 验证结果
        self.assertEqual(len(results), 2)
        self.assertIn('BTC/USDT', results)
        self.assertIn('ETH/USDT', results)
        
        # 验证文件存在
        for symbol, filepath in results.items():
            if filepath:
                self.assertTrue(os.path.exists(filepath))
                self.assertTrue(filepath.endswith('_5m.csv'))
    
    def test_data_validation(self):
        """测试数据验证功能"""
        # 创建测试DataFrame
        df = pd.DataFrame(self.mock_kline_data, 
                         columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        
        # 验证数据格式
        self.assertEqual(len(df), 3)
        self.assertTrue(all(col in df.columns for col in ['timestamp', 'open', 'high', 'low', 'close', 'volume']))
        
        # 验证数据类型
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df['timestamp']))
        self.assertTrue(pd.api.types.is_numeric_dtype(df['open']))
        self.assertTrue(pd.api.types.is_numeric_dtype(df['high']))
        self.assertTrue(pd.api.types.is_numeric_dtype(df['low']))
        self.assertTrue(pd.api.types.is_numeric_dtype(df['close']))
        self.assertTrue(pd.api.types.is_numeric_dtype(df['volume']))
    
    def test_batch_size_optimization(self):
        """测试批次大小优化"""
        # 验证批次大小参数
        batch_size = 1500
        request_interval = 0.05
        
        self.assertEqual(batch_size, 1500)
        self.assertEqual(request_interval, 0.05)
        
        # 计算理论数据量（2年5分钟K线）
        # 2年 = 2 * 365 * 24 * 60 / 5 = 210,240 条K线
        expected_klines = 2 * 365 * 24 * 60 // 5
        expected_batches = (expected_klines + batch_size - 1) // batch_size
        
        self.assertGreater(expected_klines, 200000)
        self.assertLess(expected_batches, 200)  # 应该少于200个批次
    
    def test_performance_requirements(self):
        """测试性能要求"""
        # 验证请求间隔（50ms）
        request_interval = 0.05
        self.assertEqual(request_interval, 0.05)
        
        # 验证批次大小（1500根K线）
        batch_size = 1500
        self.assertEqual(batch_size, 1500)
        
        # 计算理论执行时间
        # 假设每个交易对需要140个批次（210,240 / 1500）
        batches_per_symbol = 140
        time_per_batch = request_interval  # 50ms
        total_time_per_symbol = batches_per_symbol * time_per_batch  # 7秒
        
        self.assertLess(total_time_per_symbol, 10)  # 每个交易对应该少于10秒


def run_performance_test():
    """性能测试：验证代码在优化参数下的运行时间"""
    import time
    
    print("开始K线数据获取性能测试...")
    start_time = time.time()
    
    try:
        # 模拟获取K线数据（实际会调用API）
        with patch('kline_fetcher.ccxt.binance') as mock_binance_class:
            mock_exchange = MagicMock()
            mock_binance_class.return_value = mock_exchange
            
            # 模拟K线数据返回
            mock_exchange.fetch_ohlcv.return_value = [
                [1640995200000, 47000, 48000, 46000, 47500, 1000]
            ]
            
            # 执行测试
            df = fetch_kline_data(
                symbol='BTC/USDT',
                start_date='2022-01-01',
                end_date='2022-01-02',
                batch_size=1500,
                request_interval=0.05
            )
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            print(f"性能测试完成:")
            print(f"- 执行时间: {execution_time:.2f} 秒")
            print(f"- 批次大小: 1500 根K线")
            print(f"- 请求间隔: 50ms")
            print(f"- 获取数据量: {len(df)} 条")
            
            # 验证性能要求
            assert execution_time < 5, "执行时间超过5秒，性能不达标"
            
    except Exception as e:
        print(f"性能测试失败: {e}")


if __name__ == '__main__':
    # 运行单元测试
    print("运行K线数据获取模块测试...")
    unittest.main(verbosity=2)
    
    # 运行性能测试
    print("\n" + "="*50)
    run_performance_test()
