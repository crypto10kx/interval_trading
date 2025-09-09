"""
ATR计算模块测试用例
测试ATR计算功能的正确性和性能
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from atr_calculator import calculate_tr, calculate_atr, process_single_symbol_atr


class TestATRCalculator(unittest.TestCase):
    """ATR计算器测试类"""
    
    def setUp(self):
        """测试前准备"""
        # 创建测试数据
        np.random.seed(42)  # 固定随机种子确保测试可重复
        self.test_data = {
            'timestamp': pd.date_range('2023-01-01', periods=200, freq='5min'),
            'open': np.random.uniform(100, 200, 200),
            'high': np.random.uniform(150, 250, 200),
            'low': np.random.uniform(50, 150, 200),
            'close': np.random.uniform(100, 200, 200),
            'volume': np.random.uniform(1000, 10000, 200)
        }
        
        # 确保high >= low
        self.test_data['high'] = np.maximum(self.test_data['high'], self.test_data['low'])
        
        self.df = pd.DataFrame(self.test_data)
    
    def test_calculate_tr_basic(self):
        """测试TR计算基本功能"""
        tr = calculate_tr(self.df)
        
        # 验证TR值非负
        self.assertTrue((tr >= 0).all(), "TR值不应为负数")
        
        # 验证TR长度
        self.assertEqual(len(tr), len(self.df), "TR长度应与输入数据相同")
        
        # 验证第一个TR值（应该为high-low，因为没有前一日收盘价）
        expected_first_tr = self.df['high'].iloc[0] - self.df['low'].iloc[0]
        self.assertAlmostEqual(tr.iloc[0], expected_first_tr, places=6, 
                             msg="第一个TR值应为high-low")
    
    def test_calculate_tr_values(self):
        """测试TR计算值正确性"""
        # 创建已知数据的测试用例
        test_df = pd.DataFrame({
            'high': [100, 105, 98, 110],
            'low': [95, 100, 95, 105],
            'close': [98, 102, 96, 108]
        })
        
        tr = calculate_tr(test_df)
        
        # 手动计算第一个有效TR值（索引1）
        # high-low = 105-100 = 5
        # |high-close_prev| = |105-98| = 7
        # |low-close_prev| = |100-98| = 2
        # max(5, 7, 2) = 7
        expected_tr_1 = 7.0
        self.assertAlmostEqual(tr.iloc[1], expected_tr_1, places=6)
    
    def test_calculate_atr_basic(self):
        """测试ATR计算基本功能"""
        df_with_atr = calculate_atr(self.df, period=20)
        
        # 验证ATR列存在
        self.assertIn('atr', df_with_atr.columns, "ATR列应存在")
        
        # 验证ATR值非负
        self.assertTrue((df_with_atr['atr'] >= 0).all(), "ATR值不应为负数")
        
        # 验证ATR长度
        self.assertEqual(len(df_with_atr), len(self.df), "ATR数据长度应与输入相同")
    
    def test_calculate_atr_period(self):
        """测试ATR计算周期"""
        df_with_atr = calculate_atr(self.df, period=10)
        
        # 验证ATR值
        atr_values = df_with_atr['atr'].dropna()
        self.assertGreater(len(atr_values), 0, "应该有有效的ATR值")
        
        # 验证ATR平滑性（应该比TR更平滑）
        tr = calculate_tr(self.df)
        tr_std = tr.std()
        atr_std = df_with_atr['atr'].std()
        self.assertLess(atr_std, tr_std, "ATR应该比TR更平滑")
    
    def test_calculate_atr_edge_cases(self):
        """测试ATR计算边界情况"""
        # 测试空DataFrame
        empty_df = pd.DataFrame(columns=['high', 'low', 'close'])
        with self.assertRaises(ValueError):
            calculate_atr(empty_df)
        
        # 测试数据长度小于周期
        short_df = self.df.head(5)
        df_with_atr = calculate_atr(short_df, period=10)
        self.assertIn('atr', df_with_atr.columns)
        
        # 测试缺少必要列
        incomplete_df = self.df[['high', 'low']].copy()
        with self.assertRaises(ValueError):
            calculate_atr(incomplete_df)
    
    def test_calculate_atr_performance(self):
        """测试ATR计算性能"""
        # 创建大数据集测试性能
        large_data = {
            'timestamp': pd.date_range('2023-01-01', periods=10000, freq='5min'),
            'open': np.random.uniform(100, 200, 10000),
            'high': np.random.uniform(150, 250, 10000),
            'low': np.random.uniform(50, 150, 10000),
            'close': np.random.uniform(100, 200, 10000),
            'volume': np.random.uniform(1000, 10000, 10000)
        }
        
        # 确保high >= low
        large_data['high'] = np.maximum(large_data['high'], large_data['low'])
        
        large_df = pd.DataFrame(large_data)
        
        # 测试计算时间（应该很快完成）
        import time
        start_time = time.time()
        df_with_atr = calculate_atr(large_df, period=120)
        end_time = time.time()
        
        calculation_time = end_time - start_time
        self.assertLess(calculation_time, 5.0, f"ATR计算时间过长: {calculation_time:.2f}秒")
        
        # 验证结果正确性
        self.assertIn('atr', df_with_atr.columns)
        self.assertEqual(len(df_with_atr), len(large_df))
    
    def test_atr_mathematical_properties(self):
        """测试ATR数学性质"""
        df_with_atr = calculate_atr(self.df, period=20)
        atr_values = df_with_atr['atr'].dropna()
        
        # ATR应该单调递增（在足够长的周期内）
        # 这里我们检查ATR的移动平均是否相对稳定
        atr_rolling_std = atr_values.rolling(window=10).std()
        self.assertLess(atr_rolling_std.mean(), atr_values.std(), 
                       "ATR的滚动标准差应该小于整体标准差")
    
    def test_process_single_symbol_atr(self):
        """测试单交易对ATR处理"""
        # 这个测试需要实际的数据文件，所以只测试函数调用
        # 在实际环境中，这个函数会处理真实的K线数据
        symbol = "TEST/USDT:USDT"
        
        # 由于没有真实的测试文件，我们期望函数返回False
        result = process_single_symbol_atr(symbol, period=120)
        self.assertFalse(result, "不存在的交易对应该返回False")


class TestATRIntegration(unittest.TestCase):
    """ATR集成测试"""
    
    def test_atr_with_real_data_structure(self):
        """测试ATR与真实数据结构的一致性"""
        # 创建模拟真实K线数据结构
        real_like_data = {
            'timestamp': pd.date_range('2023-01-01', periods=1000, freq='5min'),
            'open': np.random.uniform(1000, 2000, 1000),
            'high': np.random.uniform(1500, 2500, 1000),
            'low': np.random.uniform(500, 1500, 1000),
            'close': np.random.uniform(1000, 2000, 1000),
            'volume': np.random.uniform(10000, 100000, 1000)
        }
        
        # 确保价格逻辑正确
        real_like_data['high'] = np.maximum(real_like_data['high'], real_like_data['low'])
        real_like_data['high'] = np.maximum(real_like_data['high'], real_like_data['open'])
        real_like_data['high'] = np.maximum(real_like_data['high'], real_like_data['close'])
        real_like_data['low'] = np.minimum(real_like_data['low'], real_like_data['open'])
        real_like_data['low'] = np.minimum(real_like_data['low'], real_like_data['close'])
        
        df = pd.DataFrame(real_like_data)
        df_with_atr = calculate_atr(df, period=120)
        
        # 验证ATR值的合理性
        atr_values = df_with_atr['atr'].dropna()
        
        # ATR应该与价格水平成比例
        price_range = df['high'].max() - df['low'].min()
        atr_max = atr_values.max()
        self.assertLess(atr_max, price_range, "ATR最大值不应超过价格范围")
        
        # ATR应该为正数
        self.assertTrue((atr_values > 0).all(), "所有ATR值应为正数")


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试类
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestATRCalculator))
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestATRIntegration))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    if success:
        print("✓ 所有ATR计算测试通过")
    else:
        print("✗ 部分ATR计算测试失败")
        exit(1)
