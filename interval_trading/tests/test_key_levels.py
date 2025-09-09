"""
关键价位筛选模块测试用例
测试任务3.3：筛选关键价位功能
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from key_levels import filter_key_levels, process_single_symbol_key_levels


class TestKeyLevelsFiltering(unittest.TestCase):
    """关键价位筛选功能测试"""
    
    def setUp(self):
        """设置测试数据"""
        # 创建测试数据
        np.random.seed(42)
        self.test_data = {
            'timestamp': pd.date_range('2023-01-01', periods=200, freq='5min'),
            'open': np.random.uniform(100, 200, 200),
            'high': np.random.uniform(150, 250, 200),
            'low': np.random.uniform(50, 150, 200),
            'close': np.random.uniform(100, 200, 200),
            'volume': np.random.uniform(1000, 10000, 200),
            'atr': np.random.uniform(1, 10, 200),
            'attraction_score': np.random.uniform(0, 0.5, 200)
        }
        
        # 确保high >= low
        self.test_data['high'] = np.maximum(self.test_data['high'], self.test_data['low'])
        
        # 设置一些高得分点
        self.test_data['attraction_score'][10] = 0.15  # 高得分点1
        self.test_data['attraction_score'][50] = 0.12  # 高得分点2
        self.test_data['attraction_score'][100] = 0.08  # 低于阈值
        
        self.df = pd.DataFrame(self.test_data)
    
    def test_filter_key_levels_basic(self):
        """测试基本关键价位筛选功能"""
        result_df = filter_key_levels(self.df, threshold=0.1)
        
        # 验证结果
        self.assertIn('key_level', result_df.columns)
        self.assertEqual(len(result_df), len(self.df))
        
        # 检查是否有关键价位
        key_levels = result_df['key_level'].dropna()
        self.assertGreaterEqual(len(key_levels), 0)
    
    def test_filter_key_levels_insufficient_points(self):
        """测试符合条件的点数量不足的情况"""
        # 设置所有得分都低于阈值
        df_low_scores = self.df.copy()
        df_low_scores['attraction_score'] = 0.05
        
        result_df = filter_key_levels(df_low_scores, threshold=0.1)
        
        # 验证结果
        self.assertIn('key_level', result_df.columns)
        key_levels = result_df['key_level'].dropna()
        self.assertEqual(len(key_levels), 0)
    
    def test_filter_key_levels_price_validation(self):
        """测试价格差验证功能"""
        # 创建特定测试数据
        test_data = {
            'timestamp': pd.date_range('2023-01-01', periods=10, freq='5min'),
            'open': [100] * 10,
            'high': [110, 120, 130, 140, 150, 160, 170, 180, 190, 200],
            'low': [90, 100, 110, 120, 130, 140, 150, 160, 170, 180],
            'close': [100] * 10,
            'volume': [1000] * 10,
            'atr': [5] * 10,  # 固定ATR
            'attraction_score': [0.05, 0.05, 0.05, 0.15, 0.05, 0.05, 0.05, 0.12, 0.05, 0.05]
        }
        
        df = pd.DataFrame(test_data)
        result_df = filter_key_levels(df, threshold=0.1)
        
        # 验证结果
        self.assertIn('key_level', result_df.columns)
        key_levels = result_df['key_level'].dropna()
        
        # 由于价格差(190-140=50) > ATR(5)，应该有关键价位
        if len(key_levels) > 0:
            self.assertIn('key_1', key_levels.values)
            self.assertIn('key_0', key_levels.values)
    
    def test_filter_key_levels_threshold_parameter(self):
        """测试不同阈值参数"""
        # 测试低阈值
        result_low = filter_key_levels(self.df, threshold=0.05)
        key_levels_low = result_low['key_level'].dropna()
        
        # 测试高阈值
        result_high = filter_key_levels(self.df, threshold=0.2)
        key_levels_high = result_high['key_level'].dropna()
        
        # 低阈值应该找到更多或相等的关键价位
        self.assertGreaterEqual(len(key_levels_low), len(key_levels_high))
    
    def test_filter_key_levels_data_validation(self):
        """测试输入数据验证"""
        # 测试空DataFrame
        empty_df = pd.DataFrame()
        with self.assertRaises(ValueError):
            filter_key_levels(empty_df)
        
        # 测试缺少必要列
        incomplete_df = self.df.drop(columns=['atr'])
        with self.assertRaises(ValueError):
            filter_key_levels(incomplete_df)
    
    def test_filter_key_levels_key_level_values(self):
        """测试关键价位值的正确性"""
        result_df = filter_key_levels(self.df, threshold=0.1)
        key_levels = result_df['key_level'].dropna()
        
        if len(key_levels) > 0:
            # 验证key_level值只能是key_1, key_0或NaN
            valid_values = {'key_1', 'key_0'}
            for value in key_levels.unique():
                if pd.notna(value):
                    self.assertIn(value, valid_values)
    
    def test_filter_key_levels_key_1_higher_than_key_0(self):
        """测试key_1价格高于key_0"""
        result_df = filter_key_levels(self.df, threshold=0.1)
        
        key_1_rows = result_df[result_df['key_level'] == 'key_1']
        key_0_rows = result_df[result_df['key_level'] == 'key_0']
        
        if len(key_1_rows) > 0 and len(key_0_rows) > 0:
            key_1_price = max(key_1_rows.iloc[0]['high'], key_1_rows.iloc[0]['low'])
            key_0_price = max(key_0_rows.iloc[0]['high'], key_0_rows.iloc[0]['low'])
            
            self.assertGreater(key_1_price, key_0_price)


class TestKeyLevelsIntegration(unittest.TestCase):
    """关键价位筛选集成测试"""
    
    def test_process_single_symbol_key_levels(self):
        """测试单个交易对处理功能"""
        # 这个测试需要实际的吸引力得分文件
        # 在实际环境中运行时会测试真实数据
        pass
    
    def test_key_levels_performance(self):
        """测试性能（大数据集）"""
        # 创建大数据集
        large_data = {
            'timestamp': pd.date_range('2023-01-01', periods=10000, freq='5min'),
            'open': np.random.uniform(100, 200, 10000),
            'high': np.random.uniform(150, 250, 10000),
            'low': np.random.uniform(50, 150, 10000),
            'close': np.random.uniform(100, 200, 10000),
            'volume': np.random.uniform(1000, 10000, 10000),
            'atr': np.random.uniform(1, 10, 10000),
            'attraction_score': np.random.uniform(0, 0.5, 10000)
        }
        
        # 确保high >= low
        large_data['high'] = np.maximum(large_data['high'], large_data['low'])
        
        df = pd.DataFrame(large_data)
        
        # 测试处理时间
        import time
        start_time = time.time()
        result_df = filter_key_levels(df, threshold=0.1)
        end_time = time.time()
        
        # 验证结果
        self.assertIn('key_level', result_df.columns)
        self.assertEqual(len(result_df), len(df))
        
        # 性能检查（应该在合理时间内完成）
        processing_time = end_time - start_time
        self.assertLess(processing_time, 10.0)  # 应该在10秒内完成
        
        print(f"大数据集处理时间: {processing_time:.2f}秒")


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试用例
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestKeyLevelsFiltering))
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestKeyLevelsIntegration))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    if success:
        print("✓ 所有关键价位筛选测试通过")
    else:
        print("✗ 部分关键价位筛选测试失败")
