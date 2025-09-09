"""
吸引力得分计算模块测试用例
测试吸引力得分计算功能的正确性和性能
"""

import unittest
import pandas as pd
import numpy as np
import sys
import os

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from attraction_score import calculate_attraction_score, _calculate_point_attraction_score


class TestAttractionScore(unittest.TestCase):
    """吸引力得分计算器测试类"""
    
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
            'volume': np.random.uniform(1000, 10000, 200),
            'atr': np.random.uniform(1, 10, 200)
        }
        
        # 确保high >= low
        self.test_data['high'] = np.maximum(self.test_data['high'], self.test_data['low'])
        
        self.df = pd.DataFrame(self.test_data)
    
    def test_calculate_attraction_score_basic(self):
        """测试吸引力得分计算基本功能"""
        df_with_attraction = calculate_attraction_score(self.df, w=20, band_width=0.1)
        
        # 验证吸引力得分列存在
        self.assertIn('attraction_score', df_with_attraction.columns, "吸引力得分列应存在")
        
        # 验证吸引力得分值范围[0,1]
        attraction_scores = df_with_attraction['attraction_score']
        self.assertTrue((attraction_scores >= 0).all(), "吸引力得分不应为负数")
        self.assertTrue((attraction_scores <= 1).all(), "吸引力得分不应超过1")
        
        # 验证吸引力得分长度
        self.assertEqual(len(df_with_attraction), len(self.df), "吸引力得分数据长度应与输入相同")
    
    def test_calculate_attraction_score_edge_cases(self):
        """测试吸引力得分计算边界情况"""
        # 测试空DataFrame
        empty_df = pd.DataFrame(columns=['high', 'low', 'atr'])
        with self.assertRaises(ValueError):
            calculate_attraction_score(empty_df)
        
        # 测试数据长度小于窗口大小
        short_df = self.df.head(5)
        df_with_attraction = calculate_attraction_score(short_df, w=20)
        self.assertIn('attraction_score', df_with_attraction.columns)
        
        # 测试缺少必要列
        incomplete_df = self.df[['high', 'low']].copy()
        with self.assertRaises(ValueError):
            calculate_attraction_score(incomplete_df)
    
    def test_calculate_attraction_score_parameters(self):
        """测试吸引力得分计算参数"""
        # 测试不同窗口大小
        df_with_attraction_10 = calculate_attraction_score(self.df, w=10, band_width=0.1)
        df_with_attraction_50 = calculate_attraction_score(self.df, w=50, band_width=0.1)
        
        # 验证结果不同
        self.assertFalse(
            np.array_equal(df_with_attraction_10['attraction_score'], 
                          df_with_attraction_50['attraction_score']),
            "不同窗口大小应产生不同结果"
        )
        
        # 测试不同带宽系数
        df_with_attraction_01 = calculate_attraction_score(self.df, w=20, band_width=0.1)
        df_with_attraction_05 = calculate_attraction_score(self.df, w=20, band_width=0.5)
        
        # 验证结果不同
        self.assertFalse(
            np.array_equal(df_with_attraction_01['attraction_score'], 
                          df_with_attraction_05['attraction_score']),
            "不同带宽系数应产生不同结果"
        )
    
    def test_calculate_point_attraction_score(self):
        """测试单点吸引力得分计算"""
        # 创建测试数据
        window_high = np.array([100, 105, 98, 110])
        window_low = np.array([95, 100, 95, 105])
        window_atr = np.array([2, 3, 1.5, 2.5])
        target_price = 100
        band_width = 0.1
        
        score = _calculate_point_attraction_score(
            target_price, window_high, window_low, window_atr, band_width
        )
        
        # 验证得分范围
        self.assertGreaterEqual(score, 0, "吸引力得分不应为负数")
        self.assertLessEqual(score, 1, "吸引力得分不应超过1")
    
    def test_attraction_score_mathematical_properties(self):
        """测试吸引力得分数学性质"""
        df_with_attraction = calculate_attraction_score(self.df, w=20, band_width=0.1)
        attraction_scores = df_with_attraction['attraction_score']
        
        # 吸引力得分应该相对稳定（在相似价格水平下）
        # 这里我们检查得分的分布是否合理
        score_std = attraction_scores.std()
        self.assertLess(score_std, 0.5, "吸引力得分标准差不应过大")
        
        # 大部分得分应该较低（因为价格相近的概率较低，且除以w后得分被稀释）
        low_scores = (attraction_scores < 0.1).sum()
        self.assertGreater(low_scores, len(attraction_scores) * 0.5, 
                          "大部分吸引力得分应该较低")
    
    def test_attraction_score_performance(self):
        """测试吸引力得分计算性能"""
        # 创建大数据集测试性能
        large_data = {
            'timestamp': pd.date_range('2023-01-01', periods=5000, freq='5min'),
            'open': np.random.uniform(100, 200, 5000),
            'high': np.random.uniform(150, 250, 5000),
            'low': np.random.uniform(50, 150, 5000),
            'close': np.random.uniform(100, 200, 5000),
            'volume': np.random.uniform(1000, 10000, 5000),
            'atr': np.random.uniform(1, 10, 5000)
        }
        
        # 确保high >= low
        large_data['high'] = np.maximum(large_data['high'], large_data['low'])
        
        large_df = pd.DataFrame(large_data)
        
        # 测试计算时间（应该合理完成）
        import time
        start_time = time.time()
        df_with_attraction = calculate_attraction_score(large_df, w=120, band_width=0.1)
        end_time = time.time()
        
        calculation_time = end_time - start_time
        self.assertLess(calculation_time, 30.0, f"吸引力得分计算时间过长: {calculation_time:.2f}秒")
        
        # 验证结果正确性
        self.assertIn('attraction_score', df_with_attraction.columns)
        self.assertEqual(len(df_with_attraction), len(large_df))
    
    def test_attraction_score_with_zero_atr(self):
        """测试ATR为0时的处理"""
        # 创建包含ATR为0的测试数据
        test_data = {
            'high': [100, 105, 98, 110],
            'low': [95, 100, 95, 105],
            'atr': [0, 2, 0, 3]  # 包含ATR为0的情况
        }
        
        df = pd.DataFrame(test_data)
        df_with_attraction = calculate_attraction_score(df, w=4, band_width=0.1)
        
        # 验证没有NaN值
        self.assertFalse(df_with_attraction['attraction_score'].isna().any(), 
                        "ATR为0时不应产生NaN值")
        
        # 验证得分范围
        attraction_scores = df_with_attraction['attraction_score']
        self.assertTrue((attraction_scores >= 0).all(), "吸引力得分不应为负数")
        self.assertTrue((attraction_scores <= 1).all(), "吸引力得分不应超过1")
    
    def test_attraction_score_with_nan_atr(self):
        """测试ATR为NaN时的处理"""
        # 创建包含ATR为NaN的测试数据
        test_data = {
            'high': [100, 105, 98, 110],
            'low': [95, 100, 95, 105],
            'atr': [np.nan, 2, np.nan, 3]  # 包含ATR为NaN的情况
        }
        
        df = pd.DataFrame(test_data)
        df_with_attraction = calculate_attraction_score(df, w=4, band_width=0.1)
        
        # 验证没有NaN值
        self.assertFalse(df_with_attraction['attraction_score'].isna().any(), 
                        "ATR为NaN时不应产生NaN值")
        
        # 验证得分范围
        attraction_scores = df_with_attraction['attraction_score']
        self.assertTrue((attraction_scores >= 0).all(), "吸引力得分不应为负数")
        self.assertTrue((attraction_scores <= 1).all(), "吸引力得分不应超过1")


class TestAttractionScoreIntegration(unittest.TestCase):
    """吸引力得分集成测试"""
    
    def test_attraction_score_with_real_data_structure(self):
        """测试吸引力得分与真实数据结构的一致性"""
        # 创建模拟真实K线数据结构
        real_like_data = {
            'timestamp': pd.date_range('2023-01-01', periods=1000, freq='5min'),
            'open': np.random.uniform(1000, 2000, 1000),
            'high': np.random.uniform(1500, 2500, 1000),
            'low': np.random.uniform(500, 1500, 1000),
            'close': np.random.uniform(1000, 2000, 1000),
            'volume': np.random.uniform(10000, 100000, 1000),
            'atr': np.random.uniform(10, 100, 1000)
        }
        
        # 确保价格逻辑正确
        real_like_data['high'] = np.maximum(real_like_data['high'], real_like_data['low'])
        real_like_data['high'] = np.maximum(real_like_data['high'], real_like_data['open'])
        real_like_data['high'] = np.maximum(real_like_data['high'], real_like_data['close'])
        real_like_data['low'] = np.minimum(real_like_data['low'], real_like_data['open'])
        real_like_data['low'] = np.minimum(real_like_data['low'], real_like_data['close'])
        
        df = pd.DataFrame(real_like_data)
        df_with_attraction = calculate_attraction_score(df, w=120, band_width=0.1)
        
        # 验证吸引力得分值的合理性
        attraction_scores = df_with_attraction['attraction_score']
        
        # 吸引力得分应该非负
        self.assertTrue((attraction_scores >= 0).all(), "所有吸引力得分应为非负数")
        
        # 吸引力得分应该在合理范围内
        self.assertTrue((attraction_scores <= 1).all(), "所有吸引力得分不应超过1")
        
        # 大部分得分应该较低（因为价格相近的概率较低，且除以w后得分被稀释）
        low_scores = (attraction_scores < 0.1).sum()
        self.assertGreater(low_scores, len(attraction_scores) * 0.3, 
                          "相当比例的吸引力得分应该较低")


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    test_suite = unittest.TestSuite()
    
    # 添加测试类
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestAttractionScore))
    test_suite.addTest(unittest.TestLoader().loadTestsFromTestCase(TestAttractionScoreIntegration))
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    if success:
        print("✓ 所有吸引力得分计算测试通过")
    else:
        print("✗ 部分吸引力得分计算测试失败")
        exit(1)
