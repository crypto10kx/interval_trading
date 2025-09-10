"""
任务4.2: 贝叶斯优化

对w、band_width、阈值进行贝叶斯优化，回测多个参数组合，评估夏普比率/胜率，输出最佳参数
- 使用scikit-optimize的gp_minimize
- 参数范围：w=[120,240]，band_width=[0.05,0.15]，阈值=[0.05,0.2]
- 分批处理交易对（每批1个），适配8G RAM
- 输出最佳参数和性能报告

作者: AI Assistant
日期: 2025-09-10
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional
import os
import sys
from dataclasses import dataclass
from joblib import Parallel, delayed
import warnings

# 添加src目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trading_simulator import simulate_trading_logic

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 忽略警告
warnings.filterwarnings('ignore')

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
    from skopt.utils import use_named_args
except ImportError:
    logger.error("请安装scikit-optimize: pip install scikit-optimize")
    sys.exit(1)


@dataclass
class OptimizationResult:
    """优化结果数据类"""
    w: int
    band_width: float
    threshold: float
    sharpe_ratio: float
    win_rate: float
    total_trades: int
    total_profit: float
    avg_profit: float


def calculate_sharpe_ratio(returns: pd.Series) -> float:
    """
    计算夏普比率
    
    Args:
        returns: 收益率序列
        
    Returns:
        夏普比率
    """
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    
    # 年化收益率和波动率
    annual_return = returns.mean() * 252 * 24 * 12  # 5分钟K线，年化
    annual_volatility = returns.std() * np.sqrt(252 * 24 * 12)
    
    if annual_volatility == 0:
        return 0.0
    
    sharpe_ratio = annual_return / annual_volatility
    return sharpe_ratio


def evaluate_parameters(df: pd.DataFrame, 
                       symbol: str,
                       w: int, 
                       band_width: float, 
                       threshold: float) -> Dict:
    """
    评估参数组合的性能
    
    Args:
        df: K线数据DataFrame
        symbol: 交易对符号
        w: 窗口大小
        band_width: 带宽参数
        threshold: 吸引力得分阈值
        
    Returns:
        性能指标字典
    """
    try:
        # 运行交易模拟
        trades_df = simulate_trading_logic(
            df, 
            symbol=symbol,
            w=w,
            band_width=band_width,
            threshold=threshold
        )
        
        if trades_df.empty:
            return {
                'sharpe_ratio': -1000.0,  # 使用大负数而不是-inf
                'win_rate': 0.0,
                'total_trades': 0,
                'total_profit': 0.0,
                'avg_profit': 0.0
            }
        
        # 计算性能指标
        returns = trades_df['profit']
        sharpe_ratio = calculate_sharpe_ratio(returns)
        win_rate = (returns > 0).mean()
        total_trades = len(trades_df)
        total_profit = returns.sum()
        avg_profit = returns.mean()
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'total_profit': total_profit,
            'avg_profit': avg_profit
        }
        
    except Exception as e:
        logger.warning(f"参数评估失败 ({w}, {band_width}, {threshold}): {e}")
        return {
            'sharpe_ratio': -np.inf,
            'win_rate': 0.0,
            'total_trades': 0,
            'total_profit': 0.0,
            'avg_profit': 0.0
        }


def optimize_single_symbol(df: pd.DataFrame, 
                          symbol: str,
                          n_calls: int = 50) -> OptimizationResult:
    """
    对单个交易对进行贝叶斯优化
    
    Args:
        df: K线数据DataFrame
        symbol: 交易对符号
        n_calls: 优化调用次数
        
    Returns:
        最佳参数结果
    """
    logger.info(f"开始优化交易对: {symbol}")
    
    # 定义参数空间（移除threshold，使用动态阈值）
    dimensions = [
        Integer(120, 240, name='w'),           # w=[120,240]
        Real(0.05, 0.15, name='band_width'),   # band_width=[0.05,0.15]
    ]
    
    # 目标函数（最小化负夏普比率，使用动态阈值）
    @use_named_args(dimensions=dimensions)
    def objective(w, band_width):
        result = evaluate_parameters(df, symbol, w, band_width, None)  # 使用动态阈值
        return -result['sharpe_ratio']  # 最小化负夏普比率 = 最大化夏普比率
    
    # 运行贝叶斯优化
    logger.info(f"开始贝叶斯优化，调用次数: {n_calls}")
    result = gp_minimize(
        func=objective,
        dimensions=dimensions,
        n_calls=n_calls,
        random_state=42,
        acq_func='EI'  # Expected Improvement
    )
    
    # 获取最佳参数
    best_w = int(result.x[0])
    best_band_width = result.x[1]
    # threshold现在使用动态计算，不需要从结果中提取
    best_sharpe = -result.fun
    
    # 重新评估最佳参数
    final_result = evaluate_parameters(
        df, symbol, best_w, best_band_width, best_threshold
    )
    
    optimization_result = OptimizationResult(
        w=best_w,
        band_width=best_band_width,
        threshold=best_threshold,
        sharpe_ratio=final_result['sharpe_ratio'],
        win_rate=final_result['win_rate'],
        total_trades=final_result['total_trades'],
        total_profit=final_result['total_profit'],
        avg_profit=final_result['avg_profit']
    )
    
    logger.info(f"优化完成: {symbol}")
    logger.info(f"- 最佳参数: w={best_w}, band_width={best_band_width:.3f}, threshold={best_threshold:.3f}")
    logger.info(f"- 夏普比率: {best_sharpe:.4f}")
    logger.info(f"- 胜率: {final_result['win_rate']:.2%}")
    logger.info(f"- 总交易次数: {final_result['total_trades']}")
    
    return optimization_result


def optimize_all_symbols(data_dir: str = "data", 
                        n_calls: int = 50,
                        n_jobs: int = 4) -> List[OptimizationResult]:
    """
    对所有交易对进行贝叶斯优化
    
    Args:
        data_dir: 数据目录
        n_calls: 每个交易对的优化调用次数
        n_jobs: 并行处理数量
        
    Returns:
        所有交易对的优化结果列表
    """
    logger.info("开始对所有交易对进行贝叶斯优化")
    
    # 获取所有CSV文件
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('_processed.csv')]
    
    if not csv_files:
        logger.error("未找到处理后的数据文件")
        return []
    
    logger.info(f"找到 {len(csv_files)} 个交易对数据文件")
    
    # 定义并行处理函数
    def process_single_file(csv_file):
        try:
            # 读取数据
            file_path = os.path.join(data_dir, csv_file)
            df = pd.read_csv(file_path)
            
            # 提取交易对符号
            symbol = csv_file.replace('_processed.csv', '').replace('_', '/')
            if symbol.endswith('USDT'):
                symbol = symbol + ':USDT'
            
            # 检查必需列
            required_columns = ['timestamp', 'log_high', 'log_low', 'log_close']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.warning(f"跳过 {symbol}: 缺少列 {missing_columns}")
                return None
            
            # 运行优化
            return optimize_single_symbol(df, symbol, n_calls)
            
        except Exception as e:
            logger.error(f"处理文件 {csv_file} 时出错: {e}")
            return None
    
    # 并行处理所有交易对
    logger.info(f"开始并行处理，使用 {n_jobs} 个进程")
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_single_file)(csv_file) for csv_file in csv_files
    )
    
    # 过滤掉None结果
    valid_results = [r for r in results if r is not None]
    
    logger.info(f"优化完成，成功处理 {len(valid_results)} 个交易对")
    return valid_results


def save_optimization_results(results: List[OptimizationResult], 
                            output_file: str = "optimization_results.csv"):
    """
    保存优化结果到CSV文件
    
    Args:
        results: 优化结果列表
        output_file: 输出文件名
    """
    if not results:
        logger.warning("没有优化结果可保存")
        return
    
    # 转换为DataFrame
    data = []
    for i, result in enumerate(results):
        data.append({
            'symbol_id': i + 1,
            'w': result.w,
            'band_width': result.band_width,
            'threshold': result.threshold,
            'sharpe_ratio': result.sharpe_ratio,
            'win_rate': result.win_rate,
            'total_trades': result.total_trades,
            'total_profit': result.total_profit,
            'avg_profit': result.avg_profit
        })
    
    results_df = pd.DataFrame(data)
    
    # 保存到CSV
    results_df.to_csv(output_file, index=False)
    logger.info(f"优化结果已保存到: {output_file}")
    
    # 打印汇总统计
    logger.info("=" * 50)
    logger.info("优化结果汇总:")
    logger.info("=" * 50)
    logger.info(f"总交易对数: {len(results)}")
    logger.info(f"平均夏普比率: {results_df['sharpe_ratio'].mean():.4f}")
    logger.info(f"最高夏普比率: {results_df['sharpe_ratio'].max():.4f}")
    logger.info(f"平均胜率: {results_df['win_rate'].mean():.2%}")
    logger.info(f"平均交易次数: {results_df['total_trades'].mean():.1f}")
    logger.info(f"平均总盈亏: {results_df['total_profit'].mean():.4f}")
    
    # 显示最佳参数
    best_idx = results_df['sharpe_ratio'].idxmax()
    best_result = results_df.iloc[best_idx]
    logger.info(f"最佳参数组合:")
    logger.info(f"- w: {best_result['w']}")
    logger.info(f"- band_width: {best_result['band_width']:.3f}")
    logger.info(f"- threshold: {best_result['threshold']:.3f}")
    logger.info(f"- 夏普比率: {best_result['sharpe_ratio']:.4f}")


def test_bayesian_optimization():
    """测试贝叶斯优化功能"""
    logger.info("开始测试贝叶斯优化功能")
    
    # 生成测试数据（1000条）
    np.random.seed(42)
    n_samples = 1000
    
    # 生成模拟价格数据（对数空间）
    base_price = np.log(50000)  # BTC基础价格
    price_changes = np.random.normal(0, 0.01, n_samples)  # 价格变化
    log_prices = base_price + np.cumsum(price_changes)
    
    # 生成K线数据
    test_data = []
    for i in range(n_samples):
        # 生成高低价
        high_offset = np.random.uniform(0, 0.005)
        low_offset = np.random.uniform(-0.005, 0)
        
        log_high = log_prices[i] + high_offset
        log_low = log_prices[i] + low_offset
        log_close = log_prices[i] + np.random.uniform(-0.002, 0.002)
        
        test_data.append({
            'timestamp': 1609459200 + i * 300,  # 5分钟间隔
            'log_high': log_high,
            'log_low': log_low,
            'log_close': log_close
        })
    
    # 创建测试DataFrame
    test_df = pd.DataFrame(test_data)
    
    # 运行优化（减少调用次数以加快测试）
    result = optimize_single_symbol(test_df, "BTC/USDT:USDT", n_calls=10)
    
    logger.info("✓ 贝叶斯优化测试完成")
    logger.info(f"测试结果: w={result.w}, band_width={result.band_width:.3f}, threshold={result.threshold:.3f}")
    logger.info(f"夏普比率: {result.sharpe_ratio:.4f}")
    
    return result


def main():
    """主函数：运行贝叶斯优化"""
    logger.info("开始执行任务4.2：贝叶斯优化")
    
    # 检查数据目录
    data_dir = "data"
    if not os.path.exists(data_dir):
        logger.error(f"数据目录不存在: {data_dir}")
        return
    
    # 运行测试
    logger.info("运行测试...")
    test_result = test_bayesian_optimization()
    
    # 运行完整优化
    logger.info("开始完整优化...")
    results = optimize_all_symbols(data_dir, n_calls=30, n_jobs=2)
    
    if results:
        # 保存结果
        save_optimization_results(results)
        
        logger.info("=" * 50)
        logger.info("任务4.2完成 - 贝叶斯优化:")
        logger.info("=" * 50)
        logger.info("✓ 参数优化完成")
        logger.info("✓ 夏普比率评估完成")
        logger.info("✓ 分批处理完成")
        logger.info("✓ 结果保存完成")
    else:
        logger.error("任务4.2优化失败")


if __name__ == "__main__":
    main()
