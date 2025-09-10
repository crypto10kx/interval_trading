
# 区间交易系统 Range 分析报告

## 分析参数
- **观察窗口 (w)**: 120
- **带宽参数 (band_width)**: 0.1
- **阈值 (threshold)**: 0.075
- **分析时间**: 2025-09-10 23:42:01

## 总体统计
- **交易对数量**: 15
- **总Range数量**: 11
- **平均Range数量**: 0.73

## 详细结果

| 排名 | 交易对 | Range数量 | 价格范围 | 数据质量 | 扩展价位有效性 |
|------|--------|-----------|----------|----------|----------------|
| 1 | NEAR_USDT_USDT | 1 | 8.07 | good | log_level_-2: 0, log_level_-1: 0, log_level_2: 0, log_level_3: 0 |
| 2 | SOL_USDT_USDT | 1 | 254.96 | good | log_level_-2: 0, log_level_-1: 0, log_level_2: 0, log_level_3: 0 |
| 3 | BTC_USDT_USDT | 1 | 69821.80 | good | log_level_-2: 0, log_level_-1: 0, log_level_2: 0, log_level_3: 0 |
| 4 | LTC_USDT_USDT | 1 | 98.08 | good | log_level_-2: 0, log_level_-1: 0, log_level_2: 0, log_level_3: 0 |
| 5 | XRP_USDT_USDT | 1 | 2.53 | good | log_level_-2: 0, log_level_-1: 0, log_level_2: 0, log_level_3: 0 |
| 6 | BNB_USDT_USDT | 1 | 508.62 | good | log_level_-2: 0, log_level_-1: 0, log_level_2: 0, log_level_3: 0 |
| 7 | AVAX_USDT_USDT | 1 | 56.87 | good | log_level_-2: 0, log_level_-1: 0, log_level_2: 0, log_level_3: 0 |
| 8 | ETH_USDT_USDT | 1 | 2042.83 | good | log_level_-2: 0, log_level_-1: 0, log_level_2: 0, log_level_3: 0 |
| 9 | DOT_USDT_USDT | 1 | 8.36 | good | log_level_-2: 0, log_level_-1: 0, log_level_2: 0, log_level_3: 0 |
| 10 | ATOM_USDT_USDT | 1 | 11.86 | good | log_level_-2: 0, log_level_-1: 0, log_level_2: 0, log_level_3: 0 |
| 11 | UNI_USDT_USDT | 1 | 15.95 | good | log_level_-2: 0, log_level_-1: 0, log_level_2: 0, log_level_3: 0 |
| 12 | ADA_USDT_USDT | 0 | 1.05 | poor | log_level_-2: 0, log_level_-1: 0, log_level_2: 0, log_level_3: 0 |
| 13 | BCH_USDT_USDT | 0 | 630.32 | poor | log_level_-2: 0, log_level_-1: 0, log_level_2: 0, log_level_3: 0 |
| 14 | LINK_USDT_USDT | 0 | 26.26 | poor | log_level_-2: 0, log_level_-1: 0, log_level_2: 0, log_level_3: 0 |
| 15 | DOGE_USDT_USDT | 0 | 0.43 | poor | log_level_-2: 0, log_level_-1: 0, log_level_2: 0, log_level_3: 0 |

## 统计摘要
- **最大Range数量**: 1
- **最小Range数量**: 0
- **Range数量中位数**: 1.00
- **Range数量标准差**: 0.44

## 数据质量分布
- **优秀 (Range > 10)**: 0 个交易对
- **良好 (Range 5-10)**: 0 个交易对
- **一般 (Range 1-4)**: 11 个交易对
- **较差 (Range = 0)**: 4 个交易对

## 建议
1. 重点关注Range数量较多的交易对进行进一步分析
2. 对于Range数量为0的交易对，需要检查数据质量或调整参数
3. 建议对Range数量>5的交易对进行手动验证
