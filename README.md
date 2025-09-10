# 区间交易系统 2.0

基于自创分析框架的加密货币区间交易系统，通过峰谷识别、关键价位计算和动态交易策略进行区间交易。

## 项目结构

```
range-trader/
├── src/                          # 核心模块
│   ├── data_acquisition.py       # 数据获取模块
│   ├── kline_fetcher.py          # K线数据获取
│   ├── log_transformer.py        # 对数价格变换
│   ├── peak_valley.py            # 峰谷识别算法
│   ├── atr_calculator.py         # ATR计算
│   ├── attraction_score.py       # 吸引力得分计算
│   ├── key_levels.py             # 关键价位筛选
│   ├── extended_levels.py        # 扩展价位计算
│   ├── trading_simulator.py      # 交易策略模拟
│   ├── bayesian_optimizer.py     # 贝叶斯优化
│   └── task_4_2_*.py            # 任务4.2相关模块
├── data/                         # 数据文件
│   ├── *_5m.csv                 # 原始K线数据
│   └── *_5m_processed.csv       # 处理后数据
├── tests/                        # 测试文件
│   ├── test_*.py                # 各模块单元测试
│   └── ...
├── scripts/                      # 脚本文件
│   ├── analysis/                # 分析脚本
│   │   ├── calculate_ranges_dynamic.py
│   │   ├── analyze_ranges.py
│   │   └── analyze_log_impact.py
│   ├── optimization/            # 优化脚本
│   │   └── test_bayesian_simple.py
│   └── utilities/               # 工具脚本
│       └── batch_process_modules.py
├── docs/                        # 文档
│   ├── range_analysis_report.md
│   └── manual_validation_report.md
├── pyproject.toml               # 项目配置
├── poetry.lock                  # 依赖锁定
├── 项目描述2.0.md              # 项目详细描述
└── README.md                    # 本文件
```

## 核心功能

### 1. 数据获取与预处理
- 获取Binance永续合约前15个USDT交易对的2年5分钟K线数据
- 对所有价格数据进行对数变换，确保算法一致性

### 2. 峰谷识别算法（HL算法）
- 使用颗粒度参数g=3识别摆动高低点
- 对连续相邻的峰谷进行去重处理

### 3. 关键价位计算
- 基于吸引力得分算法识别关键价位
- 使用两阶段筛选算法确保关键价位质量
- 动态计算扩展价位

### 4. 动态Range计算
- 观察窗口跟随K线移动
- 关键价位和扩展价位动态更新
- Range计数基于扩展价位触及

### 5. 交易策略
- 中性挂单策略，3:1风险收益比
- OCO机制管理订单
- 动态更新挂单价格

### 6. 参数优化
- 贝叶斯优化寻找最佳参数组合
- 基于夏普比率优化性能

## 快速开始

1. 安装依赖：
```bash
poetry install
```

2. 运行测试：
```bash
poetry run python -m pytest tests/
```

3. 运行分析脚本：
```bash
poetry run python scripts/analysis/calculate_ranges_dynamic.py
```

## 技术特点

- **对数价格空间**：所有计算在对数空间进行，确保不同价格水平下算法一致性
- **向量化计算**：使用NumPy向量化操作，提高计算效率
- **内存优化**：针对M2 Mac 8GB内存优化，支持大数据集处理
- **动态更新**：关键价位和扩展价位实时更新

## 文档

详细的项目描述请参考 [项目描述2.0.md](项目描述2.0.md)

## 许可证

MIT License