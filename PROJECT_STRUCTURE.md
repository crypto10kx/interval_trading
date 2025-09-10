# 项目结构说明

## 整理后的项目结构

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
│   └── extended_levels.py        # 扩展价位计算
├── data/                         # 数据文件
│   ├── *_5m.csv                 # 原始K线数据（15个交易对）
│   └── *_5m_processed.csv       # 处理后数据（对数变换+ATR）
├── tests/                        # 测试文件
│   ├── test_atr_calculator.py
│   ├── test_attraction_score.py
│   ├── test_data_acquisition.py
│   ├── test_extended_levels.py
│   ├── test_key_levels.py
│   └── test_kline_fetcher.py
├── scripts/                      # 脚本文件
│   ├── analysis/                # 分析脚本
│   │   ├── calculate_ranges_dynamic.py  # 动态Range计算
│   │   ├── analyze_ranges.py           # Range分析
│   │   └── analyze_log_impact.py       # 对数影响分析
│   ├── optimization/            # 优化脚本
│   │   └── test_bayesian_simple.py     # 贝叶斯优化测试
│   └── utilities/               # 工具脚本
│       └── batch_process_modules.py    # 批量处理模块
├── docs/                        # 文档
│   ├── range_analysis_report.md        # Range分析报告
│   └── manual_validation_report.md     # 手动验证报告
├── pyproject.toml               # 项目配置
├── poetry.lock                  # 依赖锁定
├── 项目描述2.0.md              # 项目详细描述
├── 区间交易系统架构文档.md      # 系统架构文档
└── README.md                    # 项目说明
```

## 模块功能说明

### 核心模块 (src/)

1. **data_acquisition.py**: 数据获取主模块
2. **kline_fetcher.py**: 从Binance API获取K线数据
3. **log_transformer.py**: 对数价格变换
4. **peak_valley.py**: HL算法峰谷识别
5. **atr_calculator.py**: ATR计算（真实价格空间）
6. **attraction_score.py**: 吸引力得分计算
7. **key_levels.py**: 关键价位筛选（两阶段算法）
8. **extended_levels.py**: 扩展价位计算

### 数据文件 (data/)

- **原始数据**: 15个USDT永续合约交易对的2年5分钟K线数据
- **处理后数据**: 包含对数价格、ATR、吸引力得分等计算字段

### 测试文件 (tests/)

- 每个核心模块都有对应的单元测试
- 测试覆盖主要功能和边界情况

### 脚本文件 (scripts/)

- **analysis/**: 分析脚本，用于Range计算和性能分析
- **optimization/**: 优化脚本，用于参数调优
- **utilities/**: 工具脚本，用于批量处理

### 文档 (docs/)

- **range_analysis_report.md**: Range分析报告
- **manual_validation_report.md**: 手动验证报告

## 整理原则

1. **模块化**: 每个功能独立成模块
2. **清晰分层**: 核心模块、测试、脚本、文档分离
3. **去除重复**: 删除重复和无关文件
4. **统一命名**: 使用一致的命名规范
5. **功能完整**: 保留所有必要的功能模块

## 使用说明

1. 核心功能在 `src/` 目录下
2. 运行测试: `poetry run python -m pytest tests/`
3. 运行分析: `poetry run python scripts/analysis/calculate_ranges_dynamic.py`
4. 查看文档: 参考 `项目描述2.0.md` 和 `README.md`