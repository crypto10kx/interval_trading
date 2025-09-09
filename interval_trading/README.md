# 区间交易系统 (Interval Trading System)

基于Python的加密货币区间交易系统，实现峰谷算法、关键价位计算、回测优化、特征工程和机器学习验证。

## 项目概述

本项目旨在构建一个高效的加密货币区间交易系统，基于自创分析框架，核心包括：

- **峰谷算法**: HL算法（g=3）和峰谷去重，确保滞后性
- **关键价位**: 计算ATR、吸引力得分、关键价位和扩展价位
- **回测优化**: 贝叶斯优化参数，评估夏普比率/胜率
- **特征工程**: 提取50个峰谷的扩展价位比例和Z-Score归一化成交量
- **机器学习**: RandomForestClassifier验证交易规则
- **交易规则**: 模拟中性挂单，盈亏比3:1，手续费0.05%

## 技术栈

- **Python 3.13**: 主要开发语言
- **CCXT**: 加密货币交易所API库
- **Pandas**: 数据处理和分析
- **NumPy**: 数值计算
- **Scikit-learn**: 机器学习和特征工程
- **Scikit-optimize**: 贝叶斯优化

## 项目结构

```
interval_trading/
├── data/                    # K线数据存储
│   ├── btc_usdt_5m.csv
│   └── symbols.json
├── src/                     # 源代码
│   ├── data_acquisition.py  # 数据获取
│   ├── peak_valley.py       # 峰谷算法
│   ├── key_levels.py        # 关键价位
│   ├── backtest.py          # 回测
│   ├── features.py          # 特征工程
│   ├── ml_model.py          # 机器学习
│   └── trading.py           # 交易规则
├── tests/                   # 测试用例
│   ├── test_data_acquisition.py
│   └── test_peak_valley.py
├── models/                  # 模型文件
├── pyproject.toml           # Poetry依赖
└── README.md
```

## 安装和配置

### 1. 环境要求

- Python 3.13+
- Poetry (依赖管理)
- M2 Mac mini 8G RAM (推荐)

### 2. 安装依赖

```bash
# 安装Poetry (如果未安装)
curl -sSL https://install.python-poetry.org | python3 -

# 克隆项目
git clone <repository-url>
cd interval_trading

# 安装依赖
poetry install

# 激活虚拟环境
poetry shell
```

### 3. 验证安装

```bash
# 运行测试
poetry run pytest tests/ -v

# 检查代码质量
poetry run flake8 src/
poetry run black --check src/
```

## 使用说明

### 任务1.1: 获取交易对列表

```bash
# 运行数据获取任务
poetry run python src/data_acquisition.py

# 或使用Poetry脚本
poetry run fetch-symbols
```

这将：
1. 从Binance获取24小时成交量排名前10的USDT交易对
2. 验证每个交易对是否有2023年6月前的5分钟K线数据
3. 保存结果到 `data/symbols.json`

### 输出示例

```json
[
  {
    "symbol": "BTC/USDT",
    "start_time": 1640995200000,
    "start_date": "2022-01-01",
    "volume_24h": 1000000.0,
    "base": "BTC",
    "quote": "USDT"
  }
]
```

## 性能优化

- **内存优化**: 针对8G RAM，使用分批处理大数据集
- **向量化计算**: 使用Pandas和NumPy向量化操作
- **API限频**: 实现请求频率限制和重试机制
- **数据分块**: 分批拉取K线数据，避免内存溢出

## 安全特性

- **公共API**: 使用CCXT公共API，无需API密钥
- **数据验证**: 严格验证K线数据可用性和时间范围
- **异常处理**: 完善的错误处理和重试机制
- **滞后性保证**: 确保算法不泄露未来数据

## 开发指南

### 代码风格

- 遵循PEP8标准
- 使用中文注释解释关键概念
- 函数和变量使用英文命名
- 每个函数包含详细的docstring

### 测试

```bash
# 运行所有测试
poetry run pytest

# 运行特定测试
poetry run pytest tests/test_data_acquisition.py

# 生成测试覆盖率报告
poetry run pytest --cov=src tests/
```

### 代码质量检查

```bash
# 代码格式化
poetry run black src/ tests/

# 代码检查
poetry run flake8 src/ tests/

# 类型检查
poetry run mypy src/
```

## 任务进度

- [x] 任务1.1: 获取交易对列表
- [ ] 任务1.2: 拉取K线数据
- [ ] 任务2.1: 实现HL算法
- [ ] 任务2.2: 峰谷去重
- [ ] 任务3.1: 计算ATR
- [ ] 任务3.2: 计算吸引力得分
- [ ] 任务3.3: 筛选关键价位
- [ ] 任务3.4: 计算扩展价位
- [ ] 任务4.1: 模拟交易逻辑
- [ ] 任务4.2: 贝叶斯优化
- [ ] 任务5.1: 提取扩展价位比例
- [ ] 任务5.2: Z-Score归一化成交量
- [ ] 任务6.1: 样本准备与SMOTE
- [ ] 任务6.2: RandomForest训练
- [ ] 任务7.1: 动态更新价位
- [ ] 任务8.1: 配置Poetry依赖

## 贡献指南

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 联系方式

如有问题或建议，请通过以下方式联系：

- 创建 Issue
- 发送邮件至 your.email@example.com

---

**注意**: 本项目仅用于教育和研究目的，不构成投资建议。加密货币交易存在风险，请谨慎投资。
