# OFDM仿真系统

这是一个基于Python的OFDM（正交频分复用）系统仿真项目。

## 功能特点

- 完整的OFDM基带处理链路
- 支持QPSK/16QAM/64QAM调制
- AWGN和瑞利衰落信道模型
- BER/SER性能评估
- 可配置的系统参数
- nnrx

## 目录结构

```
ofdm_sim/
├── src/                    # 源代码
│   ├── config.py          # 参数配置
│   ├── ofdm_tx.py         # 发送端处理
│   ├── ofdm_rx.py         # 接收端处理
│   ├── channel.py         # 信道模型
│   └── metrics.py         # 性能指标
|   |—— nnrx/
|       |—— customer-layers.py  #神经网络
|       |—— data_generator.py   #本地数据生成
|       |—— train_with_local.py #本地训练
|       |—— train_with_remote.py#结合远端数据在线训练
├── experiments/           # 实验脚本
├── tests/                # 单元测试
├── assets/              # 资源文件
├── config.yaml          # 配置文件
└── requirements.txt     # 项目依赖
```

## 快速开始

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 运行仿真：
```bash
python experiments/run_experiments.py
```

## 开发规范

- 遵循PEP 8编码规范
- 使用类型注解
- 完整的文档字符串
- 单元测试覆盖

## 许可证

MIT License 
