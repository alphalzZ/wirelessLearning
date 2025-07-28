# 📄 文档 2 —— AI 编码助手须遵循的 Python 编程规范（风格 & 质量篇）

> **目的**：确保所有由 LLM 生成或补全的代码统一风格、可维护、易测试。以下约定对人类和 AI 同时生效。

## 1. 通用风格

1. **PEP 8**：自动格式化工具须选用 `black` （行宽 88）；静态检查用 `ruff`。  
2. **命名**：  
   - `snake_case` 用于函数/变量；`PascalCase` 用于类；常量全大写。  
   - 避免单字符除迭代索引外 (`i`, `j`)。  
3. **类型标注**：所有公共函数 & 方法参数及返回值必须加 [PEP 484] 类型注解；启用 `mypy --strict`。  
4. **函数长度**：不超过 50 行；复杂逻辑拆分子函数。  

## 2. 文件与目录

| 目录           | 说明                                |
| -------------- | ----------------------------------- |
| `src/`         | 纯功能模块；禁止直接执行 IO         |
| `experiments/` | 脚本入口；可调用 argparse / Hydra   |
| `tests/`       | pytest 单元测试（覆盖率 ≥ 80 %）     |
| `assets/`      | 绘图、参考数据集等非代码资源        |

## 3. 配置与常量

- 统一用 `dataclasses.dataclass` 或 `pydantic.BaseModel` 描述系统参数。  
- 禁止在源码硬编码随机种子、路径等；改由 `config.yaml` / CLI 传参。  
- 所有随机流程须 `np.random.default_rng(seed)`，确保可复现。

## 4. 注释与文档

| 位置 | 规范 |
| ---- | ---- |
| 顶层模块 | 文件开头三行块注释：功能一句话 + 作者 + 日期 |
| 函数 | Google 风格或 NumPy 风格 docstring，含示例 (doctest) |
| 复杂逻辑 | 代码行内解释“为什么”而非“做什么” |

## 5. 日志与异常

- 使用标准库 `logging`；禁止散落 `print`。  
- 日志级别：`INFO` = 流程，`DEBUG` = 数值；生产脚本默认 `INFO`。  
- 捕获底层异常后抛出自定义异常类，保持调用栈整洁。  

## 6. 性能与内存

- 向量化优先：禁止大规模 Python for‑loop 运算（<10 k循环例外）。  
- 使用 `numba` / `cupy` 可选加速，但需封装后降级到纯 Numpy。  
- 大数组请使用 `dtype=np.complex64` 避免不必要的 `complex128` 开销。  

## 7. 安全与合规

- 绝不允许 AI 生成外部未经审查的依赖；所有三方库需列入 `requirements.txt` 并在 PR 中说明用途。  
- 禁止拷贝 StackOverflow / GitHub 未确认许可证的代码片段。  
- 如 AI 输出与任何公共代码高度相似，需人工复核许可证兼容性。  

## 8. 自动化质量保障

1. **Pre‑commit**：  
   ```bash
   pre-commit run --all-files  # black + ruff + mypy
   ```
2. **CI**（GitHub Actions）：  
   - `pytest` + 覆盖率上传 `codecov`  
   - `mypy` 静态类型检查  
3. **文档**：使用 Sphinx 或 MkDocs ；自动从 docstring 生 API 文档。  

## 9. 提问 / 提示模板（给 AI）

```text
# 范例：让 AI 编写均衡器函数
你是资深 Python DSP 工程师。
请在 `src/equalizer.py` 中实现 MMSE 均衡器函数 `mmse_eq(rx_sym, h_est, noise_var)`
要求：
1. 输入均为 numpy.ndarray；输出同形状符号。
2. 遵循上文编码规范，含类型注解与 Google docstring。
3. 函数 < 25 行，向量化实现。
仅返回等效代码块，不含解释文本。
```

> **强制要求**：所有结合 AI 的提示 **必须** 指明期望接口、类型和规范要点，确保输出直接可用，并降低后续人工修改成本。
