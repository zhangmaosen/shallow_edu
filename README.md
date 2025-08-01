# 本地大模型教学系统

本项目演示了如何使用 Autogen 框架与本地运行的 Ollama 大语言模型进行交互，创建一个完整的教学系统。

## 功能特性

1. **多代理协作教学团队** - 自动生成和评审教学课程
2. **沉浸式学习脚本生成** - 基于材料自动生成实践导向的学习内容
3. **交互式教学助手** - 与用户逐步交互完成教学过程并进行评分评估
4. **流式对话体验** - 所有交互都采用实时流式输出，提供流畅的对话体验

## 项目结构

```
.
├── src/
│   ├── ollama_agent.py          # Ollama模型基础交互示例
│   ├── teaching_team.py          # 多代理教学团队系统
│   ├── web_surfer_agent.py       # 网页内容爬取代理
│   └── teaching_assistant.py     # 交互式教学助手
├── docs/
│   ├── c1.txt                   # 原始教学材料
│   └── prompt_engineering_course_script.md  # 生成的学习脚本
├── examples/
│   └── conversation_example.py   # 对话示例
├── notebook/
│   └── test.ipynb               # Jupyter Notebook测试
├── tests/
│   ├── test_teaching_team.py    # 教学团队测试
│   ├── test_file_handler_agent.py  # FileHandlerAgent专门测试
│   ├── test_file_handler_simple.py # FileHandlerAgent简单测试
│   ├── test_file_handler_direct.py # FileHandlerAgent直接工具调用测试
│   ├── test_file_handler_integration.py # FileHandlerAgent集成测试
│   └── run_tests.py             # 测试运行脚本
└── README.md
```

## 系统要求

- Python 3.x
- Ollama (已安装并运行)
- gemma3:27b 模型 (或其他兼容模型)

## 安装步骤

1. 安装依赖:
   ```bash
   pip install -r requirements.txt
   ```

2. 安装 Ollama (如果尚未安装):
   访问 https://ollama.com/ 下载并安装

3. 拉取所需模型:
   ```bash
   ollama pull gemma3:27b
   ollama pull qwen3:30b
   ```

## 使用方法

### 1. 多代理教学团队
使用多个AI代理协作生成和评审教学内容:

```bash
python src/teaching_team.py
```

### 2. 交互式教学助手
与用户逐步交互完成教学过程:

```bash
python src/teaching_assistant.py
```

### 3. 基础模型交互
简单的Ollama模型交互示例:

```bash
python src/ollama_agent.py
```

## 运行测试

为了确保系统正常运行，项目包含了针对教学团队中各个Agent的单元测试:

```bash
python tests/run_tests.py
```

测试内容包括:
- FileHandlerAgent（文件处理代理）的功能测试
- CourseGeneratorAgent（课程生成代理）的初始化测试
- CurriculumDirectorAgent（教研组负责人代理）的初始化测试
- StudentAgent（学生代理）的初始化测试
- 教学团队集成测试

### FileHandlerAgent专门测试

FileHandlerAgent负责文件读取和保存操作，有多项专门测试验证其功能：

1. **简单测试** (`test_file_handler_simple.py`) - 基本功能验证
   ```bash
   python tests/test_file_handler_simple.py
   ```

2. **直接工具调用测试** (`test_file_handler_direct.py`) - 验证工具函数的直接调用
   ```bash
   python tests/test_file_handler_direct.py
   ```

3. **集成测试** (`test_file_handler_integration.py`) - 在团队环境中测试工具调用
   ```bash
   python tests/test_file_handler_integration.py
   ```

测试覆盖的功能：
- 保存文件功能（包括自动扩展名添加）
- 读取文件功能
- 异常处理（如读取不存在的文件）
- 长内容处理
- 在团队环境中的工具调用

## 工作流程

1. **内容生成**: [teaching_team.py](file:///home/userroot/dev/shallow_edu/course/src/teaching_team.py) 使用多个AI代理基于原始材料生成学习脚本
2. **教学执行**: [teaching_assistant.py](file:///home/userroot/dev/shallow_edu/course/src/teaching_assistant.py) 使用生成的学习脚本与用户交互完成教学
3. **评估反馈**: 教学完成后对用户表现进行评分和评估

## 流式对话特性

本系统所有交互都采用流式对话实现，具有以下优势：

- **实时响应**: 用户可以实时看到AI代理的回复，无需等待完整响应
- **自然交互**: 对话过程更加自然流畅，类似真实的人际交流
- **即时反馈**: 用户可以在AI代理回复过程中获得即时信息反馈

系统使用 AutoGen 的 `team.run_stream()` 方法和 `Console()` 类实现流式输出，确保每个字符逐字显示，提供打字机效果般的体验。

## 自定义配置

可以通过修改 `.env` 文件来配置模型参数:
- `NUM_CTX`: 上下文长度 (默认: 60000)

## 许可证

本项目基于 MIT 许可证开源。