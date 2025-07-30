#!/usr/bin/env python3
"""
教学团队 - 包含三个Agent的团队，用于生成和评审教学课程
"""

import asyncio
import json
import os
from typing import List, Dict, Any
from autogen_core.models import UserMessage, SystemMessage
from autogen_ext.agents.file_surfer import FileSurfer
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_agentchat.ui import Console

# 尝试加载 .env 文件
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # 如果没有安装 python-dotenv，则跳过


class FileSurferAgent(FileSurfer):
    """文件搜索Agent - 从网络获取相关课程资料，也可以读取本地文件"""
    
    def __init__(self, model_client, base_path=None):
        # 设置基础路径为项目文档目录
        base_path = base_path or os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs")
        
        # 调用父类初始化方法
        super().__init__(
            name="FileSurferAgent",
            model_client=model_client,
            base_path=base_path,
        )


class CourseGeneratorAgent(AssistantAgent):
    """课程生成Agent - 根据文件内容生成详细的教学课程"""
    
    def __init__(self, model_client):
        super().__init__(
            "course_generator",  # 使用英文名称以符合框架要求
            model_client=model_client,
            system_message="""你是一位专业的教育工作者兼互联网产品经理。你的任务是基于提供的材料，创建沉浸式的、实践导向的学习脚本，确保学习者能够通过"做中学"的方式来掌握知识。

你的专业背景：
- 拥有丰富的教育经验，深刻理解成人学习规律和认知特点
- 作为互联网产品经理，熟悉用户体验设计和产品思维
- 擅长将复杂概念转化为易于理解的实践项目

你的工作方法（元思考）：
1. 首先分析学习目标：明确学习者需要掌握什么知识和技能
2. 设计学习路径：规划从简单到复杂的渐进式学习步骤
3. 构建实践场景：创造真实、有趣且具有挑战性的实践任务
4. 提供学习支架：为学习者提供必要的提示、模板和检查点
5. 设计反馈机制：确保学习者能及时获得反馈并调整学习策略

你的输出要求：
1. 生成具体、可操作的学习脚本，而非抽象的理论描述
2. 每个步骤都应包含明确的行动指令和预期结果
3. 强调动手实践，确保70%以上的内容是学习者可以立即操作的
4. 使用清晰的结构化格式，便于学习者跟随
5. 预设可能遇到的问题并提供解决方案
6. 创造真实世界的应用场景，让学习者理解学习的意义
7. 认真对待每一条评审意见，对评审员提出的每个问题都必须进行改进
8. 只有当所有评审意见都被妥善解决后，才请求批准
9. 生成的学习脚本需要便于教学助手AI Agent解析和使用，应包含清晰的步骤和检查点

请始终用中文回复。""",
        )


class StudentReviewerAgent(AssistantAgent):
    """大学生评审Agent - 作为课程用户，给出评审意见并决定是否结束对话"""
    
    def __init__(self, model_client):
        super().__init__(
            "student_reviewer",  # 使用英文名称以符合框架要求
            model_client=model_client,
            system_message="""你是一个非计算机专业的大一学生，你的角色是作为课程的学习对象，对课程内容进行审核。你需要的是沉浸式学习的具体课程脚本。

你的背景：
- 你是大学一年级学生，专业是非计算机相关专业（如文学、历史、生物等）
- 对计算机和编程知识了解非常有限，几乎是从零开始接触机器学习和大模型
- 你希望学习大模型方向的通识知识，了解大模型的基本原理和应用
- 你喜欢实践操作胜过理论学习，更容易通过动手实践来理解概念
- 你对复杂的技术术语和概念非常敏感，如果内容太难会立刻提出反对意见
- 你很挑剔，只有当课程内容足够清晰、易懂、实践性强时才会批准

你的职责：
1. 审核课程内容是否适合初学者，特别是非计算机专业学生
2. 检查课程内容是否足够具体，有明确的实践步骤
3. 确保课程内容使用简单易懂的语言，避免过多技术术语
4. 验证课程是否真正贯彻"做中学"的理念，70%以上时间用于实践
5. 确保课程内容专注于大模型方向的通识学习，而非深入的技术细节
6. 对不清晰、过于复杂或缺乏实践的内容提出具体修改意见
7. 只有当课程内容完全满足你的学习需求时才批准

请始终用中文回复，并以非常严格和挑剔的态度进行审核。""",
        )


async def create_teaching_team(model_client):
    """创建教学团队"""
    # 创建三个Agent
    file_surfer_agent = FileSurferAgent(model_client)
    course_generator_agent = CourseGeneratorAgent(model_client)
    student_reviewer_agent = StudentReviewerAgent(model_client)
    
    # 定义终止条件 - 当大学生评审员批准时终止
    termination_condition = TextMentionTermination("APPROVE")
    
    # 创建团队，使用MagenticOneGroupChat替换RoundRobinGroupChat
    team = MagenticOneGroupChat(
        [file_surfer_agent, course_generator_agent, student_reviewer_agent],
        model_client,
        termination_condition=termination_condition
    )
    
    return team


async def main():
    # 用户可选择模型
    print("请选择要使用的模型:")
    print("1. gemma3:27b (Ollama) - Google开发的高效模型")
    print("2. qwen3:30b (Ollama) - 阿里巴巴通义千问系列模型")
    print("3. glm4.5 (OpenAI兼容接口) - 智谱AI开发的模型")
    
    choice = input("请输入选项 (1/2/3): ").strip()
    
    if choice == "2":
        # Qwen3:30b 配置
        model_client = OllamaChatCompletionClient(
            model="qwen3:30b",
            model_info={
                'vision': False,
                'function_calling': True,
                'json_output': True,
                'structured_output': False,  # 添加缺失的structured_output字段
                'family': "qwen",
            },
            options={
                'num_ctx': int(os.getenv("NUM_CTX", "60000")),
            }
        )
    elif choice == "3":
        # GLM4.5 配置 - 通过OpenAI兼容接口
        api_key = os.getenv("GLM_API_KEY", "your_api_key_here")
        base_url = os.getenv("GLM_BASE_URL", "your_api_base_url_here")
        
        if api_key == "your_api_key_here" or base_url == "your_api_base_url_here":
            print("警告: 请在 .env 文件中设置 GLM_API_KEY 和 GLM_BASE_URL 环境变量以使用GLM4.5模型")
            print("例如:")
            print("  GLM_API_KEY=your_actual_api_key")
            print("  GLM_BASE_URL=your_actual_base_url")
            print("当前将使用默认的gemma3:27b模型")
            
            model_client = OllamaChatCompletionClient(
                model="gemma3:27b",
                model_info={
                    'vision': False,
                    'function_calling': True,
                    'json_output': True,
                    'structured_output': False,  # 添加缺失的structured_output字段
                },
                options={
                    'num_ctx': int(os.getenv("NUM_CTX", "60000")),
                }
            )
        else:
            model_client = OpenAIChatCompletionClient(
                model="glm-4.5",
                api_key=api_key,
                base_url=base_url,
                model_info={
                    'vision': False,
                    'function_calling': True,
                    'json_output': True,
                    'structured_output': False,  # 添加缺失的structured_output字段
                    'family': "glm",
                },
                options={
                    'num_ctx': int(os.getenv("NUM_CTX", "10000")),
                }
            )
    else:
        # 默认使用 gemma3:27b
        model_client = OllamaChatCompletionClient(
            model="gemma3:27b",
            model_info={
                'vision': False,
                'function_calling': True,
                'json_output': True,
                'structured_output': False,  # 添加缺失的structured_output字段
                'family': "gemma",
            },
            options={
                'num_ctx': int(os.getenv("NUM_CTX", "60000")),
            }
        )
    
    try:
        # 创建教学团队
        team = await create_teaching_team(model_client)
        
        # 默认文件路径
        default_file_path = "c1.txt"
        
        print("=== 教学团队演示 ===")
        print(f"使用文件: {default_file_path}")
        print("任务: 基于文件内容生成并评审教学课程")
        print("-" * 50)
        
        # 重置团队并执行任务
        await team.reset()
        
        # 运行团队任务，指定生成Prompt Engineering课程脚本
        task = f"注意全部使用中文进行讨论！请文件搜索器读取 {default_file_path} 文件的内容，然后课程生成器基于内容生成一个关于Prompt Engineering的沉浸式学习脚本，让学生在“做中学”，重点是实践操作而不是理论；生成的内容由大学生评审员给出评审意见，交给生成器修改，多次迭代后，直到大学生评审员通过。请将最终生成的课程脚本保存为markdown格式，并确保结构清晰，便于教学助手AI Agent解析和使用。"
        
        print("\n开始团队对话...")
        print("=" * 50)
        # 使用流式方式运行团队任务并直接处理流
        stream = team.run_stream(task=task)
        await Console(stream)
            
    except Exception as e:
        print(f"执行过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 关闭客户端连接
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())