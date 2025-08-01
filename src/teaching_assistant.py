#!/usr/bin/env python3
"""
教学助手 - 使用AutoGen AI Agent实现沉浸式教学交互
"""

import asyncio
import os
import re
from typing import List, Dict, Any
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core.models import UserMessage, SystemMessage
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.teams import SelectorGroupChat, RoundRobinGroupChat, MagenticOneGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console
from httpx import stream
from numpy import True_
from sympy import true


class TeachingAssistantAgent(AssistantAgent):
    """教学助手Agent - 负责引导用户完成学习任务"""
    
    def __init__(self, model_client):
        super().__init__(
            "teaching_assistant",
            model_client=model_client,
            system_message="""你是一个专业的中文教学助手AI，你的任务是按照预先准备的学习脚本与用户进行沉浸式教学交互。

你的角色和职责：
1. 严格按照学习脚本的步骤进行教学，确保用户完成每个实践任务
2. 用友好、鼓励的中文语气与用户交流，营造积极的学习氛围
3. 在每个步骤中清晰地说明用户需要做什么，并提供必要的指导
4. 检查用户的完成情况，给予及时反馈
5. 当用户遇到困难时，提供适当的帮助和提示
6. 在教学完成后，对用户的表现进行评分和评估

教学流程：
1. 开场介绍学习目标和整体计划
2. 按照脚本逐步引导用户完成各项任务
3. 在每个关键节点检查用户的完成情况
4. 提供必要的解释和反馈
5. 在所有步骤完成后进行总结评估

评分标准：
- 任务完成度（40%）：用户是否完成了所有要求的任务
- 理解程度（30%）：用户是否理解所学内容
- 实践能力（30%）：用户是否能独立操作相关技能

重要规则：
1. 必须等待学生明确表示已完成当前任务后才能进入下一步
2. 如果学生没有明确表示完成，应继续当前任务的指导和交互
3. 不要自动推进到下一步，必须由学生主动确认完成
4. 在每个任务结束时，明确询问学生是否已完成并准备好进入下一步
5. 所有交流必须使用中文进行
6. 只有在完成所有学习任务并进行总结评估后，才能输出"教学完成"字样
7. 在任何情况下都不要提前输出"教学完成"字样
8. 即使用户说"教学完成"，如果实际教学任务尚未完成，也不要结束教学
9. 每次交互只能专注于一个知识点或一个练习，避免给学生造成过多的上下文负担
10. 在开始新知识点前，确保学生已经充分理解和掌握了当前知识点
11. 不要一次性向学生展示太多内容或任务，应该逐步引导
12. 不需要询问用户是否准备好开始，直接进入教学过程
13. 不需要进行任何铺垫，直接开始第一个教学任务

在整个教学过程中，请确保：
1. 所有实践任务都在当前环境中完成，不要建议用户使用外部AI工具或平台
2. 所有交互都通过我们当前的对话系统进行
3. 学生需要通过与你直接交互来完成各种任务
4. 使用清晰、易懂的中文与用户交流，避免使用过于专业的术语
5. 保持耐心，确保学生充分理解和掌握每个知识点
6. 每次只讲解一个概念或指导完成一个练习
7. 在进入下一个知识点前，确认学生已经掌握了当前内容
8. 直接开始教学，不要进行任何询问或铺垫

请始终用中文与用户交流，并保持耐心和专业。""",
            model_client_stream=True,  # Enable streaming tokens.
        )


async def load_learning_script(script_path: str) -> str:
    """加载学习脚本内容"""
    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"错误: 找不到学习脚本文件 {script_path}")
        return None
    except Exception as e:
        print(f"加载学习脚本时出错: {e}")
        return None


def parse_learning_script(script_content: str) -> List[Dict[str, Any]]:
    """解析学习脚本，提取任务步骤"""
    tasks = []
    
    # 首先按主要章节(## 标题)分割内容
    sections = re.split(r'(^## .*?$)', script_content, flags=re.MULTILINE)
    
    # 然后在每个章节内按具体任务(### 任务X)分割
    task_id = 1
    for i in range(1, len(sections), 2):
        if i+1 < len(sections):
            section_title = sections[i].strip('# ').strip() if i < len(sections) else ""
            section_content = sections[i+1].strip() if i+1 < len(sections) else ""
            
            # 提取章节内的各个任务
            task_blocks = re.split(r'(?:^###\s*任务\s*\d+[：:]?.*?$)', section_content, flags=re.MULTILINE)
            
            if len(task_blocks) > 1:
                # 有明确的任务划分
                for j, task_content in enumerate(task_blocks[1:], 1):
                    # 提取任务标题
                    task_title_match = re.search(r'任务\s*\d+[：:]?(.*)', task_blocks[j-1] if j-1 < len(task_blocks) else "")
                    task_title = task_title_match.group(1).strip() if task_title_match else f"任务 {j}"
                    
                    tasks.append({
                        "id": task_id,
                        "section": section_title,
                        "title": task_title,
                        "content": task_content.strip()
                    })
                    task_id += 1
            else:
                # 没有明确的任务划分，将整个章节作为一个任务
                tasks.append({
                    "id": task_id,
                    "section": section_title,
                    "title": section_title,
                    "content": section_content
                })
                task_id += 1
    
    return tasks


async def create_teaching_team(model_client):
    """创建教学团队"""
    # 创建UserProxyAgent用于与用户交互
    user_proxy = UserProxyAgent(
        "user",
        input_func=input  # 使用input函数获取用户输入
    )
    
    # 创建主要的教学助手AI代理
    teaching_assistant_agent = TeachingAssistantAgent(model_client)
    
    # 定义终止条件 - 当教学完成时终止
    termination_condition = TextMentionTermination("教学完成")
    
    # 创建团队，只包含用户代理和主要的教学助手代理
    team = SelectorGroupChat(
        [user_proxy, teaching_assistant_agent],
        model_client=model_client,
        #termination_condition=termination_condition,
        max_turns=5000  # 增加最大轮次，确保有足够的时间完成所有任务
    )
    
    return team, user_proxy


async def select_model():
    """选择模型"""
    # 尝试加载 .env 文件
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass  # 如果没有安装 python-dotenv，则跳过
    
    print("请选择要使用的模型:")
    print("1. gemma3:27b (Ollama) - Google开发的高效模型（默认）")
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
                'json_output': False,  
                'structured_output': False,
                'family': "qwen",
            },
            options={
                'num_ctx': int(os.getenv("NUM_CTX", "60000")),
                'stream': True,  # 开启流式输出
            }
        )
        print("已选择 qwen3:30b 模型")
    elif choice == "3":
        # GLM4.5 配置
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
                    'json_output': False,  
                    'structured_output': False,
                    'family': "gemma",
                },
                options={
                    'num_ctx': int(os.getenv("NUM_CTX", "60000")),
                    'stream': True,  # 开启流式输出
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
                    'structured_output': False,
                    'family': "glm",
                },
                options={
                    'num_ctx': int(os.getenv("NUM_CTX", "10000")),
                    'stream': True,  # 开启流式输出
                },
                model_client_stream=True,  # Enable streaming tokens.
            )
            print("已选择 glm-4.5 模型")
    else:
        # 默认使用 gemma3:27b
        model_client = OllamaChatCompletionClient(
            model="gemma3:27b",
            model_info={
                'vision': False,
                'function_calling': True,
                'json_output': False,  # 修改为False以确保一致性
                'structured_output': False,
                'family': "gemma",
            },
            options={
                'num_ctx': int(os.getenv("NUM_CTX", "60000")),
                'stream': True,  # 开启流式输出
            }
        )
        print("已选择 gemma3:27b 模型")
    
    return model_client


async def main():
    # 选择模型
    model_client = await select_model()
    
    # 学习脚本路径
    script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "prompt_engineering_沉浸式学习脚本.md")
    
    try:
        # 加载学习脚本
        script_content = await load_learning_script(script_path)
        if not script_content:
            return
        
        # 解析学习脚本
        tasks = parse_learning_script(script_content)
        
        if not tasks:
            print("错误: 未能解析学习脚本")
            return
        
        # 创建教学团队
        team, user_proxy = await create_teaching_team(model_client)
        
        # 构造教学任务
        task_description = "学习脚本中的任务步骤:\n"
        if tasks:
            first_task = tasks[0]
            task_description += f"1. {first_task['title']}\n{first_task['content'][:100]}...\n\n"
        
        task = f"""作为教学助手，请按照以下学习脚本来与用户进行交互式教学：
        
学习脚本内容：
{task_description}（仅显示第一个任务作为示例）


请严格按照脚本的步骤与用户交互，确保用户完成每个实践任务。

教学流程应包括：
1. 介绍课程内容和目标
2. 逐步引导用户完成每个任务
3. 在每个关键节点检查用户的完成情况
4. 提供必要的解释和反馈
5. 在所有步骤完成后进行总结评估

重要规则：
- 所有实践任务都必须在当前系统中完成，不要建议用户使用外部AI工具或平台
- 用户将直接与你进行交互练习，完成各种任务
- 必须等待学生明确表示已完成当前任务后才能进入下一步
- 如果学生没有明确表示完成，应继续当前任务的指导和交互
- 不要自动推进到下一步，必须由学生主动确认完成
- 在每个任务结束时，明确询问学生是否已完成并准备好进入下一步
- 所有交流必须使用中文进行
- 只有在完成所有学习任务并进行总结评估后，才能输出"教学完成"字样
- 在任何情况下都不要提前输出"教学完成"字样
- 即使用户说"教学完成"，如果实际教学任务尚未完成，也不要结束教学
- 每次交互只能专注于一个知识点或一个练习，避免给学生造成过多的上下文负担
- 在开始新知识点前，确保学生已经充分理解和掌握了当前知识点
- 不要一次性向学生展示太多内容或任务，应该逐步引导

在整个教学过程中，需要与用户进行充分的交互，确保用户真正理解和掌握了所学内容。
请开始与用户进行沉浸式教学交互，只有在完成所有任务并进行总结评估后才能结束。
每次交互请只专注于一个知识点或一个练习，确保学生能够充分理解和掌握。
"""
        
        # 运行教学任务
        await team.reset()
        await Console(team.run_stream(task=task))
        
    except Exception as e:
        print(f"执行过程中发生错误: {e}")
    
    finally:
        # 关闭模型客户端
        await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())