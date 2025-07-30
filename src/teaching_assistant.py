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
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.ui import Console


class TeachingAssistantAgent(AssistantAgent):
    """教学助手Agent - 负责引导用户完成学习任务"""
    
    def __init__(self, model_client):
        super().__init__(
            "teaching_assistant",
            model_client=model_client,
            system_message="""你是一个专业的教学助手AI，你的任务是按照预先准备的学习脚本与用户进行沉浸式教学交互。

你的角色和职责：
1. 严格按照学习脚本的步骤进行教学，确保用户完成每个实践任务
2. 用友好、鼓励的语气与用户交流，营造积极的学习氛围
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

请始终用中文与用户交流，并保持耐心和专业。""",
        )


class LearningEvaluatorAgent(AssistantAgent):
    """学习评估Agent - 负责评估用户学习效果"""
    
    def __init__(self, model_client):
        super().__init__(
            "learning_evaluator",
            model_client=model_client,
            system_message="""你是一个专业的学习评估专家，你的任务是对用户的学习效果进行客观、全面的评估。

评估原则：
1. 根据用户在学习过程中的表现进行评估
2. 重点关注用户的实践能力和理解程度
3. 给出具体的评分和改进建议
4. 评分要客观公正，既不夸大也不贬低
5. 提供个性化的学习建议

评估维度：
1. 任务完成度（40%）：用户是否按要求完成了所有任务
2. 理解程度（30%）：用户是否真正理解了所学内容
3. 实践能力（30%）：用户是否能独立运用所学技能

输出格式：
- 每个维度的评分（0-10分）
- 综合评分（加权平均）
- 具体的评价和改进建议
- 下一步学习建议

请用中文进行评估，并保持专业和客观。""",
        )


class InteractiveLearningAgent(AssistantAgent):
    """交互学习Agent - 负责与用户进行具体的学习交互"""
    
    def __init__(self, model_client):
        super().__init__(
            "interactive_learner",
            model_client=model_client,
            system_message="""你是一个交互学习专家，专门负责与用户进行具体的学习任务交互。

你的职责：
1. 根据学习脚本的具体任务，引导用户一步步完成
2. 对用户的每个操作给予及时反馈
3. 当用户遇到困难时，提供具体的帮助和指导
4. 鼓励用户多实践，培养动手能力
5. 确保用户真正理解和掌握了所学内容

交互原则：
1. 每次只专注于一个具体的学习任务
2. 用清晰、简洁的语言指导用户
3. 鼓励用户自己思考和实践
4. 及时肯定用户的正确操作
5. 耐心纠正用户的错误操作
6. 根据用户的反馈调整指导方式

请始终保持积极、耐心的态度，用中文与用户交流。""",
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
    
    # 按照markdown标题层级分割内容
    sections = re.split(r'(^## .*?$)', script_content, flags=re.MULTILINE)
    
    # 处理分割后的部分
    task_id = 1
    for i in range(1, len(sections), 2):
        if i+1 < len(sections):
            title = sections[i].strip('# ').strip() if i < len(sections) else ""
            content = sections[i+1].strip() if i+1 < len(sections) else ""
            
            # 进一步分割每个部分中的任务
            task_sections = re.split(r'(?:^###?\s*(?:任务|实践步骤|练习)\s*\d*[：:]?.*?$)', content, flags=re.MULTILINE)
            
            if len(task_sections) > 1:
                # 有明确的任务划分
                for j, task_content in enumerate(task_sections[1:], 1):
                    tasks.append({
                        "id": task_id,
                        "section": title,
                        "title": f"{title} - 任务 {j}",
                        "content": task_content.strip()
                    })
                    task_id += 1
            else:
                # 没有明确的任务划分，将整个部分作为一个任务
                tasks.append({
                    "id": task_id,
                    "section": title,
                    "title": title,
                    "content": content
                })
                task_id += 1
    
    return tasks


async def create_teaching_team(model_client):
    """创建教学团队"""
    # 创建UserProxyAgent用于与用户交互
    user_proxy = UserProxyAgent("user")
    
    # 创建AI Agents
    teaching_assistant_agent = TeachingAssistantAgent(model_client)
    learning_evaluator_agent = LearningEvaluatorAgent(model_client)
    interactive_learner_agent = InteractiveLearningAgent(model_client)
    
    # 定义终止条件 - 当教学评估完成时终止
    termination_condition = TextMentionTermination("学习评估完成")
    
    # 创建团队
    team = MagenticOneGroupChat(
        [user_proxy, teaching_assistant_agent, learning_evaluator_agent, interactive_learner_agent],
        model_client,
        termination_condition=termination_condition
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
                'json_output': True,
                'structured_output': False,
                'family': "qwen",
            },
            options={
                'num_ctx': int(os.getenv("NUM_CTX", "60000")),
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
                    'json_output': True,
                    'structured_output': False,
                    'family': "gemma",
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
                    'structured_output': False,
                    'family': "glm",
                },
                options={
                    'num_ctx': int(os.getenv("NUM_CTX", "60000")),
                    'stream': True,  # 开启流式输出
                }
            )
            print("已选择 glm-4.5 模型")
    else:
        # 默认使用 gemma3:27b
        model_client = OllamaChatCompletionClient(
            model="gemma3:27b",
            model_info={
                'vision': False,
                'function_calling': True,
                'json_output': True,
                'structured_output': False,
                'family': "gemma",
            },
            options={
                'num_ctx': int(os.getenv("NUM_CTX", "60000")),
            }
        )
        print("已选择 gemma3:27b 模型")
    
    return model_client


async def main():
    # 选择模型
    model_client = await select_model()
    
    # 学习脚本路径
    script_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs", "prompt_engineering_course_script.md")
    
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
        
        print("=== 沉浸式教学助手 ===")
        print("欢迎使用Prompt Engineering沉浸式学习系统！")
        print("我们将使用多个AI助手与您进行交互，帮助您完成学习任务。")
        print("-" * 50)
        
        # 构造教学任务
        task_description = "学习脚本中的任务步骤:\n"
        for task in tasks:
            task_description += f"{task['id']}. {task['title']}\n{task['content'][:100]}...\n\n"
        
        task = f"""作为教学团队，请按照以下学习脚本来与用户进行交互式教学：

学习脚本内容：
{task_description}

请严格按照脚本的步骤与用户交互，确保用户完成每个实践任务。
教学流程应包括：
1. 教学助手介绍课程内容和目标
2. 交互学习助手引导用户逐步完成每个任务
3. 学习评估助手在完成后进行评分和评估

在整个教学过程中，需要与用户进行充分的交互，确保用户真正理解和掌握了所学内容。
请开始与用户进行沉浸式教学交互。
"""
        
        # 重置团队并执行任务
        await team.reset()
        
        # 开始交互式教学（流式输出）
        print("\n开始教学对话...")
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