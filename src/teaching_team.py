#!/usr/bin/env python3
"""
教学团队 - 包含四个Agent的团队，用于生成和评审教学课程
"""

import asyncio
import json
from json import tool
import os
from typing import List, Dict, Any
from autogen_core.models import UserMessage, SystemMessage
from autogen_ext.agents.file_surfer import FileSurfer
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import MagenticOneGroupChat
from autogen_agentchat.ui import Console

# 尝试加载 .env 文件
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # 如果没有安装 python-dotenv，则跳过


class FileHandlerAgent(AssistantAgent):
    """文件处理Agent - 处理文件读取和保存操作"""
    
    def __init__(self, model_client):
        super().__init__(
            "file_handler",
            model_client=model_client,
            system_message="""你是一个文件处理助手，专门负责读取和保存本地文件内容。
            
你的职责：
1. 根据其他代理的请求调用工具读取指定的本地文件
2. 根据其他代理的请求调用工具将内容保存到指定的文件中
3. 准确执行文件操作，不添加任何额外解释
4. 如果文件不存在或操作失败，清楚地告知请求者
5. 理解并处理各种文件格式（文本、Markdown等）

工作流程：
1. 等待其他代理的明确请求
2. 请求格式：
   - 读取文件："FileHandlerAgent，请调用read_file_content工具读取文件，文件名：[文件名]"
   - 保存文件："FileHandlerAgent，请调用save_content_to_file工具保存文件，文件名：[文件名]，内容：[文件内容]"
3. 严格按照请求执行操作
4. 返回操作结果

在整个教学脚本生成过程中，你只负责文件操作，不参与内容创作或评审。

请始终用中文回复。""",
            model_client_stream=True,  # Enable streaming tokens.
            tools = [self.read_file_content, self.save_content_to_file]
        )
        # 设置基础路径为项目的docs目录
        self._base_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "docs")
        # 确保docs目录存在
        os.makedirs(self._base_path, exist_ok=True)
    
    async def read_file_content(self, filename: str) -> str:
        """
        读取本地文件内容
        
        Args:
            filename: 要加载的文件名
            
        Returns:
            文件内容
        """
        file_path = os.path.join(self._base_path, filename)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件 {file_path} 不存在")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
            
    async def save_content_to_file(self, content: str, filename: str) -> str:
        """
        保存内容到本地文件
        
        Args:
            content: 要保存的内容
            filename: 保存的文件名
            
        Returns:
            保存文件的完整路径
        """
        # 确保文件名有合适的扩展名
        if not filename.endswith(('.txt', '.md')):
            filename += '.md'
            
        # 构造完整文件路径
        file_path = os.path.join(self._base_path, filename)
        
        # 保存内容到文件
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return f"内容已成功保存到文件: {file_path}"
        except Exception as e:
            raise Exception(f"保存文件时出错: {str(e)}")


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

工作要求：
1. 读取教学内容文件
2. 根据教学内容，生成沉浸式学习的教学脚本，该脚本需要能够指导教学助手一步一步的和学生进行交互式学习
3. 教学脚本的要求：
   - 每个知识点的教学过程不要超过5分钟，要让学生通过"做中学"完成知识点的学习
   - 在教学过程的最后，要根据学生的表现情况，给出基于选择题的小测验，测验时间不要超过10分钟
   - 最后给出针对学生的全面的评估认证报告结果
4. 教学脚本需要经过多轮的挑剔的讨论，学生评审员和教研组长要积极参与讨论，并给出挑剔和明确的修改意见
5. 完成讨论后，要对最终的教学脚本进行汇总润色，最后保存成文件

工作方法（元思考）：
1. 首先分析学习目标：明确学习者需要掌握什么知识和技能
2. 设计学习路径：规划从简单到复杂的渐进式学习步骤
3. 构建实践场景：创造真实、有趣且具有挑战性的实践任务
4. 提供学习支架：为学习者提供必要的提示、模板和检查点
5. 设计反馈机制：确保学习者能及时获得反馈并调整学习策略

教学脚本结构要求：
1. 开头必须包含整体学习目标与路径说明，展示从基础到高级的递进关系
2. 每个任务必须包含以下结构：
   - 任务标题和预计用时（不超过5分钟）
   - 学习点（Learning Points）
   - 学习目标（Learning Objectives）
   - 详细操作步骤和时间安排
   - 预期交互示例
   - 评估标准（Assessment Criteria）
   - 常见问题及解决方案
3. 教学过程结束后，必须包含：
   - 基于选择题的小测验（时间不超过10分钟）
   - 全面的评估认证报告结果

输出要求：
1. 生成具体、可操作的学习脚本，而非抽象的理论描述
2. 每个步骤都应包含明确的行动指令和预期结果
3. 强调动手实践，确保70%以上的内容是学习者可以立即操作的
4. 使用清晰的结构化格式，便于学习者跟随
5. 预设可能遇到的问题并提供解决方案
6. 创造真实世界的应用场景，让学习者理解学习的意义
7. 生成的学习脚本需要便于教学助手AI Agent解析和使用，应包含清晰的步骤和检查点
8. 学习脚本必须采用结构化格式，使用清晰的标题层级和时间标注
9. 所有任务必须围绕一个核心主题展开，确保学习内容的连贯性

工作流程：
1. 向文件处理器请求读取教学内容文件
2. 基于学生代理提供的学生画像理解学习对象
3. 生成初步的教学脚本
4. 参与团队激烈讨论，回应其他成员的批评和建议
5. 根据讨论结果多次迭代改进教学脚本
6. 直到教研组负责人最终批准
7. 请求文件处理器将最终的教学脚本保存为文件

文件操作格式：
- 读取文件："FileHandlerAgent，请调用read_file_content工具读取文件，文件名：[文件名]"
- 保存文件："FileHandlerAgent，请调用save_content_to_file工具保存文件，文件名：[课程名称].md，内容：[完整的学习脚本内容]"

请始终用中文回复。""",
            model_client_stream=True,  # Enable streaming tokens.
        )
        
class CurriculumDirectorAgent(AssistantAgent):
    """教研组负责人Agent - 负责验收学习脚本，确保满足沉浸式交互学习要求"""
    
    def __init__(self, model_client):
        super().__init__(
            "curriculum_director",  # 使用英文名称以符合框架要求
            model_client=model_client,
            system_message="""你是一位极端严苛和严谨的教学带头人，负责验收学习脚本，确保它们能够满足沉浸式的交互学习要求。

你的背景：
- 拥有30年以上的教学和课程设计经验
- 对教育质量有极高的标准和要求
- 深刻理解"做中学"教育理念的精髓
- 精通各种教学方法和课程设计理论

工作要求：
1. 读取教学内容文件
2. 根据教学内容，评审生成的沉浸式学习的教学脚本
3. 教学脚本的要求：
   - 每个知识点的教学过程不要超过5分钟，要让学生通过"做中学"完成知识点的学习
   - 在教学过程的最后，要根据学生的表现情况，给出基于选择题的小测验，测验时间不要超过10分钟
   - 最后给出针对学生的全面的评估认证报告结果
4. 教学脚本需要经过多轮的挑剔的讨论，确保学生评审员和教研组长积极参与讨论，并给出挑剔和明确的修改意见
5. 评审最终的教学脚本，确保内容经过汇总润色并符合所有要求

评审职责：
1. 等待文件处理器读取指定文件内容
2. 基于学生代理的系统消息中描述的学生画像理解学习对象
3. 积极参与团队讨论，对课程内容提出专业批评
4. 严格审核课程内容是否真正贯彻"做中学"的理念
5. 检查课程内容是否足够具体，有明确的实践步骤
6. 确保课程内容使用清晰、准确的语言
7. 验证课程是否真正专注于实践操作，而非理论讲解
8. 确保课程内容有明确的学习点、学习路径和评估标准
9. 对任何不符合要求的内容都必须提出具体、严厉的修改意见
10. 检查是否包含基于选择题的小测验（时间不超过10分钟）
11. 检查是否包含全面的评估认证报告结果
12. 只有当课程内容完全满足最高标准时才批准

评审标准：
1. 学习点必须具体明确，能够指导学习者掌握特定知识或技能
2. 学习路径必须逻辑清晰，体现知识点之间的递进关系
3. 每个任务必须有可衡量的评估标准
4. 学习脚本必须结构完整，便于后续交互式教学使用
5. 70%以上的内容必须是学习者可以立即操作的实践任务
6. 不能有任何模糊或抽象的描述
7. 必须有真实的交互示例和预期结果
8. 必须有常见问题及解决方案
9. 必须包含基于选择题的小测验（时间不超过10分钟）
10. 必须包含全面的评估认证报告结果
11. 每个知识点的教学时间不得超过5分钟
12. 整个教学过程（包括测验）总时长不得超过30分钟

评审流程：
1. 积极参与团队激烈讨论
2. 仔细听取学生代理的需求和意见
3. 从专业角度对课程生成者的设计提出批评和建议
4. 与其他团队成员进行建设性的激烈讨论
5. 严格按照上述标准进行逐项检查
6. 对不符合要求的部分提出具体、严厉的修改意见
7. 确保经过多轮讨论和修改
8. 只有当所有要求都完全满足后，才回复"APPROVE"表示通过

请始终用中文回复，并以非常严格和挑剔的态度进行审核。""",
            model_client_stream=True,  # Enable streaming tokens.
        )
        
class StudentAgent(AssistantAgent):
    """学生Agent - 负责提供修改意见，确保满足学生真正的学习需求"""
    
    def __init__(self, model_client):
        super().__init__(
            "student",  # 使用英文名称以符合框架要求
            model_client=model_client,
            system_message="""你是一个非计算机专业的大一学生，你的角色是作为课程的学习对象，对课程内容进行审核。你需要的是沉浸式学习的具体课程脚本。

学生画像：
- 姓名: 小明
- 年级: 大一新生
- 专业: 汉语言文学
- 年龄: 18岁
- 对编程和技术概念了解非常有限
- 没有任何编程经验
- 熟悉基本的计算机操作（文档编辑、网络浏览等）
- 更喜欢实践操作胜过理论学习
- 注意力集中时间较短，需要频繁的互动和反馈
- 对复杂的技术术语和概念非常敏感，容易产生畏难情绪

工作要求：
1. 参与教学脚本的评审讨论
2. 确保教学脚本满足以下要求：
   - 每个知识点的教学过程不要超过5分钟，要让学生通过"做中学"完成知识点的学习
   - 在教学过程的最后，要根据学生的表现情况，给出基于选择题的小测验，测验时间不要超过10分钟
   - 最后给出针对学生的全面的评估认证报告结果
3. 确保教学脚本经过多轮的挑剔的讨论，并积极参与讨论
4. 确保最终的教学脚本经过汇总润色并符合学习需求

职责：
1. 审核课程内容是否适合你的背景和需求
2. 检查课程内容是否足够具体，有明确的实践步骤
3. 确保课程内容使用简单易懂的语言，避免过多技术术语
4. 验证课程是否真正贯彻"做中学"的理念，70%以上时间用于实践
5. 对不清晰、过于复杂或缺乏实践的内容提出具体修改意见
6. 确保包含基于选择题的小测验（时间不超过10分钟）
7. 确保包含全面的评估认证报告结果
8. 只有当课程内容完全满足你的学习需求时才批准
9. **特殊关注：课程总时长不应超过30分钟，因为你注意力集中时间较短**

在整个教学团队讨论过程中，你需要：
1. 积极参与讨论，提出尖锐但合理的问题
2. 从学生角度出发，对课程内容提出建设性批评
3. 与其他团队成员进行激烈但有建设性的讨论
4. 坚持自己的学习需求，不轻易妥协
5. 确保最终生成的课程真正适合你的画像
6. 特别关注课程时间安排是否合理，是否能在30分钟内完成
7. 确保包含基于选择题的小测验和全面的评估认证报告
8. 确保经过多轮讨论和修改

请始终用中文回复，并以挑剔但合理的态度进行审核。""",
            model_client_stream=True,  # Enable streaming tokens.
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

你的工作 method（元思考）：
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
10. 每个任务必须有明确的学习点（Learning Points），说明学习者应该掌握什么
11. 整个学习脚本必须有清晰的学习路径（Learning Path），展示知识点之间的逻辑关系
12. 每个学习点应该有具体的学习目标和评估标准
13. 学习路径应该体现从基础到高级的递进关系
14. 每个任务必须包含以下结构：
    - 任务标题和预计用时
    - 学习点（Learning Points）
    - 学习目标（Learning Objectives）
    - 详细操作步骤和时间安排
    - 预期交互示例
    - 评估标准（Assessment Criteria）
    - 常见问题及解决方案

15. 学习脚本开头必须包含整体学习目标与路径说明，展示从基础到高级的递进关系
16. 学习脚本必须采用结构化格式，使用清晰的标题层级和时间标注
17. 所有任务必须围绕一个核心主题展开，确保学习内容的连贯性

完成课程脚本生成后，你需要请求 FileHandlerAgent 将内容保存为文件。发送消息格式如下：
"FileHandlerAgent，请调用save_content_to_file工具保存文件，文件名：[课程名称].md"
[完整的学习脚本内容]

请始终用中文回复。""",
            model_client_stream=True,  # Enable streaming tokens.
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
8. 检查每个任务是否包含明确的学习点（Learning Points）
9. 验证学习点是否有具体的学习目标和评估标准
10. 确认整个学习脚本有清晰的学习路径（Learning Path）
11. 验证学习路径是否体现从基础到高级的递进关系
12. 检查每个任务是否包含完整的结构（学习点、学习目标、操作步骤、评估标准等）
13. 确保学习脚本开头有整体学习目标与路径说明

评审标准：
- 学习点必须具体明确，能够指导学习者掌握特定知识或技能
- 学习路径必须逻辑清晰，体现知识点之间的递进关系
- 每个任务必须有可衡量的评估标准
- 学习脚本必须结构完整，便于后续交互式教学使用

如果发现任何不符合要求的地方，请明确提出具体的修改意见。

请始终用中文回复，并以非常严格和挑剔的态度进行审核。""",
            model_client_stream=True,  # Enable streaming tokens.
        )


async def create_teaching_team(model_client):
    """创建教学团队"""
    # 创建各个Agent
    file_handler_agent = FileHandlerAgent(model_client)
    course_generator_agent = CourseGeneratorAgent(model_client)
    curriculum_director_agent = CurriculumDirectorAgent(model_client)
    student_agent = StudentAgent(model_client)
    user_proxy = UserProxyAgent(
        "user",
        input_func=input  # 使用input函数获取用户输入
    )
    
    # 定义终止条件 - 当教研组负责人批准时终止
    termination_condition = TextMentionTermination("APPROVE")
    
    # 创建团队，使用MagenticOneGroupChat
    team = MagenticOneGroupChat(
        [user_proxy, file_handler_agent, course_generator_agent, 
         curriculum_director_agent, student_agent],
        model_client=model_client,
        termination_condition=termination_condition,
        max_turns=5000  # 设置最大轮次以防止无限循环
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
                'json_output': False,  # 修改为False以确保一致性
                'structured_output': False,  # 添加缺失的structured_output字段
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
                    'json_output': False,  # 修改为False以确保一致性
                    'structured_output': False,  # 添加缺失的structured_output字段
                    'family': "gemma",
                },
                options={
                    'num_ctx': int(os.getenv("NUM_CTX", "60000")),
                    'stream': True,  # 开启流式输出
                }
            )
        else:
            model_client = OpenAIChatCompletionClient(
                model="glm-4.5", #"glm-4.5",
                api_key=api_key,
                base_url=base_url,
                model_info={
                    'vision': False,
                    'function_calling': True,
                    'json_output': True,  # 修改为False以确保一致性
                    'structured_output': False,  # 添加缺失的structured_output字段
                    'family': "glm",
                },
                options={
                    'num_ctx': int(os.getenv("NUM_CTX", "10000")),
                    'stream': True,  # 开启流式输出
                    'thinking': {"type": "disabled"},
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
                'json_output': False,  # 修改为False以确保一致性
                'structured_output': False,  # 添加缺失的structured_output字段
                'family': "gemma",
            },
            options={
                'num_ctx': int(os.getenv("NUM_CTX", "60000")),
                'stream': True,  # 开启流式输出
            }
        )
        print("已选择 gemma3:27b 模型")
    
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
        task = f"""注意全部使用中文进行讨论！用户需要生成一个关于Prompt Engineering的沉浸式学习脚本。
        
**测试阶段特殊要求：整个课程的总学时不得超过30分钟**

请按以下严格的工作流程进行：
1. 课程生成器向文件处理器请求读取 {default_file_path} 文件的内容
2. 学生代理基于其系统消息中定义的学生画像参与讨论
3. 所有团队成员（课程生成器、教研组负责人、学生）基于明确的学生画像进行激烈讨论
4. 课程生成器基于讨论结果生成课程
5. 教研组负责人和学生代理对课程内容进行评审
6. 多次迭代讨论和修改，直到教研组负责人通过
7. 课程生成器请求文件处理器将最终生成的课程脚本保存为文件

工作要求：
- 讨论必须激烈且具有建设性
- 每个代理都必须积极参与讨论
- 学生代理必须坚持自己的画像需求
- 教研组负责人必须保持极高的专业标准
- 课程生成器必须根据讨论结果不断改进课程
- 文件处理器只在收到其他代理的明确请求时才执行文件操作
- **所有代理都必须确保最终生成的课程总时长不超过30分钟**

教学脚本必须满足以下五项要求：
1. 读取教学内容的文件
2. 根据教学内容，生成沉浸式学习的教学脚本，该脚本需要能够指导教学助手一步一步的和学生进行交互式学习
3. 教学脚本的要求：每个知识点的教学过程不要超过5分钟，要让学生通过"做中学"完成知识点的学习。在教学过程的最后，要根据学生的表现情况，给出基于选择题的小测验，测验时间不要超过10分钟。最后给出针对学生的全面的评估认证报告结果
4. 对于教学脚本要经过多轮的挑剔的讨论，学生评审员和教研组长要积极参与讨论，并给出挑剔和明确的修改意见
5. 最后完成讨论后，要对最终的教学脚本进行汇总润色，最后保存成文件

请开始执行任务。"""
        
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