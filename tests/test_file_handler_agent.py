#!/usr/bin/env python3
"""
专门测试FileHandlerAgent的工具调用功能
"""

import asyncio
import os
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

# 添加src目录到路径以便导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from autogen_core.models import SystemMessage, UserMessage
from autogen_ext.models.ollama import OllamaChatCompletionClient
from teaching_team import FileHandlerAgent


class TestFileHandlerAgentTools(unittest.TestCase):
    """测试FileHandlerAgent的工具调用功能"""

    def setUp(self):
        """测试初始化"""
        # 创建模拟的模型客户端
        self.model_client = AsyncMock()
        self.agent = FileHandlerAgent(self.model_client)
        
        # 创建临时目录用于测试
        self.temp_dir = tempfile.mkdtemp()
        self.agent._base_path = self.temp_dir

    def tearDown(self):
        """测试清理"""
        # 清理临时目录
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)

    def test_save_content_to_file_tool_call(self):
        """测试save_content_to_file工具调用"""
        # 模拟工具调用
        async def test_save():
            result = await self.agent.save_content_to_file(
                content="这是测试内容", 
                filename="test.txt"
            )
            return result
        
        # 运行异步测试
        result = asyncio.run(test_save())
        
        # 验证结果
        self.assertIn("内容已成功保存到文件", result)
        self.assertIn("test.txt", result)
        
        # 验证文件确实被创建
        file_path = os.path.join(self.temp_dir, "test.txt")
        self.assertTrue(os.path.exists(file_path))
        
        # 验证文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            self.assertEqual(content, "这是测试内容")

    def test_read_file_content_tool_call(self):
        """测试read_file_content工具调用"""
        # 先创建一个测试文件
        test_file_path = os.path.join(self.temp_dir, "read_test.txt")
        with open(test_file_path, 'w', encoding='utf-8') as f:
            f.write("这是读取测试内容")
        
        # 模拟工具调用
        async def test_read():
            result = await self.agent.read_file_content("read_test.txt")
            return result
        
        # 运行异步测试
        result = asyncio.run(test_read())
        
        # 验证结果
        self.assertEqual(result, "这是读取测试内容")

    def test_read_nonexistent_file(self):
        """测试读取不存在的文件"""
        async def test_read():
            try:
                await self.agent.read_file_content("nonexistent.txt")
                return False  # 不应该到达这里
            except FileNotFoundError:
                return True  # 正确捕获异常
        
        # 运行异步测试
        result = asyncio.run(test_read())
        
        # 验证结果
        self.assertTrue(result, "应该抛出FileNotFoundError异常")

    def test_save_content_with_auto_extension(self):
        """测试保存文件时自动添加扩展名"""
        async def test_save():
            result = await self.agent.save_content_to_file(
                content="测试内容", 
                filename="test_file"  # 没有扩展名
            )
            return result
        
        # 运行异步测试
        result = asyncio.run(test_save())
        
        # 验证结果
        self.assertIn("内容已成功保存到文件", result)
        self.assertIn("test_file.md", result)
        
        # 验证文件确实被创建
        file_path = os.path.join(self.temp_dir, "test_file.md")
        self.assertTrue(os.path.exists(file_path))


class TestFileHandlerAgentMessages(unittest.TestCase):
    """测试FileHandlerAgent的消息处理"""

    def setUp(self):
        """测试初始化"""
        self.model_client = AsyncMock()
        self.agent = FileHandlerAgent(self.model_client)

    def test_system_message_content(self):
        """测试系统消息内容"""
        # 验证系统消息包含关键信息
        self.assertIn("文件处理助手", self.agent._system_message)
        self.assertIn("读取和保存本地文件内容", self.agent._system_message)
        self.assertIn("严格按照请求执行操作", self.agent._system_message)


if __name__ == "__main__":
    unittest.main()