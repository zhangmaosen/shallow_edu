#!/usr/bin/env python3
"""
测试教学团队中的各个Agent
"""

import asyncio
import os
import sys
import unittest
from unittest.mock import AsyncMock, patch, mock_open

# 添加src目录到路径以便导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from autogen_ext.models.ollama import OllamaChatCompletionClient
from teaching_team import (
    FileHandlerAgent,
    CourseGeneratorAgent,
    CurriculumDirectorAgent,
    StudentAgent
)


class TestFileHandlerAgent(unittest.TestCase):
    """测试文件处理Agent"""

    def setUp(self):
        """测试初始化"""
        # 创建模拟的模型客户端
        self.model_client = AsyncMock()
        self.agent = FileHandlerAgent(self.model_client)

    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.agent.name, "file_handler")
        self.assertIsNotNone(self.agent._base_path)

    @patch("builtins.open", new_callable=mock_open, read_data="test content")
    @patch("os.path.exists", return_value=True)
    def test_read_file_content(self, mock_exists, mock_file):
        """测试读取文件内容"""
        # 使用asyncio.run()来运行异步测试
        content = asyncio.run(self.agent.read_file_content("test.txt"))
        self.assertEqual(content, "test content")

    @patch("builtins.open", new_callable=mock_open)
    def test_save_content_to_file(self, mock_file):
        """测试保存文件内容"""
        # 使用asyncio.run()来运行异步测试
        result = asyncio.run(self.agent.save_content_to_file("test content", "test.txt"))
        self.assertIn("内容已成功保存到文件", result)
        mock_file.assert_called_once_with(
            os.path.join(self.agent._base_path, "test.txt"), 
            'w', 
            encoding='utf-8'
        )


class TestCourseGeneratorAgent(unittest.TestCase):
    """测试课程生成Agent"""

    def setUp(self):
        """测试初始化"""
        self.model_client = AsyncMock()
        self.agent = CourseGeneratorAgent(self.model_client)

    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.agent.name, "course_generator")


class TestCurriculumDirectorAgent(unittest.TestCase):
    """测试教研组负责人Agent"""

    def setUp(self):
        """测试初始化"""
        self.model_client = AsyncMock()
        self.agent = CurriculumDirectorAgent(self.model_client)

    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.agent.name, "curriculum_director")


class TestStudentAgent(unittest.TestCase):
    """测试学生Agent"""

    def setUp(self):
        """测试初始化"""
        self.model_client = AsyncMock()
        self.agent = StudentAgent(self.model_client)

    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.agent.name, "student")


class TestTeachingTeamIntegration(unittest.TestCase):
    """测试教学团队集成"""

    def setUp(self):
        """测试初始化"""
        self.model_client = AsyncMock()

    def test_agent_creation(self):
        """测试所有Agent创建"""
        file_handler = FileHandlerAgent(self.model_client)
        course_generator = CourseGeneratorAgent(self.model_client)
        curriculum_director = CurriculumDirectorAgent(self.model_client)
        student = StudentAgent(self.model_client)

        self.assertIsNotNone(file_handler)
        self.assertIsNotNone(course_generator)
        self.assertIsNotNone(curriculum_director)
        self.assertIsNotNone(student)


if __name__ == "__main__":
    unittest.main()