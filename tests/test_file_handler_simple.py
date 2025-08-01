#!/usr/bin/env python3
"""
简单测试FileHandlerAgent的功能
"""

import asyncio
import os
import sys
import tempfile

# 添加src目录到路径以便导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from unittest.mock import AsyncMock
from teaching_team import FileHandlerAgent


async def test_file_handler_agent():
    """测试FileHandlerAgent的功能"""
    print("开始测试FileHandlerAgent...")
    
    # 创建模拟的模型客户端
    model_client = AsyncMock()
    agent = FileHandlerAgent(model_client)
    
    # 创建临时目录用于测试
    temp_dir = tempfile.mkdtemp()
    agent._base_path = temp_dir
    
    print(f"使用临时目录: {temp_dir}")
    
    try:
        # 测试保存文件功能
        print("\n1. 测试保存文件功能...")
        result = await agent.save_content_to_file(
            content="这是测试内容，用于验证FileHandlerAgent的保存功能。", 
            filename="test_save.txt"
        )
        print(f"保存结果: {result}")
        
        # 验证文件是否创建成功
        file_path = os.path.join(temp_dir, "test_save.txt")
        if os.path.exists(file_path):
            print("✓ 文件创建成功")
            # 读取并验证文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if content == "这是测试内容，用于验证FileHandlerAgent的保存功能。":
                    print("✓ 文件内容正确")
                else:
                    print("✗ 文件内容不正确")
        else:
            print("✗ 文件未创建")
        
        # 测试读取文件功能
        print("\n2. 测试读取文件功能...")
        # 先创建一个文件用于读取测试
        read_test_path = os.path.join(temp_dir, "read_test.txt")
        with open(read_test_path, 'w', encoding='utf-8') as f:
            f.write("这是读取测试内容")
        
        # 读取文件
        content = await agent.read_file_content("read_test.txt")
        print(f"读取内容: {content}")
        if content == "这是读取测试内容":
            print("✓ 文件读取功能正常")
        else:
            print("✗ 文件读取内容不正确")
        
        # 测试读取不存在的文件
        print("\n3. 测试读取不存在的文件...")
        try:
            await agent.read_file_content("nonexistent.txt")
            print("✗ 应该抛出异常但没有抛出")
        except FileNotFoundError as e:
            print(f"✓ 正确抛出异常: {e}")
        except Exception as e:
            print(f"✗ 抛出了错误类型的异常: {e}")
        
        # 测试自动添加扩展名功能
        print("\n4. 测试自动添加扩展名功能...")
        result = await agent.save_content_to_file(
            content="测试自动扩展名", 
            filename="auto_ext"  # 没有扩展名
        )
        print(f"保存结果: {result}")
        
        # 检查是否创建了.md文件
        md_file_path = os.path.join(temp_dir, "auto_ext.md")
        if os.path.exists(md_file_path):
            print("✓ 自动扩展名功能正常")
        else:
            print("✗ 自动扩展名功能异常")
        
        print("\n所有测试完成!")
        
    except Exception as e:
        print(f"测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理临时文件
        print("\n清理临时文件...")
        try:
            for file in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, file))
            os.rmdir(temp_dir)
            print("临时文件清理完成")
        except Exception as e:
            print(f"清理临时文件时出错: {e}")


if __name__ == "__main__":
    asyncio.run(test_file_handler_agent())