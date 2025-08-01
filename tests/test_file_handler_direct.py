#!/usr/bin/env python3
"""
直接测试FileHandlerAgent的工具调用功能
模拟真实使用场景中的工具调用
"""

import asyncio
import os
import sys
import tempfile
from unittest.mock import AsyncMock, MagicMock

# 添加src目录到路径以便导入
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from teaching_team import FileHandlerAgent


async def test_file_handler_tool_calls():
    """直接测试FileHandlerAgent的工具调用"""
    print("开始直接测试FileHandlerAgent的工具调用功能...")
    
    # 创建模拟的模型客户端
    model_client = AsyncMock()
    
    # 创建FileHandlerAgent
    file_handler_agent = FileHandlerAgent(model_client)
    
    # 创建临时目录用于测试
    temp_dir = tempfile.mkdtemp()
    file_handler_agent._base_path = temp_dir
    
    print(f"使用临时目录: {temp_dir}")
    
    try:
        # 测试1: 直接调用save_content_to_file工具
        print("\n=== 测试1: 保存文件工具调用 ===")
        save_result = await file_handler_agent.save_content_to_file(
            content="这是直接测试FileHandlerAgent工具调用的内容。",
            filename="direct_test.txt"
        )
        print(f"保存结果: {save_result}")
        
        # 验证文件是否创建
        file_path = os.path.join(temp_dir, "direct_test.txt")
        if os.path.exists(file_path):
            print("✓ 文件成功创建")
            # 验证文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if content == "这是直接测试FileHandlerAgent工具调用的内容。":
                    print("✓ 文件内容正确")
                else:
                    print("✗ 文件内容不正确")
        else:
            print("✗ 文件未创建")
        
        # 测试2: 直接调用read_file_content工具
        print("\n=== 测试2: 读取文件工具调用 ===")
        # 首先确保文件存在
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write("这是用于读取测试的内容。")
        
        read_result = await file_handler_agent.read_file_content("direct_test.txt")
        print(f"读取结果: {read_result}")
        
        if read_result == "这是用于读取测试的内容。":
            print("✓ 文件读取功能正常")
        else:
            print("✗ 文件读取内容不正确")
        
        # 测试3: 读取不存在的文件
        print("\n=== 测试3: 读取不存在的文件 ===")
        try:
            await file_handler_agent.read_file_content("nonexistent.txt")
            print("✗ 应该抛出异常但没有")
        except FileNotFoundError:
            print("✓ 正确抛出FileNotFoundError异常")
        except Exception as e:
            print(f"✗ 抛出了错误类型的异常: {e}")
        
        # 测试4: 保存文件时自动添加扩展名
        print("\n=== 测试4: 自动添加扩展名 ===")
        save_result = await file_handler_agent.save_content_to_file(
            content="测试自动扩展名功能",
            filename="auto_extension_test"  # 没有扩展名
        )
        print(f"保存结果: {save_result}")
        
        # 检查.md文件是否创建
        md_file_path = os.path.join(temp_dir, "auto_extension_test.md")
        if os.path.exists(md_file_path):
            print("✓ 自动扩展名功能正常")
            with open(md_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if content == "测试自动扩展名功能":
                    print("✓ 自动扩展名文件内容正确")
                else:
                    print("✗ 自动扩展名文件内容不正确")
        else:
            print("✗ 自动扩展名文件未创建")
        
        # 测试5: 保存较长内容
        print("\n=== 测试5: 保存较长内容 ===")
        long_content = """
# 这是一个较长的测试内容

用于测试FileHandlerAgent是否能正确处理较长的文件内容。

## 包含多种格式

1. 列表项一
2. 列表项二
3. 列表项三

**粗体文本** 和 *斜体文本*

> 这是一个引用块
> 包含多行内容

```python
# 这是一个代码块
def test_function():
    print("Hello, World!")
    return True
```

### 结束部分

这是文件的结尾部分，用于确保所有内容都能被正确保存。
        """.strip()
        
        save_result = await file_handler_agent.save_content_to_file(
            content=long_content,
            filename="long_content_test.md"
        )
        print(f"保存结果: {save_result}")
        
        # 验证长内容文件
        long_file_path = os.path.join(temp_dir, "long_content_test.md")
        if os.path.exists(long_file_path):
            print("✓ 长内容文件成功创建")
            with open(long_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if content == long_content:
                    print("✓ 长内容文件内容正确")
                else:
                    print("✗ 长内容文件内容不正确")
        else:
            print("✗ 长内容文件未创建")
        
        print("\n=== 所有直接测试完成 ===")
        
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
    asyncio.run(test_file_handler_tool_calls())