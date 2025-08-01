#!/usr/bin/env python3
"""
运行所有测试的脚本
"""

import asyncio
import os
import sys
import unittest

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def run_tests():
    """运行所有测试"""
    # 发现并运行测试
    loader = unittest.TestLoader()
    start_dir = os.path.dirname(__file__)
    suite = loader.discover(start_dir, pattern='test_*.py')

    # 添加 FileHandlerAgent 测试
    try:
        file_handler_suite = loader.loadTestsFromName('test_file_handler_agent')
        suite.addTests(file_handler_suite)
    except ImportError:
        print("警告: 无法找到 FileHandlerAgent 的测试模块 (test_file_handler_agent.py)")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # 返回测试结果
    return result.wasSuccessful()

if __name__ == "__main__":
    print("开始运行教学团队测试...")
    success = run_tests()
    if success:
        print("所有测试通过!")
        sys.exit(0)
    else:
        print("部分测试失败!")
        sys.exit(1)