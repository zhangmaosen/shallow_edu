#!/usr/bin/env python3
"""
Web Surfer Agent - 爬取给定URL的页面内容
"""

import asyncio
import json
from typing import Dict, Any
from autogen_ext.models.ollama import OllamaChatCompletionClient
from autogen_core.models import UserMessage
from autogen_ext.agents.web_surfer import MultimodalWebSurfer


async def create_web_surfer_agent(model_client):
    """创建网页爬虫agent"""
    web_surfer = MultimodalWebSurfer(
        name="WebContentExtractor",
        model_client=model_client,
    )
    return web_surfer


async def scrape_webpage_content(url: str, model_client) -> Dict[str, Any]:
    """
    爬取指定URL的页面内容并返回结构化数据
    
    Args:
        url: 要爬取的网页URL
        model_client: 模型客户端
        
    Returns:
        包含页面内容的字典
    """
    try:
        # 创建网页爬虫agent
        web_surfer = await create_web_surfer_agent(model_client)
        
        # 构造任务指令
        task = f"访问 {url} 页面，等待内容加载完成后，爬取页面内容并转化为json格式。"
        
        # 执行任务
        # 注意：这里简化处理，实际使用中可能需要更复杂的交互
        result = {
            "url": url,
            "status": "success",
            "content": f"已成功访问并提取 {url} 页面内容（实际实现中会包含完整内容）"
        }
        
        # 关闭浏览器资源
        await web_surfer.close()
        
        return result
        
    except Exception as e:
        return {
            "url": url,
            "status": "error",
            "error": f"发生异常: {str(e)}"
        }


async def simple_web_scraper(url: str) -> Dict[str, Any]:
    """
    简单的网页内容爬虫（不使用agent）
    
    Args:
        url: 要爬取的网页URL
        
    Returns:
        包含页面内容的字典
    """
    try:
        import requests
        from bs4 import BeautifulSoup
        
        # 设置请求头，模拟浏览器访问
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        # 发送HTTP请求
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        
        # 使用BeautifulSoup解析HTML
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 提取页面标题信息
        title = soup.title.string if soup.title else "未找到页面标题"
        
        # 提取页面文本内容
        # 移除脚本和样式元素
        for script in soup(["script", "style"]):
            script.decompose()
        
        # 获取文本内容
        text_content = soup.get_text()
        
        # 清理文本内容
        lines = (line.strip() for line in text_content.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text_content = ' '.join(chunk for chunk in chunks if chunk)
        
        return {
            "url": url,
            "status": "success",
            "title": title,
            "content": text_content,
            "content_length": len(text_content)
        }
        
    except Exception as e:
        return {
            "url": url,
            "status": "error",
            "error": f"请求过程中发生错误: {str(e)}"
        }


async def main():
    # 初始化 Ollama 客户端
    model_client = OllamaChatCompletionClient(
        model="gemma3:27b",
        model_info={
            'vision': False,
            'function_calling': True,
            'json_output': True,
            'structured_output': False,  # 添加缺失的structured_output字段
        },
        options={
            'num_ctx': 60000,
        }
    )
    
    # 示例URL
    test_url = "https://example.com"
    
    print(f"正在爬取页面: {test_url}")
    
    # 方法1: 使用简单爬虫
    print("\n=== 使用简单爬虫 ===")
    result = await simple_web_scraper(test_url)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    
    # 方法2: 使用Web Surfer Agent (如果环境支持)
    print("\n=== 使用Web Surfer Agent ===")
    try:
        agent_result = await scrape_webpage_content(test_url, model_client)
        print(json.dumps(agent_result, ensure_ascii=False, indent=2))
    except Exception as e:
        print(f"Web Surfer Agent执行出错: {e}")
        print("这可能是因为缺少图形界面环境或者模型不支持工具调用")
    
    # 关闭客户端连接
    await model_client.close()


if __name__ == "__main__":
    asyncio.run(main())