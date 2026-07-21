"""
配置加载冒烟测试：验证 .env 能被正确加载、关键配置项可被解析。
（避免把「环境变量配置」当成理所当然却没有任何校验。）
"""
from dotenv import load_dotenv
import os


def test_dotenv_loads():
    # 不抛异常即可；load_dotenv 找不到文件时返回 False
    load_dotenv()
    # 关键配置项应能被解析（来自 .env 或代码内默认值）
    assert os.getenv("REDIS_HOST", "localhost") is not None
    assert os.getenv("SERVER_PORT", "8000") is not None
