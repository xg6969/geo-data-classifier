import logging
from logging.handlers import TimedRotatingFileHandler
import os
from pathlib import Path


def setup_logging():
    # 确保日志目录存在
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # 基础格式设定
    log_format = logging.Formatter(
        fmt='%(asctime)s - [%(levelname)s] - %(name)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # 根记录器
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 1. 控制台输出
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)

    # 2. 常规日志 (每天轮转一次，保留30天)
    info_handler = TimedRotatingFileHandler(
        filename=log_dir / "app.log",
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8"
    )
    info_handler.setLevel(logging.INFO)
    info_handler.setFormatter(log_format)

    # 3. 错误日志 (单独记录 ERROR 及以上级别)
    error_handler = TimedRotatingFileHandler(
        filename=log_dir / "error.log",
        when="midnight",
        interval=1,
        backupCount=30,
        encoding="utf-8"
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(log_format)

    # 避免重复挂载 handler
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(info_handler)
        logger.addHandler(error_handler)

    return logger