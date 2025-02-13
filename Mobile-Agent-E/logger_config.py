# logger_config.py

import logging
import os
from logging.handlers import RotatingFileHandler

# 创建一个日志目录（如果不存在）
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# 创建日志记录器
logger = logging.getLogger("my_app_logger")
logger.setLevel(logging.DEBUG)  # 设置全局日志级别

# 定义日志格式
formatter = logging.Formatter(
    fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 创建控制台处理器并设置级别
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # 控制台显示INFO及以上级别日志
console_handler.setFormatter(formatter)

# 创建文件处理器并设置级别（使用RotatingFileHandler避免日志文件过大）
file_handler = RotatingFileHandler(
    os.path.join(LOG_DIR, 'app.log'),
    maxBytes=5*1024*1024,  # 5 MB
    backupCount=5
)
file_handler.setLevel(logging.DEBUG)  # 文件记录DEBUG及以上级别日志
file_handler.setFormatter(formatter)

# 避免重复添加处理器
if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# 可选：禁用第三方库的过多日志输出
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.WARNING)