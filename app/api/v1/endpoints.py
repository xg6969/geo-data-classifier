import logging
logger = logging.getLogger("api.endpoints")

# 正常业务流记录
logger.info(f"开始处理重点目标区 [{target_area}] 的压缩包文件。")

# AI推理或深度验证记录
logger.info(f"启动测井文件特征校验，提取GR与PE特征...")

# 异常记录
logger.error(f"ZIP解压失败: 文件损坏或格式不支持 - {str(e)}", exc_info=True)