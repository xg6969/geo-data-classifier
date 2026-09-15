import fitz  # PyMuPDF
import json
import os
import logging
from pydantic import BaseModel, Field
from typing import List
from openai import OpenAI

logger = logging.getLogger("DataClassifierEngine")

# 从环境变量获取，如果本地没有设置则回退到空字符串或其他默认处理
api_key = os.getenv("ZHIPU_API_KEY", "您的备用处理逻辑")

client = OpenAI(
    api_key=api_key,
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)


# ================= 1. Pydantic 结构化输出模型 (对齐 Excel 数据元结构) =================
class LiteratureMetadata(BaseModel):
    title: str = Field(..., description="文献标题")
    doc_type: str = Field(..., description="文献类型")
    business_domain: str = Field(..., description="业务域")
    business_activity: str = Field(..., description="业务活动")
    object_type: str = Field(..., description="业务对象类型")
    object_name: List[str] = Field(..., description="业务对象名称")
    resource_type: List[str] = Field(..., description="资源类型")

    # 根据附表新增的可选字段
    subject_classification: str = Field("", description="学科分类")
    authors: str = Field("", description="作者/发明人")
    publish_year: str = Field("", description="出版年份")
    publication_name: str = Field("", description="出版物名称")
    security_level: str = Field("公开", description="数据安全等级")


# ================= 2. 文本解析函数 (保持不变) =================
def extract_literature_head(file_path: str, max_pages: int = 2) -> str:
    text = ""
    try:
        with fitz.open(file_path) as doc:
            for i in range(min(max_pages, len(doc))):
                text += doc[i].get_text() + "\n"
    except Exception as e:
        logger.error(f"PDF解析失败: {str(e)}")
        raise ValueError(f"PDF解析失败: {str(e)}")
    return text[:3000].strip()


# ================= 3. LLM 抽取逻辑 (扩充字段与对齐标准值) =================
def extract_metadata_via_llm(text_content: str) -> dict:
    if not text_content:
        raise ValueError("提取的文本为空，无法进行智能解析")

    prompt = f"""
        你是一个专业的非常规油气地质数据分析专家。请阅读以下文献开头，提取指定的业务元数据。

        提取标准值选项：
        1. 【标题】：提取文献的官方完整标题。如果文献同时包含中英文标题，请务必优先提取中文标题。
        2. 【类型】：绝对单选。只能从 [期刊论文, 会议论文, 学位论文, 政府报告, 专利, 专著, 技术标准, 其他] 中输出唯一一个。
        3. 【业务域】：绝对单选。只能从 [盆地评价, 区带评价, 甜点评价, 区块评价, 合作环境评价, 经济评价, 其他] 中输出唯一一个。
        4. 【业务活动】：绝对单选。仔细阅读摘要，从 [盆地概况, 构造稳定性, 沉积特征, 烃源岩条件, 资源禀赋评价, 储层条件评价, 保存条件评价, 单井EUR预测, 其他] 中挑选唯一最相关的一个，严禁输出多个。
        5. 【业务对象类型】：绝对单选。只能从 [国家, 盆地, 区带, 甜点, 工区, 井, 项目, 区块, 露头] 中输出唯一一个。
        6. 【业务对象名称】：优先提取文献中具体论述的地质单元（如具体的盆地名称）。即使只有一个也必须输出为数组（如 ["鄂尔多斯盆地"]）。
        7. 【资源类型】：从 [煤层气, 页岩气, 页岩油, 致密气, 致密油, 常规油气, 天然气水合物] 中提取，必须输出为数组，可多选。
        8. 【学科分类】：绝对单选。只能从 [基础地质, 地球物理, 测录井, 岩石物理, 地球化学, 储层物性分析, 其他] 中输出唯一一个。
        9. 【作者/发明人】：提取作者名称，多个用逗号分隔。如果是中文文献，请务必提取中文姓名。
        10. 【出版物名称】：如期刊名、会议名或大学名称。优先提取中文。
        11. 【出版年份】：提取四位数字年份，如 2023。
        12. 【数据安全等级】：绝对单选。从 [公开, 内部, 敏感, 重要, 核心] 中选择（公开发表的期刊文献统一填“公开”）。

        文献内容：
        {text_content}

        请严格以 JSON 格式输出，不要带有```json标记。必须包含上述 12 个属性对应的英文字段：
        title, doc_type, business_domain, business_activity, object_type, object_name, resource_type, subject_classification, authors, publication_name, publish_year, security_level
        """

    try:
        response = client.chat.completions.create(
            model="glm-4-flash",
            messages=[
                {"role": "system", "content": "你是一个严格输出JSON的AI助手。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        # =========== 新增：实时打印本次调用的 Token 消耗 ===========
        if hasattr(response, 'usage') and response.usage:
            logger.info(
                f"【API 计费监控】本次调用成功！共消耗 Token: {response.usage.total_tokens} (提示词: {response.usage.prompt_tokens}, 生成: {response.usage.completion_tokens})")
        # ========================================================

        raw_content = response.choices[0].message.content.strip()
        if raw_content.startswith("```json"):
            raw_content = raw_content[7:-3].strip()
        elif raw_content.startswith("```"):
            raw_content = raw_content[3:-3].strip()

        data = json.loads(raw_content)

        # 强制格式清洗防错
        str_fields = ["title", "doc_type", "business_domain", "business_activity", "object_type",
                      "subject_classification", "authors", "publication_name", "publish_year", "security_level"]
        for field in str_fields:
            val = data.get(field)
            if isinstance(val, list):
                data[field] = str(val[0]).strip() if len(val) > 0 else ""
            elif val is None:
                data[field] = ""
        # ================= 新增：业务对象类型 白名单强制拦截 =================
        valid_obj_types = ["国家", "盆地", "区带", "甜点", "工区", "井", "项目", "区块", "露头"]
        if data.get("object_type") not in valid_obj_types:
            # 如果大模型胡编乱造，直接强制设为空，由人工在前端下拉框补全
            data["object_type"] = ""

        list_fields = ["object_name", "resource_type"]
        for field in list_fields:
            val = data.get(field)
            if isinstance(val, str):
                data[field] = [val] if val.strip() else []
            elif val is None:
                data[field] = []

        return data

    except json.JSONDecodeError:
        logger.error(f"JSON解析失败: {raw_content}")
        raise ValueError("模型解析结果格式错误")
    except Exception as e:
        logger.error(f"大模型调用失败: {str(e)}")
        raise ValueError(f"大模型调用失败: {str(e)}")