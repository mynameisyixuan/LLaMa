from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import io
import logging
from typing import List, Dict, Optional
import os

# ==================== 配置 ====================
MODEL_PATH = "/root/autodl-tmp/atom_model"
app = FastAPI()

# 允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 日志配置
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 全局变量存放模型
model = None
tokenizer = None

# ==================== 第一层：三分类 ====================
LEVEL1_CATEGORIES = ["企业数据", "个人信息", "公共数据"]

# 第一层少样本示例
LEVEL1_EXAMPLES = [
    # 企业数据示例
    {"field": "财务预算需求", "category": "企业数据"},
    {"field": "风险预警处置", "category": "企业数据"},
    {"field": "直流运维换流变", "category": "企业数据"},
    {"field": "居民生活用电", "category": "企业数据"},
    {"field": "工程前期管理工程需求", "category": "企业数据"},
    {"field": "采购目录物资名称", "category": "企业数据"},
    {"field": "岗位设置岗级", "category": "企业数据"},
    {"field": "调度运行发电", "category": "企业数据"},
    {"field": "市场成员购电管理", "category": "企业数据"},
    {"field": "信息技术名称", "category": "企业数据"},
    {"field": "电网业务长期发展规划", "category": "企业数据"},
    # 个人信息示例
    {"field": "员工手机号", "category": "个人信息"},
    {"field": "供应商联系人", "category": "个人信息"},
    {"field": "客户身份证号", "category": "个人信息"},
    {"field": "个人银行账号", "category": "个人信息"},
    {"field": "家庭住址", "category": "个人信息"},
    # 公共数据示例
    {"field": "用户日用电量", "category": "公共数据"},
    {"field": "电厂发电量", "category": "公共数据"},
    {"field": "日前交易电价", "category": "公共数据"},
    {"field": "全社会用电量", "category": "公共数据"},
    {"field": "统调发电量", "category": "公共数据"},
]

def build_prompt_level1(field_name):
    categories_text = "、".join(LEVEL1_CATEGORIES)
    examples_text = "\n".join([f"字段：{ex['field']}\n类别：{ex['category']}" for ex in LEVEL1_EXAMPLES])
    prompt = f"""你是一个数据分类专家。请将给定的字段名归入以下三大类之一：

1. **企业数据**：指电力公司内部管理、业务流程、资源配置、生产经营相关的数据，包括但不限于：
   - 发展规划、电网规划、基建项目、投资计划、财务预算、物资采购、人力资源、调度运行、交易管理、安监应急、设备运维、营销服务等所有与公司运营直接相关的数据。
   - **特别注意**：即使涉及“电网”、“输电”、“配电”、“设备”、“风险”、“预警”、“检修”、“运行”等关键词，只要属于企业内部业务（如设备检修计划、电网运行方式、风险预警处置、直流运维、变电检测等），仍归为企业数据。
   - 企业账户信息（如账户基本信息、账户金额、开户银行）也属企业数据，不属个人信息。
   - 关键词示例：规划、计划、方案、项目、需求、报表、管理、考核、预算、资金、资产、合同、制度、建设、检修、运行、分析、投资、指标、财务、账户、票据、结算、编码、名称、责任人、单位、风险、预警、设备、线路、变电站、换流站等。

2. **个人信息**：指与自然人身份、财产、健康、通信、行踪等相关，可识别特定个人的数据。必须是个人（员工、客户、供应商等自然人）的信息，不包括企业或组织对公账户。
   - 关键词示例：姓名、手机号、身份证、银行账号（个人）、地址、证件号、公积金、社保、病历、通话记录、位置、健康、婚姻、信仰等。
   - 注意：企业法人统一社会信用代码、企业账户、项目责任人（作为岗位角色）等不属个人信息。

3. **公共数据**：指电力行业基础运行数据，涉及全社会用电、发电、交易等，不包含个人隐私，且不属于特定企业内部管理。
   - 包括：全社会用电量、全社会发电量、全网负荷、跨省区交易电量、电力供需平衡信息、宏观电价水平、节能减排指标（如供电煤耗、线损率）等。
   - **特别注意**：公共数据通常冠以“全社会”、“全口径”、“全网”、“年度”、“季度”等宏观范围词，或是反映行业整体情况的统计指标。而企业内部设备检修、运行方式、风险预警等均不属于公共数据。
   - 关键词示例：全社会用电量、发电量、电价、负荷、装机容量、交易电量、供电可靠率、电压合格率、能耗、预测、供需、线损等。

请严格按以上定义判断，只输出类别名称（企业数据 / 个人信息 / 公共数据），不要输出其他内容。

字段：{field_name}
类别："""
    return prompt

# 第一层后处理规则
def post_process_level1(field_name, predicted_label):
    # 强企业数据特征词
    enterprise_keywords = [
        "规划", "计划", "方案", "项目", "需求", "报表", "管理", "考核",
        "预算", "资金", "资产", "合同", "制度", "建设", "检修", "运行",
        "分析", "投资", "指标", "财务", "账户", "票据", "结算", "编码",
        "名称", "责任人", "单位", "风险", "预警", "设备", "线路", "变电站",
        "换流站", "工单", "运维", "检测", "验收", "评价", "定级", "排查",
        "治理", "演练", "应急", "抢修", "维护", "发票", "回单", "凭证",
        "账簿", "报表", "税务", "税率", "税种", "纳税", "扣缴", "购电",
        "售电", "输电", "交易", "结算", "电价", "容量", "负荷", "发电",
        "用电", "用户", "市场", "成员", "主体", "合同", "协议"
    ]
    # 强公共数据特征词
    public_keywords = [
        "全社会", "全口径", "全网", "年度", "季度", "用电量", "发电量",
        "电价", "负荷", "装机容量", "线损", "供需", "预测", "能耗",
        "供电可靠率", "电压合格率", "节能减排", "煤耗", "厂用电率"
    ]
    # 强个人信息特征词
    strong_personal_keywords = [
        "姓名", "手机号", "身份证", "护照", "出生日期", "民族", "政治面貌",
        "宗教信仰", "生物识别", "指纹", "虹膜", "基因", "病历", "健康",
        "婚姻", "家庭住址", "通讯地址", "社保", "公积金", "工资", "奖金",
        "报销明细", "个人征信", "信用记录"
    ]
    # 强企业数据优先（但若包含强个人词则归为个人信息）
    if any(kw in field_name for kw in enterprise_keywords):
        if any(kw in field_name for kw in strong_personal_keywords):
            return "个人信息"
        return "企业数据"
    # 强公共数据优先
    if any(kw in field_name for kw in public_keywords) and predicted_label == "企业数据":
        return "公共数据"
    # 强个人信息
    if any(kw in field_name for kw in strong_personal_keywords):
        return "个人信息"
    return predicted_label

# ==================== 第二层：企业数据细分类 ====================
ENTERPRISE_SUB_CATEGORIES = [
    "发展管理", "财务管理", "安监管理", "设备管理", "营销管理",
    "基建管理", "物资管理", "人资管理", "调度管理", "交易管理", "综合管理"
]

# 增强企业数据示例
ENTERPRISE_EXAMPLES = [
    {"field": "电网业务长期发展规划", "category": "发展管理"},
    {"field": "输电网项目规划", "category": "发展管理"},
    {"field": "财务预算需求", "category": "财务管理"},
    {"field": "风险预警处置", "category": "安监管理"},
    {"field": "直流运维换流变", "category": "设备管理"},
    {"field": "居民生活用电", "category": "营销管理"},
    {"field": "工程前期管理工程需求", "category": "基建管理"},
    {"field": "采购目录物资名称", "category": "物资管理"},
    {"field": "岗位设置岗级", "category": "人资管理"},
    {"field": "调度运行发电", "category": "调度管理"},
    {"field": "市场成员购电管理", "category": "交易管理"},
    {"field": "信息技术名称", "category": "综合管理"},
    # 新增示例
    {"field": "线损指标", "category": "设备管理"},
    {"field": "线损输电", "category": "设备管理"},
    {"field": "线损变电", "category": "设备管理"},
    {"field": "线损配电", "category": "设备管理"},
    {"field": "线损营销", "category": "营销管理"},
    {"field": "节能减排供电煤耗", "category": "设备管理"},
    {"field": "节能减排厂用电率", "category": "设备管理"},
    {"field": "节能减排发电水耗", "category": "设备管理"},
    {"field": "二氧化硫排放总量", "category": "设备管理"},
    {"field": "年度生产经营计划", "category": "发展管理"},
    {"field": "生产经营计划下达", "category": "发展管理"},
    {"field": "生产经营计划调整", "category": "发展管理"},
    {"field": "生产经营计划分析", "category": "发展管理"},
    {"field": "考核考评信息", "category": "综合管理"},
    {"field": "账户基本信息", "category": "财务管理"},
    {"field": "账户金额", "category": "财务管理"},
    {"field": "账户开立变更", "category": "财务管理"},
    {"field": "账户管理台账", "category": "财务管理"},
    {"field": "收付款结算收款凭证", "category": "财务管理"},
    {"field": "收付款结算付款凭证", "category": "财务管理"},
    {"field": "收付款结算结算凭证", "category": "财务管理"},
    {"field": "票据本票", "category": "财务管理"},
    {"field": "票据汇票", "category": "财务管理"},
    {"field": "票据支票凭证", "category": "财务管理"},
    {"field": "票据使用登记台账", "category": "财务管理"},
    {"field": "票据盘点记录", "category": "财务管理"},
    {"field": "票据存根登记", "category": "财务管理"},
    {"field": "融资需求", "category": "财务管理"},
    {"field": "融资计划", "category": "财务管理"},
    {"field": "融资合同", "category": "财务管理"},
    {"field": "生产资金支出", "category": "财务管理"},
    {"field": "资金运作资金调拨", "category": "财务管理"},
    {"field": "资金异动信息", "category": "财务管理"},
    {"field": "资金支付记录", "category": "财务管理"},
    {"field": "会计核算收入", "category": "财务管理"},
    {"field": "会计核算支出", "category": "财务管理"},
    {"field": "会计核算费用", "category": "财务管理"},
    {"field": "会计核算成本核算", "category": "财务管理"},
    {"field": "报表资产负债表", "category": "财务管理"},
    {"field": "报表利润表", "category": "财务管理"},
    {"field": "报表现金流量表", "category": "财务管理"},
    {"field": "所有者权益变动表", "category": "财务管理"},
    {"field": "电子凭证电子发票", "category": "财务管理"},
    {"field": "电子凭证电子客票", "category": "营销管理"},
    {"field": "电子凭证电子行程单", "category": "综合管理"},
    {"field": "银行电子回单", "category": "财务管理"},
    {"field": "报账审核", "category": "财务管理"},
    {"field": "报账结算", "category": "财务管理"},
    {"field": "报账汇总", "category": "财务管理"},
    {"field": "发票编号", "category": "财务管理"},
    {"field": "发票客户名称", "category": "财务管理"},
    {"field": "发票开票金额", "category": "财务管理"},
    {"field": "发票开票日期", "category": "财务管理"},
    {"field": "发票开票单位", "category": "财务管理"},
    {"field": "收付款凭证明细", "category": "财务管理"},
    {"field": "税务申报申报表", "category": "财务管理"},
]

def build_prompt_enterprise(field_name):
    categories_text = "、".join(ENTERPRISE_SUB_CATEGORIES)
    examples_text = "\n".join([f"字段：{ex['field']}\n类别：{ex['category']}" for ex in ENTERPRISE_EXAMPLES])
    prompt = f"""你是一个电力行业数据分类专家。请将给定的企业数据字段名归入以下业务类别之一：

{examples_text}

现在请判断：
字段：{field_name}
类别："""
    return prompt

# ==================== 第二层：个人信息细分类 ====================
PERSONAL_SUB_CATEGORIES = ["内部员工信息", "供应商个人信息", "客户个人信息"]

# 增强个人信息示例
PERSONAL_EXAMPLES = [
    {"field": "员工手机号", "category": "内部员工信息"},
    {"field": "员工身份证号", "category": "内部员工信息"},
    {"field": "员工姓名", "category": "内部员工信息"},
    {"field": "供应商联系人", "category": "供应商个人信息"},
    {"field": "供应商银行账号", "category": "供应商个人信息"},
    {"field": "客户身份证号", "category": "客户个人信息"},
    {"field": "客户用电地址", "category": "客户个人信息"},
    {"field": "客户姓名", "category": "客户个人信息"},
    {"field": "个人银行账号", "category": "客户个人信息"},
    {"field": "家庭住址", "category": "客户个人信息"},
    {"field": "报账单据", "category": "客户个人信息"},
    {"field": "开户银行账号", "category": "客户个人信息"},
    {"field": "纳税人识别号", "category": "客户个人信息"},
]

def build_prompt_personal(field_name):
    categories_text = "、".join(PERSONAL_SUB_CATEGORIES)
    examples_text = "\n".join([f"字段：{ex['field']}\n类别：{ex['category']}" for ex in PERSONAL_EXAMPLES])
    prompt = f"""你是一个数据分类专家。请将给定的个人信息字段名归入以下子类之一：

{examples_text}

现在请判断：
字段：{field_name}
类别："""
    return prompt

# ==================== 第二层：公共数据细分类 ====================
PUBLIC_SUB_CATEGORIES = ["用电数据", "发电数据", "电力交易数据"]

# 增强公共数据示例
PUBLIC_EXAMPLES = [
    {"field": "用户日用电量", "category": "用电数据"},
    {"field": "全社会用电量", "category": "用电数据"},
    {"field": "电厂发电量", "category": "发电数据"},
    {"field": "统调发电量", "category": "发电数据"},
    {"field": "日前交易电价", "category": "电力交易数据"},
    {"field": "中长期交易电量", "category": "电力交易数据"},
    {"field": "线损变电", "category": "用电数据"},
    {"field": "线损配电", "category": "用电数据"},
    {"field": "电源接入方式", "category": "用电数据"},
    {"field": "新能源接入系统标准", "category": "用电数据"},
    {"field": "配电网拓扑结构", "category": "用电数据"},
    {"field": "节能减排供电煤耗", "category": "发电数据"},
    {"field": "二氧化硫排放总量", "category": "发电数据"},
    {"field": "电力市场供需政策", "category": "电力交易数据"},
    {"field": "年度电力市场购电量", "category": "电力交易数据"},
    {"field": "电力市场供需售电量", "category": "电力交易数据"},
    {"field": "电力市场供需用电量需求", "category": "电力交易数据"},
    {"field": "历史及年度电力市场供需分析报告", "category": "电力交易数据"},
]

def build_prompt_public(field_name):
    categories_text = "、".join(PUBLIC_SUB_CATEGORIES)
    examples_text = "\n".join([f"字段：{ex['field']}\n类别：{ex['category']}" for ex in PUBLIC_EXAMPLES])
    prompt = f"""你是一个电力行业数据分类专家。请将给定的公共数据字段名归入以下子类之一：

{examples_text}

现在请判断：
字段：{field_name}
类别："""
    return prompt

# ==================== 通用分类函数 ====================
def classify_field(model, tokenizer, prompt, categories, temperature=0.0):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            temperature=temperature,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.eos_token_id
        )
    generated_ids = outputs[0][input_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    for cat in categories:
        if cat in generated_text:
            return cat
    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    for cat in categories:
        if cat in full_output:
            return cat
    return "解析失败"

# ==================== 第二层后处理规则 ====================
def post_process_level2(field_name, level1, level2):
    if level2 != "解析失败":
        return level2

    # 企业数据子类规则
    if level1 == "企业数据":
        if any(kw in field_name for kw in ["预算", "财务", "资金", "账户", "票据", "凭证", "发票", "回单", "税务", "报表", "核算", "成本", "利润", "现金流量", "所有者权益"]):
            return "财务管理"
        if any(kw in field_name for kw in ["检修", "运维", "设备", "线路", "变压器", "换流", "变电", "直流", "输电", "配电", "线损", "能耗", "煤耗", "厂用电", "排放"]):
            return "设备管理"
        if any(kw in field_name for kw in ["规划", "计划", "方案", "项目", "发展", "建设", "投资"]):
            return "发展管理"
        if any(kw in field_name for kw in ["风险", "预警", "应急", "安全", "隐患", "排查", "治理"]):
            return "安监管理"
        if any(kw in field_name for kw in ["营销", "客户", "用电", "售电", "市场", "电费"]):
            return "营销管理"
        if any(kw in field_name for kw in ["基建", "工程", "施工", "建设"]):
            return "基建管理"
        if any(kw in field_name for kw in ["物资", "采购", "供应", "库存", "仓储"]):
            return "物资管理"
        if any(kw in field_name for kw in ["人资", "岗位", "薪酬", "绩效", "考勤", "福利", "培训"]):
            return "人资管理"
        if any(kw in field_name for kw in ["调度", "运行", "负荷", "并网"]):
            return "调度管理"
        if any(kw in field_name for kw in ["交易", "市场", "合同", "结算", "电价"]):
            return "交易管理"
        return "综合管理"

    # 个人信息子类规则
    if level1 == "个人信息":
        if any(kw in field_name for kw in ["员工", "内部", "工号", "社保", "公积金", "薪酬"]):
            return "内部员工信息"
        if any(kw in field_name for kw in ["供应商"]):
            return "供应商个人信息"
        if any(kw in field_name for kw in ["客户", "用电客户", "用户", "居民"]):
            return "客户个人信息"
        return "客户个人信息"

    # 公共数据子类规则
    if level1 == "公共数据":
        if any(kw in field_name for kw in ["用电", "负荷", "线损", "供电可靠", "电压合格"]):
            return "用电数据"
        if any(kw in field_name for kw in ["发电", "煤耗", "装机", "厂用电", "排放"]):
            return "发电数据"
        if any(kw in field_name for kw in ["交易", "电价", "市场", "购电", "售电", "供需"]):
            return "电力交易数据"
        return "用电数据"

    return level2

# ==================== 生命周期：启动时加载模型 ====================
@app.on_event("startup")
async def startup_event():
    global model, tokenizer
    print("🚀 正在启动服务 (本地测试模式)...")

    # === 将下面这些加载真实模型的代码全部注释掉 ===
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH...)
    # device = ...
    # model = AutoModelForCausalLM.from_pretrained(...)
    # model.eval()
    # print(f"✅ 模型加载完成，设备: {model.device}")

    print("✅ 已跳过大模型加载，当前使用 Mock 假数据模式测试接口！")

# ==================== 后台任务：批量处理 ====================
async def process_batch_fields(fields: List[str], output_file: str):
    results = []
    for idx, field in enumerate(fields, 1):
        logger.info(f"处理中 ({idx}/{len(fields)}): {field}")
        # 第一层分类
        level1_prompt = build_prompt_level1(field)
        level1_raw = classify_field(model, tokenizer, level1_prompt, LEVEL1_CATEGORIES)
        level1 = post_process_level1(field, level1_raw)

        # 第二层分类
        if level1 == "企业数据":
            sub_prompt = build_prompt_enterprise(field)
            sub_categories = ENTERPRISE_SUB_CATEGORIES
        elif level1 == "个人信息":
            sub_prompt = build_prompt_personal(field)
            sub_categories = PERSONAL_SUB_CATEGORIES
        elif level1 == "公共数据":
            sub_prompt = build_prompt_public(field)
            sub_categories = PUBLIC_SUB_CATEGORIES
        else:
            level2 = "解析失败"
            results.append({"field_name": field, "main_category": level1, "sub_category": level2})
            continue

        level2_raw = classify_field(model, tokenizer, sub_prompt, sub_categories)
        level2 = post_process_level2(field, level1, level2_raw)
        results.append({"field_name": field, "main_category": level1, "sub_category": level2})

    # 写入结果文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("字段名\t第一层类别\t第二层类别\n")
        for res in results:
            f.write(f"{res['field_name']}\t{res['main_category']}\t{res['sub_category']}\n")

    logger.info(f"处理完成！结果已保存至: {output_file}")

# ==================== 接口 1：交互式单字段分类 ====================
class SingleFieldRequest(BaseModel):
    field_name: str

# @app.post("/api/classify/single")
# async def api_classify_single(request: SingleFieldRequest):
#     field = request.field_name
#     if not field:
#         raise HTTPException(status_code=400, detail="字段名不能为空")
    # ==========================================

    # 然后在分类接口里，直接返回假数据，不要调用 classify_field：
@app.post("/api/classify/single")
async def api_classify_single(request: SingleFieldRequest):
    field = request.field_name
    # === 临时假逻辑 ===
    return {
        "main_category": "企业数据 (测试)",
        "sub_category": "财务管理 (测试)"
    }
    # 第一层分类
    level1_prompt = build_prompt_level1(field)
    level1_raw = classify_field(model, tokenizer, level1_prompt, LEVEL1_CATEGORIES)
    level1 = post_process_level1(field, level1_raw)

    # 第二层分类
    if level1 == "企业数据":
        sub_prompt = build_prompt_enterprise(field)
        sub_categories = ENTERPRISE_SUB_CATEGORIES
    elif level1 == "个人信息":
        sub_prompt = build_prompt_personal(field)
        sub_categories = PERSONAL_SUB_CATEGORIES
    elif level1 == "公共数据":
        sub_prompt = build_prompt_public(field)
        sub_categories = PUBLIC_SUB_CATEGORIES
    else:
        return {"main_category": level1, "sub_category": "解析失败"}

    level2_raw = classify_field(model, tokenizer, sub_prompt, sub_categories)
    level2 = post_process_level2(field, level1, level2_raw)

    return {"main_category": level1, "sub_category": level2}

# ==================== 接口 2：批量文件分类（异步后台任务） ====================
@app.post("/api/classify/batch")
async def api_classify_batch(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    try:
        # 读取文件内容
        contents = await file.read()
        text = contents.decode('utf-8')
        fields = [line.strip() for line in text.split('\n') if line.strip()]

        if not fields:
            raise HTTPException(status_code=400, detail="上传文件为空或格式不正确")

        # 生成输出文件名
        output_file = f"batch_result_{file.filename}.txt"

        # 将任务放入后台异步执行
        background_tasks.add_task(process_batch_fields, fields, output_file)

        return {
            "message": "批量处理任务已启动，请稍后查看结果文件",
            "output_file": output_file
        }
    except Exception as e:
        logger.error(f"批量处理失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

# ==================== 接口 3：批量文件分级（BERT） ====================
@app.post("/api/grade/batch")
async def api_grade_batch(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
):
    try:
        contents = await file.read()
        text = contents.decode('utf-8')
        fields = [line.strip() for line in text.split('\n') if line.strip()]

        if not fields:
            raise HTTPException(status_code=400, detail="上传文件为空或格式不正确")

        output_file = f"grade_result_{file.filename}.txt"

        # 后台任务：接入你的 BERT 模型代码
        background_tasks.add_task(process_bert_grading, fields, output_file)

        return {
            "message": "批量分级任务已启动，请稍后查看结果文件",
            "output_file": output_file
        }
    except Exception as e:
        logger.error(f"批量分级失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

# 示例：BERT 分级后台任务（请替换为你的实际代码）
async def process_bert_grading(fields: List[str], output_file: str):
    results = []
    for field in fields:
        # 这里接入你的 BERT 模型代码
        bert_level = "3"  # 替换为你 BERT 的真实预测结果
        results.append({
            "field_name": field,
            "level": bert_level
        })

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("字段名\t分级\n")
        for res in results:
            f.write(f"{res['field_name']}\t{res['level']}\n")

    logger.info(f"分级完成！结果已保存至: {output_file}")

# 启动命令: uvicorn main:app --host 0.0.0.0 --port 8000
