from baidu.baidu_api import get_completion_baidu
from myopenai.openai_api import get_completion_openai
from sparkdesk.spark_api import get_completion_spark
from zhipu.zhipu_api import get_completion_zhipu

query = """
老板拖欠我的工资不发三个月了，我该怎么办？
"""  # 示例表格数据
prompt = """
请根据以下用```符号包括的法律纠纷问题，给出分析和解决方案
query: ```{query}```
""".format(query=query)

print("temperature: 0.1")
print("system:", "你是一位精通中华人民共和国劳动法且经验丰富的律师。")
print("prompt:", prompt)
print("\n")

# ernie_4_8k_response = get_completion_baidu(prompt)  # 欠费
gpt_35_turbo = get_completion_openai(prompt)  # 尚未获得apikey
spark_v35 = get_completion_spark(prompt)
glm_4 = get_completion_zhipu(prompt)

# print("百度 ERNIE-4.0-8K: ", ernie_4_8k_response + "\n")
print("OpenAI GPT-3.5-turbo: ", gpt_35_turbo + "\n")
print("SparkDesk V3.5: ", spark_v35 + "\n")
print("ZhipuAI glm-4: ", glm_4 + "\n")
