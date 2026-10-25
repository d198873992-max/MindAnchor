# MindAnchor V0.1 - 核心记忆锚点构建脚本
import json
import networkx as nx
from openai import OpenAI

# 配置 DeepSeek API
client = OpenAI(
    api_key="你的API_KEY",
    base_url="https://api.deepseek.com"
)

# 初始化因果网络图
memory_graph = nx.DiGraph()

def extract_memory_anchors(dialogue_text):
    """调用大模型，提取对话中的因果与情感锚点"""
    prompt = f"""
    你是一个社会学与心理学分析专家。请分析以下对话，提取关键事件和记忆锚点。
    必须以严格的 JSON 格式输出，包含：
    1. entities: 涉及的实体（人物/事物）
    2. events: 发生的事件
    3. emotion_score: 情绪评分（-10 极度痛苦 到 10 极度快乐）
    4. causal_link: 这件事导致的直接后果
    对话内容：{dialogue_text}
    """
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"} # 强制JSON输出
    )
    return json.loads(response.choices[0].message.content)

def build_graph(data):
    """将数据注入本地因果图（兼容字典格式）"""
    for event in data.get('events', []):
        # 核心修复：如果大模型返回的是字典，就提取里面的文字；如果是字符串，就直接用
        if isinstance(event, dict):
            # 尝试取字典里最像名字的那个值，如果没有，就把整个字典转成字符串
            node_name = str(event.get('event') or event.get('name') or event)
        else:
            node_name = str(event)
            
        emotion = data.get('emotion_score', 0)
        
        # 加入图网络
        memory_graph.add_node(node_name, emotion=emotion)
        
        # 构建因果边（同样处理一下因果链，防止它也是字典）
        causal_link = data.get('causal_link')
        if causal_link:
            if isinstance(causal_link, dict):
                link_name = str(causal_link.get('event') or causal_link)
            else:
                link_name = str(causal_link)
            memory_graph.add_edge(node_name, link_name, relation='causes')
        
        # 核心算法：高情感锚点永久固化
        if abs(emotion) >= 7:
            print(f"🔔 [发现核心记忆锚点]: {node_name} (情绪值: {emotion})")


# --- 模拟测试 ---
if __name__ == "__main__":
    text = "今天我因为偏头痛没去学校，但我发现如果不去，我就在家写出了一个量化交易的代码。"
    print(">>> 正在解析记忆锚点...")
    
    # 1. 提取
    memory_data = extract_memory_anchors(text)
    print("提取结果:", memory_data)
    
    # 2. 构建因果图谱
    build_graph(memory_data)
    
    print("\n=== 当前你大脑的因果节点图 ===")
for node, attrs in memory_graph.nodes(data=True):
    # 使用 .get() 方法，如果这个节点没有 emotion 属性，就给个默认值 '未知'
    emotion = attrs.get('emotion', '未知') 
    print(f"节点: {node} | 情感权重: {emotion}")
