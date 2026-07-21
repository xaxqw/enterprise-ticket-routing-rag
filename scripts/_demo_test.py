"""一键实测脚本（仅用标准库 urllib，避免环境依赖问题）：
登录 -> 问几类代表问题 -> 打印真实回答。
用法：cd /d/RAG && python scripts/_demo_test.py
"""
import urllib.request, urllib.parse, json

BASE = "http://127.0.0.1:8000"
USER, PWD = "admin", "admin123"

def _post(url, data=None, headers=None, form=None):
    if form is not None:
        body = urllib.parse.urlencode(form).encode()
        h = {"Content-Type": "application/x-www-form-urlencoded"}
    else:
        body = json.dumps(data).encode("utf-8") if data is not None else b""
        h = {"Content-Type": "application/json"}
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, data=body, headers=h, method="POST")
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))

def login():
    # 优先尝试登录；若默认账号未 seed，则自助注册一个测试用户
    try:
        return _post(f"{BASE}/api/auth/login", form={"username": USER, "password": PWD})["access_token"]
    except urllib.error.HTTPError as e:
        if e.code == 401:
            _post(f"{BASE}/api/auth/register",
                  data={"username": USER, "password": PWD, "tenant_id": "default", "role": "user"})
            return _post(f"{BASE}/api/auth/login", form={"username": USER, "password": PWD})["access_token"]
        raise

def ask(token, q, top_k=5):
    return _post(f"{BASE}/api/rag/query",
                 data={"query": q, "session_id": "demo", "top_k": top_k},
                 headers={"Authorization": f"Bearer {token}"})

def show(title, q):
    print("\n" + "=" * 60)
    print(f"【{title}】")
    print(f"问：{q}")
    try:
        res = ask(token, q)
        print(f"答：{res.get('answer','')[:600]}")
        print(f"类型={res.get('query_type')} | 幻觉等级={res.get('hallucination_level')} | 缓存命中={res.get('cache_hit')} | 引用数={len(res.get('references',[]))}")
        if res.get("references"):
            print("首条引用：", res["references"][0].get("text", "")[:120].replace("\n", " "))
    except Exception as e:
        print("调用失败：", repr(e))

if __name__ == "__main__":
    token = login()
    print("登录成功，token 前缀：", token[:20], "...")
    show("计算题(验证工具智能体)", "帮我计算 (250 + 150) / 4 * 12 等于多少？")
    show("闲聊(验证意图路由)", "你好，你是谁？能做什么？")
    show("检索题(验证RAG)", "企业工单智能分流系统是怎么工作的？支持哪些处理方式？")
    show("追问(验证多轮记忆)", "那它的意图识别用的是模型还是规则？")
