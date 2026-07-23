"""
快速验证脚本：一键检查启动依赖 + 服务状态。
用法: python verify_startup.py
"""
import subprocess
import sys
from pathlib import Path

# ─── 0. 环境路径 ──────────────────────────────────────
PROJECT_DIR = Path(r"D:\Projects\personal_rag_platform")
ENV_FILE = PROJECT_DIR / ".env"
MAIN_PY = PROJECT_DIR / "src" / "main.py"
CONFIG_PY = PROJECT_DIR / "src" / "config.py"

def check(condition: bool, msg_ok: str, msg_fail: str):
    if condition:
        print(f"✅ {msg_ok}")
        return True
    else:
        print(f"❌ {msg_fail}")
        return False

# ─── 1. 配置文件检查 ──────────────────────────────────
print("═══ [0] 配置文件检查 ═══")
ok = True

ok &= check(CONFIG_PY.exists(), "config.py 存在", "config.py 不存在")
ok &= check(ENV_FILE.exists(), ".env 文件已配置", "❌ .env 未配置（cp .env.example .env）")

# 检查 IP 泄露：src/config.py 源码中不包含真实内网 IP
if CONFIG_PY.exists():
    src = CONFIG_PY.read_text()
    real_ips = [l.strip() for l in src.splitlines() if "192.168.3" in l and not l.strip().startswith("#")]
    if real_ips:
        print("⚠️  config.py 源码中仍有真实 IP（非注释行）：")
        for ip in real_ips:
            print(f"   {ip}")
        ok = False
    else:
        print("✅ config.py 已清理，无真实内网 IP 泄露")

if not ok:
    sys.exit(1)

# ─── 2. Python 依赖检查 ──────────────────────────────
print("\n═══ [1] Python 依赖检查 ═══")
deps_ok = True

required_packages = ["fastapi", "pydantic_settings", "ollama", "qdrant_client", "langchain"]
for pkg in required_packages:
    try:
        __import__(pkg.replace("-", "_"))
        print(f"✅ {pkg} installed")
    except ImportError:
        print(f"❌ {pkg} 未安装 → uv pip install -e .")
        deps_ok = False

if not deps_ok:
    print("\n→ 运行此命令安装：uv pip install -e .")

# ─── 3. 基础设施检查 ──────────────────────────────────
print("\n═══ [2] 基础设施检查 ═══")

import socket

def check_port(host, port, name):
    try:
        s = socket.create_connection((host, port), timeout=3)
        s.close()
        print(f"✅ {name}: {host}:{port} 可达")
        return True
    except Exception as e:
        print(f"❌ {name}: {host}:{port} 不可达 ({e})")
        return False

# 从 .env 动态读取配置
if ENV_FILE.exists():
    env_conf = {}
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            env_conf[k] = v

host_qdrant = env_conf.get("VECTOR_DB_HOST", "127.0.0.1")
port_qdrant = int(env_conf.get("VECTOR_DB_PORT", "6333"))
host_ollama = env_conf.get("INFERENCE_HOST", "192.168.3.200")
port_ollama = int(env_conf.get("INFERENCE_PORT", "11434"))

infrastructure_ok = True
infrastructure_ok &= check_port(host_qdrant, port_qdrant, "Qdrant (向量库)")
infrastructure_ok &= check_port(host_ollama, port_ollama, "Ollama (推理池)")

# ─── 4. main.py 导入检查 ──────────────────────────────
print("\n═══ [3] FastAPI app 导入检查 ═══")

import ast
if MAIN_PY.exists():
    try:
        src_code = MAIN_PY.read_text().splitlines()
        has_app = any("FastAPI()" in l or "app = FastAPI" in l for l in src_code)
        if has_app:
            print("✅ main.py 包含 FastAPI app")
        else:
            print("⚠️  main.py 未发现标准的 FastAPI(app) 实例定义")
    except Exception as e:
        print(f"⚠️  无法解析 main.py: {e}")
else:
    print("❌ main.py 不存在")

# ─── 5. 启动建议 ──────────────────────────────────────
print("\n═══ [4] 下一步 ═══")
if infrastructure_ok and deps_ok:
    print("""
✅ 所有检查通过！可以启动服务：

    cd D:\\Projects\\personal_rag_platform
    uv run python src/main.py

启动后访问 Swagger UI：
    http://localhost:8000/docs

或运行集成测试（可选）：
    curl http://localhost:8000/ping
""")
else:
    print("""
⚠️  以上有 ❌ 标红的项目需要先解决，再启动服务。
最常见是缺少 .env 文件或 Qdrant/Ollama 未启动。
""")

# ─── Summary ──────────────────────────────────────────
all_ok = ok and infrastructure_ok and deps_ok
print(f"\n{'='*40}")
if all_ok:
    print("✅ 所有依赖就绪，可以启动了！")
else:
    print("❌ 有项未通过，请根据上面的检查列表先修复。")
print(f"{'='*40}")
