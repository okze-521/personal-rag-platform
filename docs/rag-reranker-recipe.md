# RAG 链路升级：从 4 步到 5 步 — 加入 Reranker 重排序

> 写给运维出身、正在学 Python/AI 的自己：每一步都标注了**改什么、为什么、怎么验证**，照着做就行。

---

## 1. 为什么要加 Reranker？

当前链路的问题：

```
用户问题 → Embedding → Qdrant 向量搜索(Top-5) → 拼 Prompt → LLM 回答
```

向量搜索只靠**余弦相似度**排序，但它只看"整体语义"，不会判断"这个词到底有没有回答用户的问题"。  
结果：第 4、第 5 条常常是碰瓷的（词像但意思无关），送进 LLM 反而干扰回答。

Reranker 是第二道筛子：

```
用户问题 → Embedding → Qdrant 向量搜索(Top-10 宽召回) → Reranker 精排到 Top-3 → LLM
```

它的原理是 **Cross-Encoder**：把「问题 + 文档」拼在一起喂给模型打分，答非所问的直接踢掉。

---

## 2. 安装依赖

```bash
cd D:\Projects\personal_rag_platform

# 用清华镜像装 sentence-transformers（含 torch + transformers）
uv pip install sentence-transformers \
  --python .venv/Scripts/python.exe \
  -i https://pypi.tuna.tsinghua.edu.cn/simple
```

验证装好了：

```bash
.venv/Scripts/python.exe -c "from sentence_transformers import CrossEncoder; print('OK')"
```

---

## 3. 新增配置项

**文件：`src/config.py`**  
**位置：在 `RETRIES: int = 3` 下面加 3 行**

```python
# ── Reranker 重排序 ────────────────────────────
RERANKER_MODEL: str = "BAAI/bge-reranker-v2-m3"   # 模型名
RETRIEVAL_K: int = 10     # 向量检索取回 10 条（多取几条给 Reranker 挑）
RERANK_TOP_K: int = 3     # Reranker 最终保留 3 条（送 LLM 的最相关文档）
```

**三个参数的含义：**

| 参数 | 默认值 | 作用 |
|------|--------|------|
| `RERANKER_MODEL` | bge-reranker-v2-m3 | 和第二代 BGE-M3 Embedding 同系列，配合最佳 |
| `RETRIEVAL_K` | 10 | 向量搜索召回多少条候选文档（宽召回） |
| `RERANK_TOP_K` | 3 | 精排后保留多少条送 LLM（精选） |

> 调参经验：`RETRIEVAL_K` 越大越不遗漏，但 Reranker 慢（每条都要过 Cross-Encoder）。10→3 是性价比最高的比例。

---

## 4. 新建 Reranker 模块

**文件：`src/reranker.py`**（新建）

```python
"""
BGE-Reranker 重排序模块

原理:
    向量检索 (Bi-Encoder) — 速度快但精度有限，靠余弦相似度
    重排序 (Cross-Encoder)   — 慢但精细，问题+文档拼接后联合编码打分
"""

from __future__ import annotations
import logging
from typing import TYPE_CHECKING
from sentence_transformers import CrossEncoder
from src.config import settings

if TYPE_CHECKING:
    from src.vectorstores.qdrant_store import SearchHit

logger = logging.getLogger(__name__)


class Reranker:
    """Cross-Encoder 重排序器"""

    def __init__(self) -> None:
        model_name = settings.RERANKER_MODEL
        logger.info(f"Loading reranker model: {model_name} ...")
        # 优先读本地缓存，避免跨墙访问 huggingface.co
        try:
            self.model = CrossEncoder(
                model_name,
                max_length=512,
                device="cpu",
                trust_remote_code=True,
                local_files_only=True,
            )
        except Exception:
            logger.warning("本地缓存不完整，尝试从镜像下载...")
            import os
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            self.model = CrossEncoder(
                model_name,
                max_length=512,
                device="cpu",
                trust_remote_code=True,
            )
        logger.info("Reranker loaded ✓")

    def rerank(
        self,
        query: str,
        hits: list["SearchHit"],
        top_k: int | None = None,
    ) -> list["SearchHit"]:
        """重排序，返回 top_k 条最相关文档"""
        if not hits:
            return []

        top_k = top_k or settings.RERANK_TOP_K

        # ── 1. 构造 (query, doc) 文本对 ──────────
        pairs = [(query, hit.text) for hit in hits]

        # ── 2. Cross-Encoder 批量打分 ──────────────
        scores = self.model.predict(pairs)

        # ── 3. 按分数排序 ────────────────────────
        scored = list(zip(hits, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        return [hit for hit, _ in scored[:top_k]]
```

**关键设计决策：**

- `max_length=512` — 每条文档截断到 512 tokens，够 Cross-Encoder 判断相关性了，太长反而慢
- `device="cpu"` — Reranker 模型很小（~2.3GB），CPU 完全够，不抢台式机 GPU 显存
- 文本对批量打分：10 条文档 = 10 个 (问题, 文档) 对，一次 `model.predict()` 出结果

---

## 5. 改造核心链路

**文件：`src/rag_chain.py`**

### 5.1 新增 import

```python
from src.reranker import Reranker
```

### 5.2 `__init__` 里初始化 Reranker

```python
def __init__(self) -> None:
    self.embedder = OllamaEmbedder()
    self.store = QdrantStore()
    self.reranker = Reranker()          # ← 新增这行
    self.llm = LLMClient()
```

### 5.3 `ainvoke` 方法里插入 Reranker 步骤

改动前（旧）：

```python
# ── 2. Qdrant 检索 ──
hits = self.store.search(query_vec, top_k=top_k)    # 搜 5 条
context = self.store.search_with_context(...)

# ── 3. Prompt 拼装 ──
prompt = RAG_PROMPT.format(context=context, question=query)
```

改动后（新）：

```python
# ── 2. Qdrant 检索 (多取几条给 Reranker 挑) ──
retrieval_k = max(settings.RETRIEVAL_K, rerank_top_k)
hits = self.store.search(query_vec, top_k=retrieval_k)   # 搜 10 条

# ── 3. Reranker 重排序 ── (新增)
reranked = self.reranker.rerank(query, hits, top_k=rerank_top_k)  # 精排到 3 条
context_snippets = [h.text for h in reranked]
context = "\n\n---\n\n".join(
    f"[来源 {i}] (相关度: {getattr(h, 'score', None) or 'reranked'})\n{h.text}"
    for i, h in enumerate(reranked, 1)
)

# ── 4. Prompt 拼装 ──
prompt = RAG_PROMPT.format(context=context, question=query)
```

**对比表格：**

| 步骤 | 旧链路 | 新链路 |
|------|--------|--------|
| 检索数量 | 直接 5 条 | 先搜 10 条（`RETRIEVAL_K`） |
| 排序方式 | 向量余弦相似度 | 余弦相似度 + Cross-Encoder 精排 |
| 送 LLM | 5 条全部 | 精选 3 条（`RERANK_TOP_K`） |

---

## 6. 首次加载模型

BGE-Reranker-v2-m3 约 2.3GB，第一次从 HuggingFace 下载。国内用镜像：

```bash
cd D:\Projects\personal_rag_platform

# 预下载模型（只跑一次，以后直接用缓存）
HF_ENDPOINT=https://hf-mirror.com \
  .venv/Scripts/python.exe -c "from src.reranker import Reranker; Reranker()"
```

---

## 7. 验证

```bash
# 重启 FastAPI 服务
cd D:\Projects\personal_rag_platform

# 先把旧的关了
taskkill /F /IM python.exe /FI "WINDOWTITLE eq uvicorn*" 2>nul

# 启动新服务
PYTHONPATH=. .venv/Scripts/python.exe -m uvicorn src.main:app \
  --host 127.0.0.1 --port 8099
```

然后测试：

```bash
# 注意：请求体字段是 "query" 不是 "question"
curl -X POST http://127.0.0.1:8099/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{"query":"什么是存算分离架构？","top_k":3}'
```

> ⚠️ Windows Git Bash 对 curl 的 `-d` JSON 引号处理有问题，建议用 Python 发请求或写到临时文件再 `-d @file`。

看日志里有没有 `Reranker` 相关输出 — 有就说明 Reranker 在干活了。首次加载模型需要额外 30 秒左右（CPU 加载 2.3GB 权重）。

---

## 8. 如何回滚

如果 Reranker 太慢或效果不好：

```bash
# 最简单：把 RETRIEVAL_K 下调到 3 就退回了原来的行为
# 在 .env 里加
RETRIEVAL_K=3
RERANK_TOP_K=3
```

这样检索 3 条 → Reranker 3 条里选 3 条 → 等于没变。

---

## 10. 踩坑记录

> 这些坑都是在 2026-07-23 实战验证时踩的，一个个踩过来花了两小时。记录下来省得下次再绕路。

### 坑 1：Qdrant 数据卷丢失

**现象：** 重启 Docker 容器后 Qdrant 集合全部为空（`/collections` 返回 `[]`）。

**原因：** 旧容器被 `docker rm -f` 删除后，新容器运行时创建了一个新的匿名卷覆盖了命名卷 `qdrant_storage` 的挂载点。

**解决：**

```bash
# 1. 确保容器挂载正确的命名卷
docker rm -f qdrant
docker run -d --name qdrant \
  -p 6333:6333 -p 6334:6334 \
  -v qdrant_storage:/qdrant/storage \
  qdrant/qdrant

# 2. 如果数据确实丢了，重新灌数据
cd D:\Projects\personal_rag_platform
PYTHONPATH=. .venv/Scripts/python.exe -m src.ingest docs/
```

**教训：** 生产环境数据卷一定要**命名卷 + 定期备份**，不要依赖容器存活。

---

### 坑 2：HF 模型离线加载

**现象：** Reranker 首次加载时报 `WinError 10060` 连接超时——它尝试访问 `huggingface.co`，但墙内不通。即使设置了 `HF_ENDPOINT=https://hf-mirror.com`，`sentence-transformers` 也不认这个环境变量。

**解决：** 在 `Reranker.__init__` 里先尝试 `local_files_only=True`（模型已预下载到缓存），失败再 fallback 到镜像：

```python
try:
    self.model = CrossEncoder(model_name, local_files_only=True, ...)
except Exception:
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    self.model = CrossEncoder(model_name, ...)
```

**验证模型缓存路径：**
```bash
du -sh ~/.cache/huggingface/hub/models--BAAI--bge-reranker-v2-m3/
# 完整缓存约 2.2GB
```

---

### 坑 3：uvicorn 不在项目 .venv

**现象：** `python -m uvicorn` 报 `No module named uvicorn`。

**原因：** `uv pip install` 时默认可用了 Hermes 的全局 site-packages，`uvicorn` 被装到了 Hermes 的 venv 而不是项目 `.venv`。

**解决：**

```bash
# 确认 uvicorn 实际位置
.venv/Scripts/python.exe -c "import uvicorn; print(uvicorn.__file__)"

# 如果路径不对，直接 pip install 到项目 venv
.venv/Scripts/python.exe -m pip install "uvicorn[standard]" -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

### 坑 4：main.py 未传 top_k 到链路

**现象：** 请求填了 `top_k: 3`，但链路里始终用默认值。

**原因：** FastAPI `query` 函数只传了 `{"input": body.query}`，没传 `top_k`。

**修复：** 改为 `{"input": body.query, "top_k": body.top_k}`。

---

### 坑 5：Windows Git Bash curl JSON 引号

**现象：** curl 发 POST JSON 返回 400 Bad Request。

**原因：** Git Bash 对 `-d '{"key":"value"}'` 的单引号和花括号解析有问题。

**回避方案：** 用 Python `urllib` 代替 curl，或把 JSON 写到临时文件 `-d @/tmp/test.json`（注意 Windows 下用 `$TEMP` 或绝对路径）。

---

### 坑 6：Docker 端口占用 (winnat)

**现象：** `docker run -p 6333:6333` 报 `bind: An attempt was made to access a socket in a way forbidden by its access permissions`。

**原因：** Windows NAT 驱动偶尔会占住端口不释放。

**解决：**
```bash
net stop winnat && net start winnat
```

---

## 11. 下一步优化方向（记给自己）

- [ ] `rag_chain.py` 改成 `async def generate()` — LLM 推理时 FastAPI 不阻塞
- [ ] 加 `pytest` 测试用例：验证 Reranker 输出的前 3 条比后 7 条更相关
- [ ] 对比实验：同样 30 个问题，有 Reranker 和无 Reranker 的回答质量差异
- [ ] Docker Compose 化：`docker-compose up` 一键启动 Qdrant + FastAPI

---

> 写于 2026-07-21 · 2026-07-23 实战验证踩坑 6 处，已全部记录
