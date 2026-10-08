# 智能体记忆框架工程调研（02）——主流开源项目的工程实现与 API 设计

> 调研日期：2026-10-08。方法：逐个抓取 GitHub README（api.github.com raw）、必要时查源码（graphiti/mem0/langmem 源文件）与官方 docs。每个项目按固定七要素记录：定位 / 数据模型 / API 形状 / 存储后端 / 检索管线 / 巩固与更新 / 许可证。
> 本篇姊妹篇：01-algorithms.md（算法层）。

---

## 1. mem0ai/mem0

**定位一句话**：通用"记忆层"中间件——把对话消息变成事实条目并按 user/agent/run 三级命名空间存取，pip 即用、托管云可选。

**数据模型**：记忆条目（事实化短句）+ 元数据。核心字段：`id`、`memory`（事实文本）、`hash`、`metadata`（自由 KV）、`score`、`user_id / agent_id / run_id`（多级命名空间）、`created_at / updated_at`、`event`（本次 ADD/UPDATE/DELETE 事件名，用于回放）。另有一张 `history` 表记录每条记忆的变更轨迹（`memory.history(memory_id)` 可查）。

**API 形状**（Python 库风格，同步/异步双份）：
```python
from mem0 import Memory
memory = Memory()
memory.add(messages, user_id=..., agent_id=..., metadata={...})   # 消息 -> 事实抽取 -> 入库
memory.search(query, filters={"user_id": uid}, top_k=3)           # 返回 {"results": [{"memory": ...}]}
memory.get_all(...); memory.update(memory_id, data); memory.delete(memory_id); memory.history(memory_id)
```
同构 CLI（`mem0 add "..." --user-id alice`）与 npm SDK。典型用法是"回复前 search，回复后 add"。

**存储后端**：向量库可插拔达 20+（qdrant、pgvector、chroma、faiss、milvus、redis、pinecone、supabase、turbopuffer…），另有 graph 记忆版（Neo4j）。自托管 server 走 docker compose（FastAPI + 前端 dashboard）。

**检索管线**（2026-04 新算法，README 明示）：单趟检索（无 agentic loop）：
- 语义向量 + **BM25 关键词** + **实体匹配**三路并行、分数融合（`pip install mem0ai[nlp]` 装 spacy 做实体抽取）；
- **实体链接**：实体被抽出、嵌入、跨记忆链接，作为检索加权信号；
- **Temporal Reasoning**：对"现在状态/过去事件/未来计划"类查询按日期实例重排序；
- top_200 检索预算。基准：LoCoMo 92.5 / LongMemEval 94.4（注意：托管平台数字，OSS 非同等）。

**巩固与更新**：旧版是 LLM 三决策 ADD/UPDATE/DELETE/NOOP（冲突时改写旧条目）；**新版 v3 反转为 ADD-only**——单趟抽取、只增不改（"Memories accumulate; nothing is overwritten"），冲突交给检索端的时序排序兜底。这是行业风向转变的重要信号：改写成本高、易错，堆量+检索时择新更稳。

**多模态**：无原生图像记忆条目（托管平台有，OSS 侧主要文本事实）。
**时序字段**：条目有 created_at/updated_at，无显式 valid_at/invalid_at。
**许可证**：Apache 2.0。

---

## 2. letta-ai/letta（f.k.a. MemGPT）

**定位一句话**：OS 式状态化 agent 服务器——把记忆分成"上下文内可自编辑的块 + 上下文外可检索的库"，agent 自己通过工具维护自己的记忆。注意：仓库主体已迁往 `letta-ai/letta-code`（npm `@letta-ai/letta-code`），旧 V1 API server 在 `archive` 分支。

**数据模型**——这是它最有特色的部分，字段精确（docs 验证）：
- **Memory Block（core memory）**：`{id, label, description, value, limit, read_only, metadata}`。`value` 是纯字符串，`limit` 是**字符数上限**（XML 渲染里显示 `chars_current / chars_limit`）；block 是一等公民，可跨 agent 共享（"shared blocks"）、attach/detach。典型 label：`human`、`persona`。
- **Archival memory**：外存段落（passages），API 走 `client.blocks...` 与 `passages` 子资源 `.search()`。
- **Recall memory**：完整对话历史（含被压缩踢出的），随时可查。
- 所有消息/推理/工具调用一律持久化到 DB。

**API 形状**：REST/SDK 面向"agent 与 block"而非"记忆条目"：
```python
client.blocks.create(...); client.blocks.update(block.id, ...)   # 整体替换 value，非 append
client.agents.blocks.retrieve(agent.id, block_label)
client.agents.blocks.attach / detach(...)
# passages: client.agents... /v1/agents/{id}/passages/search?text=...
```
agent 侧通过内置 memory 工具自我编辑（memory_insert / memory_replace / memory_rethink 一族），即"记忆维护是 agent 的主动行为，不是后台管线"。

**存储后端**：Postgres（pgvector 做 archival 嵌入检索）；一切状态入库。

**检索管线**：core block 直接注入 system prompt（免检索）；archival/recall 用嵌入相似度搜索（pgvector），recall 可按时间范围查消息。

**巩固与更新**：无后台巩固器。依赖 agent 在对话中自我反思并调用 memory 工具改写 block；并发写冲突语义是 **last write wins**（docs 明示，update 是整块替换不是追加——设计上简单粗暴，但语义清晰）。

**多模态**：消息层支持图文，记忆块本身是文本。
**时序字段**：无 valid_at/invalid_at；靠消息时间戳。
**许可证**：Apache 2.0。**MIRIX 声明其记忆系统即以 Letta 为底座扩展**。

---

## 3. getzep/graphiti（Zep 时序知识图谱）

**定位一句话**：把演化中的实体/关系/事实构建成**双时序知识图谱**（context graph），事实带有效期，能回答"现在为真 vs 何时为曾经为真"。Zep 商业平台的 OSS 核。

**数据模型**（源码 `graphiti_core/nodes.py`、`edges.py` 验证，Pydantic）：
- **Entity 节点**：`uuid, name, group_id, labels, created_at, summary（随时间演化的邻域摘要）, name_embedding, attributes: dict`（可由开发者 Pydantic 自定义 schema）。
- **EntityEdge（事实边）**：`uuid, group_id, source_node_uuid, target_node_uuid, name(关系名), fact(自然语言事实), fact_embedding, episodes[](溯源), expired_at, valid_at, invalid_at, reference_time, attributes`。
- **Episode（溯源节点）**：`source(类型: text/message/json…), source_description, content(原始数据), valid_at, entity_edges, episode_metadata`。
- **Community 节点**：`summary, name_embedding`（图聚类摘要）。

时序语义是**双时序**：`valid_at` = 事实在现实世界开始为真的时刻（从 episode 内容抽取）；`invalid_at` = 被新事实否定的时间；`expired_at` = 该边失效时刻；`reference_time` = 事件发生时间。**旧事实不删除，只标 invalid**——完整历史保留。

**API 形状**：
```python
graphiti = Graphiti(uri, user, password, llm_client=..., embedder=..., cross_encoder=...)
await graphiti.add_episode(name=..., episode_body=..., source=EpisodeType.text, reference_time=..., group_id=...)
results = await graphiti.search(query)                 # 混合检索，默认 recipe
edges = await graphiti.search_(...)                    # 可选节点/边 recipe
```
另有 MCP server 与 FastAPI REST server（`server/`、`mcp_server/` 目录）。

**存储后端**：图库可插拔——Neo4j（推荐）、FalkorDB（含内嵌 falkordblite）、Amazon Neptune+OpenSearch；**Kuzu 已弃维**（上游停维）。driver 模式：`Graphiti(graph_driver=driver)`。

**检索管线**：混合检索 = 语义嵌入（节点名/事实 embedding）+ BM25 全文 + **图遍历**（graph distance）；**rerank 用 cross-encoder**（OpenAI/Gemini 的 logprob 布尔分类法做相关性打分）；支持预定义 search recipes（按节点/边、按时间窗过滤）。

**巩固与更新**：增量构建——新 episode 进来时 LLM 抽实体/事实 → **与已有图去重对齐**（dedupe，依赖 structured output）→ 相同事实保留旧边、冲突事实把旧边标 `invalid_at` 而非删除。社区结构定期重算。并发由 `SEMAPHORE_LIMIT`（默认 10）闸门，防 LLM 429。

**多模态**：episode 支持 json/消息类型，无图像条目。
**许可证**：Apache 2.0。

---

## 4. langchain-ai/langmem

**定位一句话**：LangChain 官方的记忆函数库——不是服务，是一组可组合的原函数（hot-path 工具 + 后台管理器），落库靠 LangGraph BaseStore。

**数据模型**：记忆 = `Memory`（id/content/timestamps…）存放在 **namespaced BaseStore** 里；namespace 是 tuple 且支持**模板变量**：`("memories", "{langgraph_user_id}")`——检索时按用户身份插值。`create_memory_manager(model, schemas=None)` 还支持**自定义 Pydantic schema** 把对话抽成结构化对象。Store 配置嵌入索引：`InMemoryStore(index={"dims": 1536, "embed": "openai:text-embedding-3-small"})`。

**API 形状**（函数式工厂，返回闭包/Tool）：
```python
create_memory_manager(model, schemas=None)      # (messages, existing=?) -> list[(op, Memory)]
create_memory_searcher(model, namespace=...)    # (query) -> list[Memory]
create_memory_store_manager(model, store=...)   # (messages) -> None  抽取+去重+入库一体
create_thread_extractor(model, schema=...)      # (messages) -> Summary
create_manage_memory_tool(namespace=("memories",))   # agent 工具：写记忆
create_search_memory_tool(namespace=("memories",))   # agent 工具：搜记忆
```
（源码 `src/langmem/knowledge/__init__.py` 验证。）

**存储后端**：LangGraph `BaseStore`——开发期 `InMemoryStore`，生产 `AsyncPostgresStore`（LangGraph Platform 内置）。记忆条目带 semantic index。

**检索管线**：向量语义检索（store 层嵌入）；`MemoryPhase` 区分 **hot path（对话中即时）与 background（ReflectionExecutor 异步抽取）** 两路，避免阻塞响应。

**巩固与更新**：`create_memory_store_manager` 抽取时带 `existing` 参数——新记忆与已有记忆比对后**合并/更新**（LLM 判 merge vs new）；prompt optimization 模块（`langmem.prompts`）还能把交互蒸馏回系统提示词（"RSI"路数）。

**多模态**：无图像条目。
**时序字段**：无显式 valid_at。
**许可证**：MIT。**工程启示：函数式原语 + namespace 模板变量 + hot/background 双路是干净的设计**，不绑存储即可复用。

---

## 5. Mirix-AI/MIRIX

**定位一句话**：六种记忆、六个专职 agent 的多智能体个人助理——从屏幕活动流+对话里持续构建记忆（UCSD/论文 arXiv 2507.07957），**以 Letta 为底座**。

**数据模型**——记忆类型分层的最完整实现（六类）：
| 组件 | 存什么 | 管理者 |
|---|---|---|
| Core | 常驻 block（label=human/persona, value） | core_memory_agent |
| Episodic | 带时间的情景事件（含屏幕活动摘要） | episodic_memory_agent |
| Semantic | 事实/世界知识 | semantic_memory_agent |
| Procedural | 技能/流程（含从工具报错-重试-修复轨迹蒸馏的 skill） | procedural_memory_agent |
| Resource | 资料/文档引用 | resource_memory_agent |
| Knowledge Vault | 敏感知识（密码类，带访问控制） | knowledge_vault_memory_agent |

消息体是多模态 OpenAI 形状：`content: [{"type": "text", "text": ...}]`，assistant 消息可带 `tool_calls`，`role="tool"` 轨迹**原样入会话库**（作为程序性记忆蒸馏的原料——这是很聪明的设计：工具失败→修复全过程是最强技能信号）。

**API 形状**（client-server，REST :8531 + Python 客户端 `mirix-client`）：
```python
client = MirixClient(api_key=..., base_url=...)
client.initialize_meta_agent(config={llm_config, embedding_config, meta_agent_config:{agents:[...]}})
client.add(user_id=..., messages=[...], session_id=...)
client.retrieve_with_conversation(user_id=..., messages=[...], limit=5)
client.auto_dream(user_id=..., mode="procedural", meta_agent_id=..., last_n_sessions=3, dry_run=False)
```

**存储后端**：PostgreSQL（本地、隐私优先）；BM25 全文（PG native）+ 向量相似度（pgvector 类）；dashboard :5173。

**检索管线**：各记忆 agent 按类目路由检索；`retrieve_with_conversation` 用"当前对话"做查询语境（不只是裸 query），FTS+向量混合。

**巩固与更新**：**Auto-Dream**——显式 REST 端点 `POST /memory/auto_dream?user_id=`，模式 core/episodic/semantic/resource/procedural/knowledge/experience（experience=三库联审）；合并重复、消解 stale/冲突，支持 `dry_run` 先看会动哪些条目（**工程上极佳的安全阀**）。Procedural 模式还能按 `last_n_sessions` 蒸馏近期会话成技能。

**多模态**：屏幕截图持续摄入（视觉数据→结构化记忆），文本/图像/语音。
**时序字段**：episodic 带发生时间；无 graphiti 式双时序。
**许可证**：Apache 2.0。

---

## 6. topoteretes/cognee

**定位一句话**：自托管"AI 记忆平台"——文档/代码/对话变成知识图谱+向量混合存储，**可全程无 LLM key 本地跑**（GLiNER 抽取 + 本地嵌入）。

**数据模型**：图谱化——文本变成实体/关系/可检索 chunk；代码变成符号依赖图；文档带 `external_metadata`（v1.6.1 起每个 chunk 戳来源元数据并在混合检索时透出）。会话经蒸馏（session distillation）把"被采纳的经验"固化进永久记忆。支持自定义 ontology/数据模型。

**API 形状**（**动词极简，四个**）：
```python
import cognee
await cognee.remember(text, dataset_name=...)     # 存
await cognee.recall(query, datasets=[...])        # 查（自动路由检索策略）
await cognee.improve(...)                         # 丰富/反馈/会话->图
await cognee.forget(...)                          # 删条目或整个 dataset
```
CLI 同构（`cognee-cli remember/recall/demo`）。另有 REST API、MCP server、Claude Code/Codex 插件、TS/Rust SDK。

**存储后端**：默认本地图+向量（LanceDB 类）；**1.0 起可单 Postgres 跑整个记忆层**（关系元数据+PGVector+图状态三合一，但 graph-on-postgres 生产版收费，OSS demo）。也支持 Neo4j 等。Docker 镜像内置 GLiNER。

**检索管线**：自动路由选择 graph/vector/代码检索策略；混合检索时透出 external_metadata；可选 LLM 生成答案或只返回证据片段（无 key 时）。

**巩固与更新**：`improve` 承担增量丰富与反馈；pipeline 崩溃恢复保留已完成文档。显式 `forget` 是少数把"遗忘"做成一等 API 的项目。

**多模态**：媒体处理需额外视觉/转写模型配置（有路径，非核心）。
**许可证**：Apache 2.0（badge 链接 LICENSE）。
**独特工程点**：**COGX 交换格式**——可以从 Mem0/Letta/Zep/Graphiti 导入记忆。说明"记忆可迁移"已被当卖点，值得跟进其格式。

---

## 7. agiresearch/A-mem（A-MEM: Agentic Memory）

**定位一句话**：Zettelkasten 卡片盒式 agentic 记忆——每条记忆是带结构化属性的"笔记"，新记忆自动找邻居、互相链接、**记忆演化**（论文 arXiv 2502.12110， Rutgers）。

**数据模型**（Note）：`content, context（LLM 生成的上下文描述）, tags（LLM 生成）, keywords（自动抽取）, category, timestamp（YYYYMMDDHHmm）`。无图节点/边，靠 **ChromaDB 相似度** + 生成的链接构成网络。

**API 形状**：
```python
ms = AgenticMemorySystem(model_name='all-MiniLM-L6-v2', llm_backend="openai", llm_model="gpt-4o-mini")
mid = ms.add_note(content, tags=[...], category=..., timestamp=...)   # 自动：生成context/tags/keywords->找相似->建链->演化
m = ms.read(mid); ms.search_agentic(query, k=5)
ms.update(mid, content=...); ms.delete(mid)
```

**存储后端**：ChromaDB（向量）+ 内存字典（元数据）；LLM 可 OpenAI/Ollama。

**检索管线**：ChromaDB 嵌入相似度（默认 all-MiniLM-L6-v2），返回带 tags/context 的笔记。

**巩固与更新**：**memory evolution** 是核心创新——add/update 时自动：① 生成新笔记的结构化属性；② 检索历史记忆找语义相关者；③ 更新相关笔记的 context/tags 并建立**双向链接**（类似论文引用网）。无冲突消解、无删除式遗忘。

**多模态/时序**：无。**许可证**：MIT。**工程启示：轻量、可嵌入；"记忆间建立链接并回写邻居元数据"是很便宜的增益机制，值得抄。**

---

## 8. MemTensor/MemOS

**定位一句话**："记忆操作系统"——统一 store/retrieve/manage，以**记忆立方（memory cube）**做多知识库隔离/共享，图结构存储且**可检视可编辑**（反对黑盒 embedding store），多模态与工具记忆原生。

**数据模型**：记忆条目组织为**图**（非纯向量堆），挂载在 cube 上；cube 归属 `owner_id`、读写用 `writable_cube_ids / readable_cube_ids` 分离（**读写权限建模是独门设计**）。支持 text/图像/tool trace/persona 类型。本地插件版分层：L1 traces / L2 policies / L3 world models / crystallized Skills（对应程序性记忆的分层固化）。

**API 形状**（REST，云与自托管同构）：
```python
# 云端
POST {base}/add/message   {"user_id","conversation_id","messages":[...]}
POST {base}/search/memory {"query","user_id"}
# 自托管 /product
POST /create_cube {"cube_name","owner_id","cube_id"}
POST /add         {"user_id","writable_cube_ids":[...],"messages":[...],"async_mode":"sync"}
POST /search      {"query","user_id","readable_cube_ids":[...]}
```
另有自然语言**反馈修正** API（correcting/supplementing/replacing）。

**存储后端**：自托管 = **Neo4j（图）+ Qdrant（向量）** docker compose 一键；本地插件 = SQLite + FTS5 全文 + 向量，零云依赖；云 = 托管。

**检索管线**：混合检索（FTS5/BM25 + 向量）；`MemScheduler` **异步摄取**（毫秒级延迟承诺，写入不阻塞对话主链路——生产级设计）；MemorialDashboard/Memory Viewer 可视化检视图。

**巩固与更新**：smart dedup（本地插件）；自然语言反馈驱动的记忆修正；技能分层演化（traces→policies→world model→skill）。

**多模态**：图像、工具轨迹、persona 一等公民，同一系统混合检索。
**时序字段**：图结构带时间，但未见 graphiti 式双时序对。
**许可证**：Apache 2.0。

---

## 9. ByteDance-Seed/m3-agent（M3-Agent）

**定位一句话**：字节的多模态长期记忆 agent（ICLR 2026）——**实时看视频+听音频**流式构建"实体中心的多模态记忆图谱"，推理时 think-检索-再 think 循环。附 M3-Bench 基准（100 robot 视角视频+920 web 视频）。

**数据模型**：**实体中心多模态记忆图**（memory graph，pkl 序列化）。节点=实体（人/物/场景），边=关系；情景记忆来自视频片段：30 秒切段→中间产物（**人脸检测+说话人分离**，eres2netv2 说话人声纹嵌入）→Memorization 模型生成 caption 与图。语义记忆随时间积累为世界知识。**图像不做"嵌入引用"而是转成 caption/结构进图**——多模态条目存的是感知结果（文本描述+声纹），不是原始像素嵌入。

**API 形状**：**研究代码，无通用 SDK**。管线脚本式：
```bash
python m3_agent/memorization_intermediate_outputs.py --data_file data/data.jsonl
python m3_agent/memorization_memory_graphs.py      --data_file data/data.jsonl
python m3_agent/control.py --data_file data/annotations/robot.json
```
记忆生成与控制由两个自训模型承担（`M3-Agent-Memorization` / `M3-Agent-Control`，RL 训练，Qwen2.5-Omni 底座）；也可换 prompt 调 Gemini/GPT-4o（prompts 在 `mmagent.prompts`）。记忆图谱可可视化（`visualization.py`）。

**存储后端**：无数据库——文件系统（jsonl/pkl）。检索靠 Control 模型在推理循环内主动调用检索动作（agentic retrieval，非固定管线）。

**巩固与更新**：流式在线更新——每 30s 片段增量 merge 进实体图（新观察挂到已存在实体节点）。

**借鉴点**：① 感知先于存储（先转结构化描述再入库）；② 实体中心组织使跨模态证据（"这个人"+"那个声音"）归并到同一节点；③ 检索本身是 agent 的推理步骤而非固定 top-k。

**许可证**：Apache 2.0。

---

## 横向观察：记忆类型分层怎么落地的

| 框架 | 分层方式 |
|---|---|
| Letta | core（块,常驻上下文）/ archival（外存）/ recall（对话史）——按"是否在上下文内"分 |
| MIRIX | 六类（Core/Episodic/Semantic/Procedural/Resource/Knowledge Vault）+ **每类一个专职 agent** 负责该类的写与查 |
| mem0 | 扁平事实池 + user/agent/run 命名空间（隐式分层，靠 metadata） |
| graphiti | 按数据角色分：episode（源）→ entity/edge（结构）→ community（摘要） |
| MemOS | cube（隔离/共享单元）+ L1/L2/L3 分层固化（traces→policies→world model） |
| m3-agent | episodic 图 + semantic 知识，双进程（记忆化/控制）并行 |
| langmem | 不分层，靠 namespace 模板变量交给应用 |

**结论**：分层本身不重要，重要的是**每层有明确的生命周期与检索路径**。MIRIX 的"一类型一 agent"和 Letta 的"块常驻+外存检索"是两个可操作极简版。

---

## API 设计对比表

| 项目 | 写入 API | 检索 API | 粒度/形态 | 多级隔离 | 时序 | 巩固机制 | 后端 | 许可 |
|---|---|---|---|---|---|---|---|---|
| mem0 | `memory.add(messages, user_id=)` | `memory.search(query, filters=, top_k=)` | 事实短句+metadata | user/agent/run_id | created_at（弱） | v3: ADD-only；旧版 ADD/UPDATE/DELETE | 20+ 向量库/Neo4j | Apache-2.0 |
| letta | agent 用 memory 工具自写 block；`blocks.create/update` | `passages.search()`、core 直接注入 prompt | block(label/value/limit) | 多 agent 共享 block | 无显式 | agent 自反思（last-write-wins） | Postgres+pgvector | Apache-2.0 |
| graphiti | `graphiti.add_episode(...)` | `graphiti.search()` + recipes + 自定义 | 三元组事实边（图） | group_id | **valid_at/invalid_at/expired_at（双时序）** | 冲突→旧事实标 invalid，不删 | Neo4j/FalkorDB/Neptune | Apache-2.0 |
| langmem | `create_manage_memory_tool()` | `create_search_memory_tool()` | 结构化 Memory + namespace 模板 | namespace 模板变量 | 无 | 后台 ReflectionExecutor 合并 | LangGraph BaseStore/PG | MIT |
| MIRIX | `client.add(user_id, messages, session_id)` | `retrieve_with_conversation(...)` | 六类记忆条目 | user_id + 类型 | episodic 时间戳 | **auto_dream(mode, dry_run)** | PostgreSQL（BM25+向量） | Apache-2.0 |
| cognee | `await cognee.remember(x, dataset_name=)` | `await cognee.recall(q)` | 图谱+chunk | dataset_name | external_metadata | `improve()`；**显式 forget()** | 单 Postgres/图库 | Apache-2.0 |
| A-mem | `ms.add_note(content, tags=...)` | `ms.search_agentic(q, k)` | Note(content/context/tags/keywords) | 无 | timestamp | 演化：邻居互链+回写 | ChromaDB | MIT |
| MemOS | `POST /add`（cube 权限分离） | `POST /search`（cube 权限分离） | 图结构条目+多模态 | **cube（owner/读写分离）** | 图内时间 | 反馈修正+dedup+技能分层 | Neo4j+Qdrant / SQLite | Apache-2.0 |
| m3-agent | 管线脚本产 memory graph | Control 模型 agentic 检索 | 实体中心多模态图 | 无 | 片段时间轴 | 在线 merge 到实体 | 文件系统(pkl) | Apache-2.0 |

## 我们要抄什么 / 避什么

**要抄（按优先级）**：
1. **graphiti 的双时序字段**：`valid_at / invalid_at / expired_at / reference_time` + "冲突事实只失效不删除"。这是回答"现在为真吗"的唯一干净方案；多模态记忆里的"位置/状态"类条目特别需要。
2. **mem0 v3 的 ADD-only + 多信号检索**：写入端单趟抽取只增不改（便宜、可回放），冲突压力转移到检索端（时序排序+BM25+实体匹配并行融合）。工程上比 LLM 三决策稳一个量级。
3. **MIRIX 的 auto_dream(dry_run=true)**：显式、可选、可预演的巩固端点。批量清理/合并做成独立 API 而非后台黑魔法，可测可审计。
4. **langmem 的 namespace 模板变量 + 函数式工厂**：`("memories", "{user_id}")` 让隔离策略与应用解耦；`create_xxx_tool()` 返回闭包/Tool 的工厂风格极易组合进任何 agent。
5. **Letta 的 memory block（label/value/limit 字符数 + read_only + 跨 agent 共享）**：把"记忆"建模成有配额的一等资源，配额逼出摘要能力；`chars_current/chars_limit` 渲染进提示词让 agent 自知空间。
6. **MemOS 的 cube 读写权限分离**（`writable_cube_ids`/`readable_cube_ids`）与 MemScheduler 异步摄取——多智能体/多项目场景必需。
7. **m3-agent 的"感知转结构再入库"**：多模态条目不存原始像素，存 caption+声纹+实体归并，检索全走文本结构。
8. **A-mem 的邻居回写**：新记忆入库时给语义邻居的 context/tags 打补丁并建链，成本一次 LLM 调用，收益是图状关联。
9. **cognee 的显式 forget() 与 COGX 交换格式**：遗忘是 API 不是副作用；记忆可导入导出是长期卖点。

**要避**：
1. **不要默认让 LLM 在写入时决定 UPDATE/DELETE 旧记忆**（mem0 旧路数）：一次抽取错误=永久信息丢失，且难回放；要么 ADD-only，要么 dry_run 闸门。
2. **不要把巩固做成不可关闭的后台循环**（早期 GraphRAG 式整图重算）：graphiti 的增量+低并发闸门（SEMAPHORE_LIMIT）证明"增量、可控并发、可跳过"才可持续。
3. **不要让 update 是 last-write-wins 的整块替换**（Letta blocks.update 已明示此坑）：并发写共享记忆必须带版本号或乐观锁。
4. **不要无 namespace 的扁平记忆池**：至少 user/session 二级，否则检索噪声和多租户都无解（mem0 老版本教训）。
5. **不要把多模态做成"图片嵌入+余弦"**：m3-agent 证明感知结构化（caption/diarization/实体归并）后检索质量远高于裸像素嵌入；图像本体只存引用 URI。
6. **不要忽视基准数据的适用范围**：各家 LoCoMo/LongMemEval 分数全是"自家配置自家评"（mem0 README 自己都注明 OSS 数字不可比），选型必须自测。

---

### 调研覆盖说明
- 9/9 项目均已核实（README 逐个通读；graphiti 时序字段读自源码 `graphiti_core/edges.py`；langmem 函数面读自 `src/langmem/knowledge/__init__.py`；letta block 字段读自 docs.letta.com）。
- M3-Agent 官方仓定位为 `ByteDance-Seed/m3-agent`（原任务写 M3-Agent，已核实，Apache-2.0，ICLR 2026）。
- 唯一未能从一手来源核实的细节：MIRIX 数据库表级 schema（README 未列，需读源码 `letta` 底座部分，未展开）。
