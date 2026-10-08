# 多模态智能体记忆系统调研纪要(2024–2026)

> 面向 multimem v0.2 升级。调研日期 2026-10-08。所有结论附出处;未能核实处明确标注。

## 1. 项目逐个调研

### 1.1 mem0 (mem0ai/mem0)
- **决策机制**:ADD 流程 = LLM 从对话抽取候选事实 → 每条候选与库中语义相似记忆比对 → LLM 四选一:ADD(无相似)/UPDATE(有相似但过时)/DELETE(矛盾)/NOOP(冗余)。另有 history store 记录变更轨迹,写入 vector store + entity store 双存储。
  - 论文: https://arxiv.org/abs/2504.19413 ;文档: https://docs.mem0.ai/overview
- **多模态**:自托管版图片经 `enable_vision: True` 配置的 vision LLM 提取文字/关键信息后**存为普通文本记忆条目**(非存图),检索与文本记忆同通道;支持 URL 或 base64 输入,`vision_details` 控制细节档位;未配置 vision 时图像轮次被**静默丢弃**(对 multimem 是反例教训)。
  - https://docs.mem0.ai/open-source/features/multimodal-support
- **LOCOMO**:相对 OpenAI 基线 +26%,p95 延迟 −91%(论文摘要)。

### 1.2 getzep/graphiti
- **Bi-temporal**:每条边带 `valid_at`(事实在现实世界成立时间)与 `invalid_at`(失效时间),另有系统摄取时间;支持点时查询(过去/现在两种视图)。
- **Edge invalidation**:新事实与旧边矛盾时**不删旧边**,写入 invalid_at 标记失效并插入新边,两条边都可按时间检索。
- **检索**:混合检索(向量 + BM25 + 图遍历),增量式情节摄取。
  - 论文(Zep): https://arxiv.org/abs/2501.13956 ;文档: https://help.getzep.com/graphiti/getting-started/overview.md
- 注:摘要页未给出 edge invalidation 的实现级细节(需读全文/源码)。

### 1.3 Letta / MemGPT
- **论文架构**(arXiv:2310.08560):OS 式虚拟上下文管理。main context(窗口内)= core memory(常驻关键信息,如用户画像)+ in-context instructions;external context = recall memory(对话历史)+ archival memory(长期归档);LLM 通过函数调用在层间搬数据(self-editing memory)。
- **2025-26 现状**:核心已转向 **MemFS**——git 版本化的记忆文件系统,agent 可自检/自编辑/`/remember` 教学;**dreaming / sleep-time compute** = 后台 subagent 定期回顾对话、提炼经验并更新记忆,可选"第二遍审核"再落盘;记忆重组(拆分/合并/重构)先备份再执行。
  - 论文: https://arxiv.org/abs/2310.08560 ;文档: https://docs.letta.com/configuration/memory/index.md
- 对 v0.1 的 archival 概念,Letta 证明了"文件化 + 版本化"是可行的升级方向。

### 1.4 Mirix (Mirix-AI/MIRIX)
- **六记忆类型**:Core / Episodic / Semantic / Procedural / **Resource**(外部资源引用)/ **Knowledge Vault**(凭证等敏感信息的加密保险库);由多 agent 框架动态协调写入与检索(active retrieval 由专门 agent 路由)。
- **多模态**:超文本+视觉体验,支持截图流;**ScreenshotVQA** 基准每序列约 2 万张高分辨率截图,MIRIX 比 RAG 基线准确率高 35%、存储需求低 99.9%(压缩为文本记忆);LOCOMO 85.4% SOTA。
  - 论文: https://arxiv.org/abs/2507.07957 ;官网: https://mirix.io/
- 对 multimem 的启示:Resource/Vault 可作为 v0.2+ 的记忆类型扩展方向;"截图→文本摘要"与 mem0 同路线。

### 1.5 LangMem (langchain-ai)
- **巩固双通道**:hot path(对话内即时读写,agent 自主决定)+ **background memory manager**(异步提取、整合、更新,不阻塞对话)。
- 三类记忆与 multimem 对齐:semantic(事实集)、episodic(交互经历)、procedural(提示词优化即程序性记忆)。
- 存储:embedding(如 text-embedding-3-small, 1536 维)+ Postgres 生产后端;开发期 InMemoryStore。
  - https://langchain-ai.github.io/langmem/

### 1.6 memoripy(小型库打分模式,已核验源码 0.1.2)
- 条目字段:`id/prompt/output/embedding/timestamp/access_count/concepts/decay_factor`。
- 打分:`adjusted_similarity = cosine_sim × decay_factor × reinforcement`,其中衰减 `decay = exp(-0.0001 × Δt)`(指数);被检索命中的记忆 decay_factor ×1.1 强化,未被命中的 ×0.9 弱化(使用计数驱动)。
- 附加:概念图 spreading activation(激活分数按 0.5 逐步衰减)叠加进总分;层次聚类 centroid 粗筛。
  - PyPI 源码包 https://pypi.org/project/memoripy/(memory_store.py)
- **注意**:memoripy 依赖 langchain-core + openai,不算零依赖;但其打分模式可借鉴。

### 1.7 2025–2026 新论文(视频/空间记忆)
- **M3-Agent**(bytedance, arXiv:2508.09736, 2025-08):实时音视频流 → entity-centric 多模态记忆(episodic+semantic),检索式记忆+多轮推理;M3-Bench(机器人第一视角+网络长视频),RL 训练后超 Gemini-1.5-pro/GPT-4o 基线 5–8%。抽帧细节在正文,摘要未给。
- **SVMemAgent**(arXiv:2609.18540, 2026-09):流式视频记忆,每时间步做 **replace-or-discard** 二选一的 query-agnostic 帧选择,GRPO 训练;无需预知视频长度。
- **MemLife**(arXiv:2609.40195, 2026-09):数小时级第一视角视频压缩为 **entity-grounded 第一人称文本片段**,time-indexed agentic reader 检索;MemOpt 用 RL 优化记忆写入器(写什么决定可检索性),+4.6–12%。
- **VideoTapestry**(arXiv:2610.06672, 2026-10):多 agent 长视频理解的 query-adaptive 记忆精炼。
- 检索词 "video memory agent" 下 2026 年已出现整批后续工作(arXiv API 检索记录,2026-10-08)。
- 共同趋势:**视频记忆的主流做法是"帧→事件级文本化+时间戳索引"**,而非存原始帧向量;VidMem(几何+语义记忆, arXiv:2506.04733)经查该 ID 实为核物理论文,**此条引用作废**(检索结果误导,已核实 arXiv 摘要页)。

## 2. 五个问题的答案

### A. 跨模态检索怎么做
- **主流共识 = "统一到文本"**:mem0 与 MIRIX 都把图片经 vision LLM 转成文本事实后入同一条检索通道——文本查图 = 直接 embedding 匹配 caption;图查文本 = 先 caption 再匹配。无需 CLIP 类跨模态向量。
- 帧级 vs 事件级:M3-Agent/SVMem/MemLife 都收敛到**事件级(episode)+ 时间戳**,帧只作为中间产物;multimem v0.1 的 caption+hash 方向正确,可补"同帧多 caption(不同粒度)"。
- 视觉向量(如 CLIP/SigLIP)路线在 agent 记忆系统里反而不常见(截至调研所见,标注:可能存在但未检索到代表性实践)。

### B. 视频记忆
- **抽帧策略**:流式 replace-or-discard(SVMemAgent);离线场景 query-agnostic 保留"信息量大帧"(实验显示模型偏好含文字的帧)。
- **关键帧选择**:任务奖励驱动(GRPO)而非启发式;轻量替代 = 场景切换检测 + 文字帧优先。
- **时间戳索引**:time-indexed episodes(MemLife);MIRIX 按时间顺序组织截图记忆。可落地为:每条目存 `t_start/t_end`,检索时可按时间窗过滤。

### C. 巩固最佳实践
- **触发时机**:① hot path 即时(mem0/LangMem agent 自主);② background 异步批量(LangMem background manager, Letta sleep-time/dreaming,触发条件 = 每 N 步或上下文压缩时);③ 写入前"第二遍审核"(Letta 的 review-before-apply)。
- **合并/摘要**:A-MEM 的 **memory evolution**(新记忆入库时触发旧记忆属性更新 + Zettelkasten 式链接生成, NeurIPS 2025, https://arxiv.org/abs/2502.12110)是 2025 年代表;mem0 用 ADD/UPDATE/DELETE/NOOP 防堆积。
- 对 multimem 可 dry-run 巩固端点的建议:对齐 Letta 的"提案→审核→落盘"两段式,dry-run 输出即提案单。

### D. 检索打分变体
- 经典三因子(Generative Agents, arXiv:2304.03442):`score = w_r·relevance + w_t·recency + w_i·importance`,recency = `0.995^(hours_elapsed)`(指数,底数<1),importance 由 LLM 1–10 打分。
- memoripy 变体:衰减直接乘进相似度而非加法,并用访问行为自适应调 decay_factor(命中×1.1/未中×0.9)+ spreading activation 加成。
- Graphiti:时间衰减作用于图边(失效边按 bi-temporal 过滤),不是分数项。
- 可借鉴组合:multimem 现有 relevance+recency+importance 之外,加 **access_count 强化项**(memoripy)与 **LLM importance 1–10**(Generative Agents);recency 底数做成可配参数(0.995 是默认参考)。

### E. 轻量向量检索最小方案
- **numpy 暴力**:≤1 万条记忆毫秒级,零依赖,multimem 现阶段最优;10 万条约需数百 MB 且变慢(经验值,未实测——标注不确定)。
- **sqlite-vec**:纯 C 扩展、单文件、MIT/Apache-2;vec0 虚拟表 + metadata columns + partition keys + int8/bit 量化(vec_quantize_binary);跨平台(含 WASM);**pre-v1 有 breaking changes**。适合作为 v0.2 的可选加速后端(仍只有一个可选依赖)。
  - https://alexgarcia.xyz/sqlite-vec/ ; https://alexgarcia.xyz/sqlite-vec/api-reference.html
- **hnswlib**:10 万+条时优势明显,但引入 C++ 扩展 + 索引内存开销,与"零重依赖"冲突,不建议默认。
- 建议分层:numpy 默认 → 条数 >2–5 万或需 metadata 过滤时切 sqlite-vec → 不考虑 hnswlib。

## 3. 对 multimem v0.2 的落地清单(按优先度)
1. 补上 mem0 式 ADD/UPDATE/DELETE/NOOP 决策 + history store(v0.1 已有软失效,补决策 LLM 合同即可)。
2. 图片条目强制 vision caption(fail-fast 而非静默丢弃,吸取 mem0 反例)。
3. 巩固端点对齐 LangMem/Letta:background 提案 → 审核落盘;引入 A-MEM 式"新记忆触发旧记忆更新"。
4. 打分加 access_count 强化项与 LLM importance;recency 底数可配。
5. 视频支持:事件级文本化 + t_start/t_end 索引;帧选择用"场景切换+文字帧优先"启发式起步。
6. 可选 sqlite-vec 后端(量化 + metadata 过滤),保持 numpy fallback。

## 4. 未验证/存疑项
- graphiti edge invalidation 的具体字段实现(仅到文档概述层,未读源码)。
- MIRIX active retrieval 的路由细节(摘要页未展开)。
- numpy 暴力检索的具体性能拐点(未实测,建议 v0.2 自己 benchmark)。
- VidMem 引用作废(见 1.7);Mem0 论文中 NOOP 的 prompt 细节仅由摘要页通用描述转述,未读正文。
