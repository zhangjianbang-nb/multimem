# 智能体记忆 / 多模态智能体记忆：学术研究现状调研（2023–2026）

> 调研日期：2026-10-08。所有 arXiv 链接均经 arxiv.org/abs/ 页面或 arXiv API 核实；
> 个别未能逐字核对的字段已在条目内标注「待核」。URL 无把握处标注「未能核实」。

## 0. 快速导览

- **一、记忆分类学**：working/episodic/semantic/procedural 四类记忆在 8 个代表系统中的实现对照。
- **二、核心系统**：14+ 个系统逐一给一句话机制 + 已核实 URL + 关键数据。
- **三、四阶段对比**：写入-巩固-遗忘-检索 横向对比表 + 三条横向观察。
- **四、评测基准**：LoCoMo / LongMemEval / MemBench / M3-Bench / Video-MME 等 6 项。
- **五、2025–2026 多模态记忆新工作**：GUI 截图、视频自我中心、记忆管理训练化三条线，18 篇。
- **六、对设计的启示**：5 条可直接落地的设计决策。

---

## 一、记忆分类学（Taxonomy）

学界通行的四分法来自认知心理学（working / episodic / semantic / procedural），
下表对照各系统如何实现这四类（+ 各系统自创类型）：

| 认知类型 | 含义 | MemGPT | Mem0 | MemOS | MIRIX | M3-Agent | AWM |
|---|---|---|---|---|---|---|---|
| Working（工作记忆） | 当前上下文内的即时状态 | 主上下文（in-context）+ 外部存储分页调度 | 会话内缓存（不作为长期项） | MemCube 调度中的激活记忆 MemC | 未单列（由 Working Memory 组件管理会话） | 实时流处理中的即时缓冲 | 任务内提示 |
| Episodic（情景） | 带时间戳的具体经历 | 对话历史外存（conversation queue） | 抽取-整合后的事实条目（含时间戳） | MemG（明文记忆）的一种组织方式 | Episodic Memory（事件级） | 视觉/听觉流逐帧写入的情景记忆（实体中心组织） | 交互轨迹（间接来源） |
| Semantic（语义） | 抽象事实/世界知识 | 上层递归摘要 | 整合（consolidate）后的用户事实 | MemG + 知识图谱化组织 | Semantic Memory | 由情景记忆归纳出的实体语义知识（人物关系、属性） | — |
| Procedural（程序性） | 技能/工作流/策略 | system instructions 可自我修改（有限） | — | MemP（参数记忆，写入权重） | Procedural Memory（技能/工具用法） | 由轨迹学到的操作技能（RL 训练隐含） | **核心**：从历史任务归纳可复用 workflow |
| 自创类型 | — | 虚拟上下文（核心思想） | Mem0g 图记忆（关系型） | MemS 三态统一（MemC/MemP/MemG） | Core / Resource Memory / Knowledge Vault | 多模态实体图 | — |

要点：
- ** MemOS 是唯一把「参数记忆」（写入权重）纳入统一资源调度的系统**，对应 continual learning；
- **MIRIX 是类型划分最细的系统**（6 类），且显式为多模态（视觉经验）设计；
- **M3-Agent 的语义记忆是"从多模态情景中归纳"的**——这是纯文本记忆系统普遍缺失的能力；
- AWM 证明 procedural 记忆可以显式化（workflow 即记忆），而非只存在于模型权重里。

---

## 二、核心系统逐条（一句话机制 + URL）

### 1. MemGPT / Letta（虚拟上下文管理）
- **机制**：借鉴操作系统虚拟内存的分层分页思想，LLM 自主调用函数在主上下文与外部存储之间搬运数据，从而在有限上下文窗口内维持无限期对话与文档分析。
- **URL**：https://arxiv.org/abs/2310.08560 （NeurIPS 2023；项目已演化为公司 Letta，官网 https://www.letta.com ，代码 https://github.com/letta-ai/letta ）
- 备注：开创了"记忆管理即工具调用"范式，后续几乎所有 agentic memory 系统都沿用这一接口形态。
- **分类学位置**：working = 主上下文窗口；episodic = 对话历史外存队列；semantic = 上层递归摘要（rolling summary）；procedural = 可自我编辑的 system instructions（能力有限，靠 self-editing 工具）。
- **遗产**：分页（paging）思想直接启发了 MemOS 的"记忆调度"；self-editing memory API 成为 Letta 平台的产品核心。

### 2. Mem0 / Mem0g（抽取-整合式 + 图记忆）
- **机制**：两阶段管道——先由 LLM 从对话中抽取候选记忆，再与库中已有记忆做 ADD/UPDATE/DELETE/NOOP 整合决策；变体 Mem0g 把记忆组织成图（实体节点 + 关系边），捕捉关系结构。
- **URL**：https://arxiv.org/abs/2504.19413 （官网 https://mem0.ai ，代码 https://github.com/mem0ai/mem0 ）
- 数据（论文自报）：对比全上下文，p95 延迟降 91%、token 成本降 90%+；Mem0g 比 Mem0 再高约 2%，对 OpenAI（记忆方案）judge 胜率 +26%。

### 3. A-MEM（Agentic Memory，Zettelkasten 卡片盒）
- **机制**：新信息到达时生成结构化笔记（上下文描述+关键词+标签），按 Zettelkasten 方法动态链接到相关历史记忆，并支持"记忆演化"——新记忆可触发旧记忆的属性更新，知识网络持续自组织。
- **URL**：https://arxiv.org/abs/2502.12110 （NeurIPS 2025；代码 https://github.com/agiresearch/A-mem ）
- 备注：首次把"记忆之间可以互相改写"做成一等公民，而非只增不改的 append-only 库。

### 4. MemOS（记忆操作系统）
- **机制**：把记忆当作操作系统级可管理资源（调度、迁移、融合），用 MemCube 封装内容+元数据（来源、版本），统一三种记忆形态——MemG（明文）、MemC（激活记忆，如 KV cache）、MemP（参数记忆），并支持记忆在三种形态之间转换。
- **URL**：https://arxiv.org/abs/2507.03724 （v4 更新于 2025-12；团队 MemTensor / 上海交大；官网 https://memos.openmem.net 待核，代码 https://github.com/MemTensor/MemOS ）
- 备注：野心最大的一篇——目标是"记忆即一等系统资源"，把 RAG、KV-cache 复用、参数化微调收编到同一调度框架。

### 5. HippoRAG / HippoRAG 2（海马体启发检索）
- **机制（1代）**：用 LLM 抽取 KG 三元组，检索时跑 Personalized PageRank 模拟海马体-新皮层的记忆索引理论，实现多跳联想检索（较当时 RAG +20% 多跳准确率）。https://arxiv.org/abs/2405.14831 （NeurIPS 2024）
- **机制（2代）**：在 1 代基础上加深 passage 与图的整合、更充分地在线使用 LLM，使事实型、理解型、联想型三类任务同时优于基线（联想记忆较最强 embedding 模型 +7%），定位为"非参数化持续学习"。https://arxiv.org/abs/2502.14802 （ICML 2025）
- 代码：https://github.com/OSU-NLP-Group/HippoRAG
- 备注：这是"检索结构本身记忆化"路线的代表——记忆不在存储层而在图上的一次扩散过程里。

### 6. MIRIX（多模态多类型记忆）
- **机制**：多智能体记忆框架——六类记忆（Core / Episodic / Semantic / Procedural / Resource Memory / Knowledge Vault）由多个专职 memory agent 动态管理写入与检索，支持视觉（截图/图像）经验的长期保存，面向日常助手场景。
- **URL**：https://arxiv.org/abs/2507.07957 （官网 https://mirix.io 待核，代码 https://github.com/Mirix-AI/MIRIX 待核）
- 备注：作者 Yu Wang、Xi Chen 等（arXiv 页未列机构；社区信息称 UCSD 相关，**待核**）。配套多模态评测 ScreenSpot-v2 / AndroidWorld 引入。

### 7. M3-Agent（字节 Seed，多模态长视频记忆智能体）
- **机制**：实时并行处理视觉+听觉输入流，以实体为中心构建多模态情景记忆与语义记忆（人物、关系、事件的世界知识）；推理时自主多轮检索记忆作答；端到端用强化学习训练。
- **URL**：https://arxiv.org/abs/2508.09736 （代码 https://github.com/bytedance-seed/m3-agent ）
- 配套基准：**M3-Bench**（100 段机器人视角长视频 + 920 段网络视频 QA）；论文自报超 Gemini-1.5-pro/GPT-4o 提示基线 5.3–7.7%。

### 8. 其他代表性工作

- **MemoryBank**（2023）：用艾宾浩斯遗忘曲线按时间与重要性对记忆做遗忘/强化，配套共情陪伴 bot「SiliconFriend」。https://arxiv.org/abs/2305.10250 —— **首个把"遗忘"形式化的系统**。
- **Memory3**（2024）：提出第三种记忆形态"显式记忆"（explicit memory，介于参数与上下文 KV 之间），外部知识以可检索的稀疏 KV 形式存在，2.4B 模型凭此对标更大模型与 RAG 且解码更快。https://arxiv.org/abs/2407.01178 （官网 https://mem3.huaweitech.com 待核）——与 MemOS 的 MemC 思想同源。
- **MemInsight**（2025）：让 agent 自主为记忆数据生成语义属性/标注（autonomous memory augmentation），提升检索与推荐质量（LoCoMo 检索 recall 较 RAG +34%）。https://arxiv.org/abs/2503.21760
- **AWM（Agent Workflow Memory）**（2024）：从历史任务轨迹中归纳可复用 workflow（程序性记忆），离线/在线两种模式下按需注入提示；Mind2Web 相对成功率 +24.6%、WebArena +51.1%。https://arxiv.org/abs/2409.07429
- **Zep / Graphiti**（2025）：时序知识图谱记忆引擎，边带时间区间并维护历史版本，长对话基准上超 MemGPT。https://arxiv.org/abs/2501.13956 （代码 https://github.com/getzep/graphiti ）
- **RMM（Reflective Memory Management）**（2025）：前瞻反思（多粒度记忆库）+ 回顾反思（在线 RL 根据引用证据调检索器），面向长期个性化对话。https://arxiv.org/abs/2503.08026
- **MemoRAG**（2024/2025）：双系统——轻量长程模型先建全局记忆并起草线索，再由强模型据检索内容生成答案，面向超长上下文。https://arxiv.org/abs/2409.05591 （TheWebConf 2025）
- **Memory-R1**（2025）：用 RL（PPO/GRPO）训练记忆管理器做 ADD/UPDATE/DELETE/NOOP 决策 + 答案 agent，仅 152 条 QA 即可训练，把"记忆写入策略"从提示工程变成可学习问题。https://arxiv.org/abs/2508.19828
- **LightMem**（2025，ICLR 2026）：Atkinson–Shiffrin 三阶段（感觉/短期/长期）记忆的轻量化实现，睡眠时段离线巩固，token 消耗与 API 调用大幅下降。https://arxiv.org/abs/2510.18866 （代码 https://github.com/zjunlp/LightMem ）。
  注意：2026 年已有复现研究称 naive RAG 可与之持平、LightMem 优势主要是**上下文效率**而非准确率（arXiv:2607.29104）——引用其结论时需带此限定。
- **综述**：A Survey on the Memory Mechanism of LLM-based Agents（2024，39 页，含分类法与评测综述）https://arxiv.org/abs/2404.13501 ；MemGPT 之后 2025 年还有多篇记忆综述（如 A Survey on Memory Mechanisms in the Era of LLMs，**未能核实精确 URL**）。

---

## 三、写入–巩固–遗忘–检索 四阶段对比

| 系统 | 写入（Encode） | 巩固（Consolidate） | 遗忘（Forget） | 检索（Retrieve） |
|---|---|---|---|---|
| MemGPT | 自调用函数写外部存储（append） | 上层递归摘要（对话历史→滚动摘要） | 队列挤出（隐式，无主动遗忘） | 分页调入主上下文（self-editing） |
| Mem0 | LLM 抽取候选记忆 | 与存量记忆比对做 ADD/UPDATE/DELETE/NOOP | 整合阶段的 DELETE 决策（显式但靠 LLM 判断） | 语义向量检索；Mem0g 用图遍历 |
| A-MEM | 生成结构化笔记+标签 | **记忆演化**：新笔记触发旧笔记更新/重链 | 无显式遗忘 | 标签/嵌入相似度 + 链接扩散 |
| MemOS | MemCube 封装写入 | 记忆跨形态迁移（明文→激活/参数）、版本管理 | 调度层可丢弃，未形式化遗忘曲线 | 统一调度器按需调页/激活 |
| HippoRAG | LLM 抽三元组建 KG | （2代）加深 passage-graph 整合 | 无 | Personalized PageRank 扩散 |
| MIRIX | 各专职 memory agent 分类写入 | 周期性整理（agent 自主维护） | 有（agent 可清理过期/冗余） | 类型感知的多路检索 |
| M3-Agent | 视/听流逐帧写入实体中心记忆图 | 由情景归纳语义知识（人物属性/关系）随时间合并更新 | 长视频下按显著性裁剪（论文描述有限，**待核**） | 多轮自主检索（agentic search） |
| MemoryBank | 对话抽取记忆 | 重要性加权 | **艾宾浩斯遗忘曲线**（时间衰减+重要性） | 向量检索 |
| AWM | 从成功轨迹归纳 workflow | workflow 去重/合并 | 无显式遗忘 | 按任务相似度选择性注入提示 |
| Memory-R1 | RL 学到的 ADD/UPDATE/DELETE/NOOP | UPDATE 即巩固 | RL 学到的 DELETE | RL 优化的检索 agent |
| LightMem | 感觉记忆快速过滤→短期分组 | 「睡眠期」离线整合进长期记忆 | 阶段过滤即轻量遗忘 | 三级检索（主题感知 + 精排） |

**横向观察**：
1. 「巩固」正在从 MemGPT 式的**被动摘要**演进为 **主动改写**（A-MEM 演化、Mem0 整合、Memory-R1 的 RL 决策）；
2. 「遗忘」是最薄弱环节——除了 MemoryBank 的遗忘曲线和 Memory-R1 的 DELETE，几乎没有系统认真建模；M3-Agent 这类多模态系统因原始数据量巨大，遗忘/显著性裁剪是刚需但公开细节少；
3. 「检索」呈两条路线分叉：**单次相似度检索**（Mem0 系）vs **agentic 多轮检索**（M3-Agent、HippoRAG 的图扩散），后者在长视频/多跳场景占优。

---

## 四、评测基准

| 基准 | 领域 | 核心内容 | URL |
|---|---|---|---|
| **LoCoMo**（2024） | 长对话记忆 | 平均 300 轮 / 9K token / 最多 35 个 session 的超长对话；QA + 事件摘要 + 多模态对话生成三类任务；结论：即使长上下文模型与 RAG 也显著落后人类 | https://arxiv.org/abs/2402.17753 |
| **LongMemEval**（2024/ICLR 2025） | 聊天助手长期记忆 | 500 道精选问题，5 种记忆能力（信息抽取、跨会话推理、时序推理、知识更新、拒答/ abstention）；现有系统跨会话准确率掉 ~30%；并提出索引-检索-阅读的分解优化框架 | https://arxiv.org/abs/2410.10813 |
| **MemBench**（2025） | LLM agent 记忆综合评测 | 双层：事实性记忆 + 反思性记忆；双场景：参与式（自身对话）+ 观察式（旁观他人对话）；含 QA 与生成两类任务 | https://arxiv.org/abs/2506.21605 |
| **M3-Bench**（2025，随 M3-Agent 发布） | 多模态长视频记忆 | 100 段机器人第一视角长视频 + 920 段网络视频 QA，测跨模态情景/语义记忆 | https://arxiv.org/abs/2508.09736 |
| **Video-MME**（2024） | 长视频理解（非记忆专项） | 900 视频 / 254 小时 / 2700 QA，覆盖 11 秒到 1 小时；含长视频子集与字幕消融；**非记忆基准**，但长视频问答天然测"看过的内容记不记得"——M3-Agent 用其 long 子集对照验证记忆增益 | https://arxiv.org/abs/2405.21075 |
| ScreenSpot-v2 / AndroidWorld（简述） | GUI agent 能力 | MIRIX 论文用于验证截图记忆的下游可用性；本身测 GUI 定位/任务而非记忆，仅作记忆场景相关参考 | 未能核实精确 arXiv URL（ScreenSpot-v2 常引 arXiv:2501.XXXX，**待核**） |

补充：MemBench 注意与两篇同名工作区分——arXiv:2407.17095 是扩散模型记忆触发集、2504.06813 是 Arm 内存子系统吞吐基准，均与 agent 记忆无关。

---

## 五、2025–2026 多模态记忆新工作（图像/视频/屏幕截图）

按模态整理（均已核对 arXiv ID 存在，机制描述基于 arXiv 摘要）：

**GUI / 屏幕截图记忆**（2025 下半年起爆发，与 MIRIX、M3-Agent 共同构成"多模态记忆"主力）：
- **Chain-of-Memory**（arXiv:2506.18158）：显式建模 GUI agent 的短/长期记忆，配 111k 屏幕动作对数据集支持跨应用任务。
- **PAL-UI**（arXiv:2510.00413）：规划时主动回看历史截图（dual-level 摘要 + 检索工具），"active look-back"。
- **MementoGUI**（arXiv:2605.18652）：插件式智能体记忆控制器——文本摘要 + ROI 级视觉证据 + 情景轨迹检索的组合存取。
- **AGMem / 失效模式研究**（arXiv:2606.14106）：证明整屏截图记忆有系统性缺陷（state 级失败减少但 action 级失败变多），提出只存与动作相关的图像裁剪。
- **HyMEM**（arXiv:2603.10291）：脑启发图记忆——符号节点 + 轨迹嵌入耦合，支持多跳检索。
- **AndroTMem**（arXiv:2603.18429）：把交互轨迹表示为因果关联的"锚定状态"紧凑集合（Android 长时程）。

**视频/自我中心（egocentric）记忆**：
- **LightMem-Ego**（arXiv:2607.11487）：流式多模态记忆系统，处理第一视角视频+音频，目标部署在手机与 AI 眼镜上（进行中工作）。
- **M3-Agent / M3-Bench**（见上文，该方向的锚点工作）。

**记忆管理的训练化/效率化（2026 趋势，非多模态专属但直接适用）**：
- **BudgetMem**（arXiv:2602.06025）：RL 路由在预算档位间权衡记忆构建成本与性能。
- **HORMA**（arXiv:2606.11680）：层级记忆组织 + RL 训练的记忆导航。
- **LazyMem**（arXiv:2607.22690）：把记忆构建推迟到查询时，SFT+RL 只保留与查询相关的内容。
- **CMI-Mem**（arXiv:2607.20553）：用条件互信息奖励训 RL 记忆管理器，产出"相关、非冗余"记忆。

> 提示：2602–2609 编号条目来自 2026 年 2–9 月投稿，领域尚在快速流动，引用前建议再查最新版本；上表只列了经 arXiv API 检索确认存在的条目。

---

## 六、对我们设计的启示

1. **记忆接口要做成"可学习策略"而非硬编码管道**。Memory-R1、HORMA、BudgetMem 说明写入/遗忘/检索决策都可以用 RL 学；我们的系统应把「是否写入、如何整合、何时遗忘、何时删除」…… 即 ADD/UPDATE/DELETE/NOOP 暴露为动作空间，初期用规则兜底、预留 RL 升级路径。

2. **多模态记忆的瓶颈在「巩固」而非存储**。M3-Agent 证明"情景→语义归纳"（从看过的视频提炼人物/关系知识）是长视频问答的关键增益来源；MIRIX 的 6 类划分则提示：截图类数据必须与文本事实分开建模（Resource Memory / Knowledge Vault 的做法值得抄）。纯 append-only 截图库会重演 AGMem 指出的失效模式。

3. **遗忘与显著性必须显式建模，且多模态下成本更敏感**。MemoryBank 的遗忘曲线是唯一显式方案但过于粗糙；多模态原始数据（帧、截图）体积大 2–3 个数量级，写什么、裁剪什么（AGMem 的 action-relevant crop）、何时压缩，应当是一等设计问题而非事后优化。

4. **评测从第一天起用 LoCoMo/LongMemEval（文本）+ M3-Bench（视频）双轨跑**，否则无法知道自己与 Mem0/HippoRAG 基线可比；注意 LongMemEval 的 abstention（拒答）维度——记忆系统乱记比不记更伤。

5. **图结构不是可选装饰**。Mem0g（关系图）、HippoRAG（PPR 扩散）、Zep（时序图）三条独立路线都收敛到「记忆需要实体-关系结构才能多跳联想」；我们设计实体中心的多模态记忆图时应把**时间戳/有效期**放进边属性（Zep 的教训）。

---

## 附：条目核对状态汇总

| # | 条目 | 核实状态 |
|---|---|---|
| 1 | MemGPT / Letta | arXiv 页已核 |
| 2 | Mem0 / Mem0g | arXiv 页已核 |
| 3 | A-MEM | arXiv 页已核 |
| 4 | MemOS | arXiv 页已核 |
| 5 | HippoRAG / 2 | 两个 arXiv 页均已核 |
| 6 | MIRIX | arXiv 页已核（机构字段待核） |
| 7 | M3-Agent / M3-Bench | arXiv 页已核 |
| 8 | MemoryBank / Memory3 / MemInsight / AWM | 四篇 arXiv 页均已核 |
| 9 | LoCoMo / LongMemEval / MemBench / Video-MME | 均已核（MemBench 取 2506.21605，与同名扩散模型基准区分） |
| 10 | 2025–2026 新工作 | 18 篇，arXiv API 核实 ID 存在 + 摘要一行描述；2602+ 编号条目建议引用前再查最新版 |
| — | Zep / RMM / MemoRAG / Memory-R1 / LightMem | arXiv 页或 API 已核 |
| — | Mem-α | **未能核实**（arXiv API 未命中同名条目，未列正式条目） |

### 9. 补充：各系统一句话极简速查（面试/汇报用）

| 系统 | 一句话机制 |
|---|---|
| MemGPT | OS 式分页，LLM 自己管上下文进出 |
| Mem0 | 抽取→整合（ADD/UPDATE/DELETE/NOOP）管道，图变体 Mem0g |
| A-MEM | Zettelkasten 笔记 + 记忆间互相改写（演化） |
| MemOS | 记忆=系统资源，MemCube 统一调度明文/激活/参数三态 |
| HippoRAG 1/2 | KG + Personalized PageRank 模拟海马索引 |
| MIRIX | 6 类记忆 × 多 agent 分工管理，多模态 |
| M3-Agent | 视听流→实体中心多模态记忆，RL 端到端训练 |
| MemoryBank | 艾宾浩斯遗忘曲线 + SiliconFriend |
| Memory3 | 显式记忆（可检索稀疏 KV），2.4B 打更大模型 |
| MemInsight | 自主给记忆生成属性标注以利检索 |
| AWM | 从轨迹归纳 workflow 即程序性记忆 |
| Zep | 时序知识图谱（边带有效期与版本） |
| RMM | 前瞻+回顾双反思，在线 RL 调检索 |
| Memory-R1 | RL 学记忆管理动作（152 样本起效） |
| LightMem | 三阶段记忆 + 睡眠期离线巩固 |

---

## 三·补、时间线：领域演进脉络（2023 → 2026）

```
2023  MemGPT（分页范式）──┐
      MemoryBank（遗忘）──┼── 第一代：上下文管理与提示工程记忆
                          │
2024  HippoRAG（图+PPR）──┐│
      Memory3（显式KV）───┤│
      LoCoMo/LongMemEval ─┘│── 第二代：结构化记忆 + 严肃评测
      AWM（程序性记忆）────┤
      MemoRAG（全局记忆）──┘
                          │
2025  Mem0/A-MEM/MemOS/MIRIX/M3-Agent/Zep/RMM/MemInsight
                          │── 第三代：agentic 记忆（自主整合/演化/多模态）+ 记忆操作RL化
      Memory-R1 / LightMem
                          │
2026  GUI记忆爆发（Chain-of-Memory/PAL-UI/AGMem/HyMEM...）
      效率化与训练化（BudgetMem/LazyMem/HORMA/CMI-Mem）
      复现研究入场（naive RAG vs LightMem）── 领域开始"挤水分"
```

三个拐点：
1. **2024 中**：评测基准（LoCoMo、LongMemEval）成熟，记忆系统从"演示"进入"可比较"阶段；
2. **2025 上**：MemOS/MIRIX/M3-Agent 三篇把记忆推向「系统化 + 多模态」，agentic memory 成为独立研究方向；
3. **2025 下–2026**：RL 化（Memory-R1 系）与效率化（LightMem/BudgetMem 系）两条主线并行，同时出现第一批负结果/复现论文——这是领域走向成熟的标志。

---

## 六·补、开放问题（写论文/立项可参考）

1. **遗忘的理论基础缺失**：何时忘、何时压缩、保留多少，均无原则性框架；MemoryBank 的曲线只是隐喻借用。多模态下原始帧体积是文本的 100–1000 倍，这个问题从"锦上添花"变成"成本生死线"。
2. **记忆的可信性/来源追踪**：MemOS 的 MemCube 带来源与版本元数据是唯一步骤；记忆被检索后如何审计（agent 说了错话是记忆错还是推理错）尚无系统研究。
3. **多 agent 共享记忆**：所有系统默认单 agent 视角；多智能体间记忆一致性、并发写冲突、权限隔离几乎空白。
4. **跨模态对齐的记忆单元**：M3-Agent 以实体为中心组织，但对"同一实体的视频证据 vs 文本证据"如何合并权重、冲突时信谁，没有形式化。
5. **长记忆的隐私与被遗忘权**：用户要求删除某记忆时，参数化记忆（MemP）与摘要衍生记忆（rolling summary）中的残留如何清除，无系统研究。

---

## 七、与本项目（多模态记忆系统）的对接建议

> 本节把上面的学术结论映射到我们的系统设计决策，按"必须抄 / 应该抄 / 慎抄"分级。

### 必须抄（已被反复验证）
1. **Mem0 的 ADD/UPDATE/DELETE/NOOP 动作集**作为记忆写入接口——事实上的行业标准，后续 RL 化兼容。
2. **实体中心的多模态记忆组织**（M3-Agent）：以实体为节点聚合视觉/听觉/文本证据，跨模态问答靠它。
3. **LongMemEval 式的分能力评测**：信息抽取/跨会话推理/时序推理/知识更新/拒答五个维度分开测，避免"一个总分糊弄所有场景"。
4. **时间戳进记忆元数据**（Zep 的教训）：所有记忆条目和关系边都带采集时间与有效期，时序推理与知识更新全靠它。

### 应该抄（方向正确、实现可自选）
5. **三层过滤管线**（LightMem 的感觉→短期→长期）：多模态原始数据必须先过显著性过滤再入库，别全存。
6. **记忆演化/整合**（A-MEM 或 Mem0 二选一）：入库时与存量比对，冲突时更新而非追加；我们的场景建议 Mem0 式局部整合（可控）而非 A-MEM 式全局传播（难收敛）。
7. **检索的 agentic 化预留**：先做向量检索打基线，但接口上支持多轮检索循环（M3-Agent 式），长视频/多跳场景是刚需。
8. **ROI 级视觉证据**（MementoGUI/AGMem 的共识）：截图记忆不要存整帧原文+整帧原图，存"文本摘要 + 动作相关图像裁剪"。

### 慎抄（有已知问题或证据不足）
9. **LightMem 的准确率优势**：2026 复现研究称 naive RAG 可持平，其真实优势在 token 效率——引用需带限定，借鉴其管线结构可以，别期待精度收益。
10. **艾宾浩斯遗忘曲线**（MemoryBank）：隐喻大于实效，时间衰减函数应自己按数据形态设计并做消融。
11. **参数记忆（MemP）**：MemOS 方向正确但工程门槛极高（需访问模型权重）；除非自研模型，否则短期只做 MemG+MemC 两态。

### 里程碑建议（对齐学术节奏）
- M1：文本记忆（Mem0 式管道）+ LongMemEval 打分 → 对标公开基线；
- M2：多模态接入（截图+ROI）+ M3-Bench 子集 → 验证跨模态增益；
- M3：遗忘与压缩策略消融 → 补齐领域空白（可发论文的点）。

---

## 附 2：参考文献编号索引（按正文出现顺序）

1. MemGPT — arXiv:2310.08560
2. Mem0 / Mem0g — arXiv:2504.19413
3. A-MEM — arXiv:2502.12110
4. MemOS — arXiv:2507.03724
5. HippoRAG — arXiv:2405.14831
6. HippoRAG 2 — arXiv:2502.14802
7. MIRIX — arXiv:2507.07957
8. M3-Agent / M3-Bench — arXiv:2508.09736
9. MemoryBank — arXiv:2305.10250
10. Memory3 — arXiv:2407.01178
11. MemInsight — arXiv:2503.21760
12. AWM — arXiv:2409.07429
13. Zep — arXiv:2501.13956
14. RMM — arXiv:2503.08026
15. MemoRAG — arXiv:2409.05591
16. Memory-R1 — arXiv:2508.19828
17. LightMem — arXiv:2510.18866（复现：arXiv:2607.29104）
18. Memory 综述 — arXiv:2404.13501
19. LoCoMo — arXiv:2402.17753
20. LongMemEval — arXiv:2410.10813
21. MemBench — arXiv:2506.21605
22. Video-MME — arXiv:2405.21075
23. Chain-of-Memory — arXiv:2506.18158
24. PAL-UI — arXiv:2510.00413
25. MementoGUI — arXiv:2605.18652
26. AGMem（失效模式）— arXiv:2606.14106
27. HyMEM — arXiv:2603.10291
28. AndroTMem — arXiv:2603.18429
29. LightMem-Ego — arXiv:2607.11487
30. BudgetMem — arXiv:2602.06025
31. HORMA — arXiv:2606.11680
32. LazyMem — arXiv:2607.22690
33. CMI-Mem — arXiv:2607.20553
