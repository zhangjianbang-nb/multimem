# 多模态(图像/视频/截图)智能体记忆 · 技术调研笔记

> 日期:2026-10-08。来源:arXiv 摘要页、GitHub README、官方文档(抓取自 curl/WebFetch)。
> 凡未能逐字核实原文细节的,均标注「(未逐字核实)」或「(据摘要/README)」。

---

## 1. 多模态记忆条目如何表示

### 1.1 图像条目:三层表示(业界共识做法)

一条图像记忆通常由三部分组成,检索时双通道互补:

- **原图(或缩略图)**:供 VLM 后续 re-read/喂给多模态模型;磁盘成本高,常缩放到短边 512~1024px。
- **文本 caption**:VLM 生成的自然语言描述,可被纯文本嵌入索引、也可被 LLM 直接推理。
- **多模态嵌入**:CLIP/SigLIP 类模型的图像向量,支持"以文搜图"。

代表系统:
- **Mem0**(开源,Apache-2.0):图像经 VLM(如 GPT-4o)提取"事实文本"后**只存文本记忆**,不存原图。
  需 `enable_vision: True`,否则图像 turn 被静默丢弃;支持 URL 与 base64 两种输入,格式 JPEG/PNG/WebP/GIF。
  https://docs.mem0.ai/open-source/features/multimodal-support (已核实)
- **M3-Agent**:记忆图节点直接携带视频帧/音频引用,实体中心组织。
  https://github.com/ByteDance-Seed/m3-agent (已核实 README)

### 1.2 视频记忆:分段与采样策略

- **均匀帧采样**:每 N 秒/每秒 1 帧取一帧。简单,但对短促事件会漏采。
- **关键帧/场景切分**:PySceneDetect 等工具做镜头边界检测,每场景取代表帧;
  或用 VLM 对相邻帧相似度聚类取关键帧。适合演示、GUI 操作类视频。
- **M3-Agent 的做法(已核实 README)**:
  1. 视频**切成 30 秒片段**(ffmpeg -c copy);
  2. 对每段做 VLM 描述 + **人脸检测**(face detection)+ **说话人分离**
     (speaker diarization,ModelScope 的 speech_eres2netv2 声纹模型);
  3. 视频流+音频流并行"感知",在线写入记忆图(memorization 进程)。
  即:**事件分段 = 固定 30s 窗口 + 逐段多模态描述 + 说话人/人脸归并到实体**。

### 1.3 视频→事件分段的一般管线

```
video → 切片(15~30s) → 每片:关键帧采样 → VLM caption(含 OCR/人脸/动作)
      → 音频:ASR 转写 + diarization → 事件描述(text)
      → 事件写入记忆库(episodic)→ 定期蒸馏进语义记忆
```

参考实现:M3-Agent(上述);VideoAgent(https://arxiv.org/abs/2405.16435,
未逐字核实,LLM+VLM 双代理在均匀采样帧上迭代定位证据帧)。

---

## 2. 嵌入模型选型(维度/多语言/协议)

| 模型 | 图文同空间 | 维度 | 多语言 | 协议 | 备注 |
|---|---|---|---|---|---|
| CLIP (OpenAI, 2103.00020) | 是 | 512/768/1024 (ViT-B/L) | 仅英文 | MIT(权重已核实开源) | 事实标准基线,4 亿图文对 |
| SigLIP (Google, 2303.15343) | 是 | 768/1024/1152 | 仅英文 | Apache-2.0(权重) | sigmoid loss,无需全局 batch;小 batch 也稳(摘要已核实) |
| SigLIP2 (2502.14786) | 是 | 同上 | 仅英文 | Apache-2.0(未逐字核实) | 2025 更新版 |
| jina-clip-v2 (2412.08802) | 是 | **1024**(Matryoshka 可截到 64) | **89 语言** | **CC-BY-NC-4.0(非商用!)** 商用需走 Jina/AWS/Azure/GCP API(模型卡已核实) | 文本塔同 jina-embeddings-v3;支持图文检索+文档 OCR 场景 |
| BGE-M3 (2402.03216) | 否(纯文本) | 1024 | 100+ 语言 | MIT(据常见认知,未逐字核实) | dense+sparse+multi-vector 三合一;长文本 8192 tok;无图像能力 |
| OpenAI text-embedding-3 | 否 | 1536/3072 | 优 | 商业 API | **text 与 image 不在同一空间**——OpenAI 无公开图像嵌入;Claude 亦无 |
| Cohere Embed v4 / multimodal | 是(据官方宣传,未逐字核实) | 1536 | 100+ | 商业 API | 文档型多模态嵌入代表 |
| Voyage multimodal (voyage-multimodal-3) | 是(未逐字核实) | 1024 | 英文为主 | 商业 API | 截图检索常用 |

**关键结论**:
- **OpenAI 的 text-embedding 与图像不在同一空间**(OpenAI 不提供图像 embedding API);
  需要图文同空间只能用 CLIP/SigLIP/Jina-CLIP-v2/Cohere Embed v4/Voyage。
- 中文场景首选:jina-clip-v2(89 语,注意 NC 协议)或自部署 SigLIP2(英文强)
  + 中文 VLM caption 兜底。
- BGE-M3 只管文本,适合做"文本记忆主嵌入",图文对齐交给 CLIP 系。

来源:
- CLIP: https://arxiv.org/abs/2103.00020 (已核实摘要)
- SigLIP: https://arxiv.org/abs/2303.15343 (已核实摘要)
- jina-clip-v2: https://arxiv.org/abs/2412.08802 ; 模型卡 https://huggingface.co/jinaai/jina-clip-v2 (已核实:1024 维、89 语言、CC-BY-NC-4.0)
- BGE-M3: https://arxiv.org/abs/2402.03216 (已核实摘要)

---

## 3. VLM captioning 管线

### 3.1 caption 模型选型

| 模型 | 出处 | 特点 |
|---|---|---|
| Qwen2.5-VL / Qwen3-VL | https://github.com/QwenLM/Qwen2.5-VL (未逐字核实) | 开源,中文强,支持任意分辨率、视频、grounding;离线首选 |
| Qwen-VL(初代) | https://arxiv.org/abs/2308.12966 (摘要已核实) | 图-文-box 三元组对齐,支持定位与 OCR |
| InternVL 2.5/3 | https://github.com/OpenGVLab/InternVL (未逐字核实) | 开源,高分辨率动态分块,文档理解强 |
| GPT-4o / GPT-4o-mini | https://platform.openai.com/docs/guides/vision (未逐字核实) | 商业,质量高;Mem0 默认走此路线 |
| Gemini 2.x Flash | https://ai.google.dev (未逐字核实) | 商业,长视频(1M token 上下文)直接吃整段视频 |

### 3.2 caption 写什么(记忆导向的 prompt 结构)

一条好的记忆 caption 应包含:**场景(时间/地点)+ 人物实体(姓名/身份)+
动作/事件 + 文字内容(OCR)+ 可检索关键词**。M3-Agent 还会把说话人 diarization
结果(谁在什么时候说了什么)并入事件节点(README 已核实)。

### 3.3 caption+原图双通道检索的得失

**得**:
- caption 走文本嵌入,与 query 语义对齐好(问题常是抽象的:"用户上次怎么配置的?");
- 原图嵌入兜底 caption 漏掉的视觉细节(颜色、布局、logo);
- LLM 推理时 caption 比原图省 token。

**失**:
- caption 是有损压缩:细节、空间关系、画面文字可能写丢或写错;
- 双通道要在 rerank 层融合两路分数(分数尺度不同,需归一化);
- 存储/索引翻倍;VLM caption 管线本身有成本和延迟。
- 经验折中(据 Mem0/M3-Agent 做法,未逐字核实):**默认只存 caption 文本,
  原图落盘留引用路径,仅命中后按需重读原图**。

---

## 4. 多模态记忆系统实例

### 4.1 M3-Agent(字节 Seed,ICLR 2026)— 三层记忆的代表

论文:"Seeing, Listening, Remembering, and Reasoning: A Multimodal Agent with
Long-Term Memory" https://arxiv.org/abs/2508.09736 (摘要已核实)
代码: https://github.com/ByteDance-Seed/m3-agent

- **输入**:实时视频流+音频流。**记忆组织:实体中心的多模态图(memory graph)**。
- 记忆层次(据论文摘要与 README,spatial 一词在早期宣传材料中出现;摘要原文只列
  episodic+semantic 两类——**注意:网传"三层"中 spatial 记忆在正式摘要未单列**):
  - **情景记忆 episodic**:每 30s 片段的事件描述节点(带时间戳、帧引用、说话人);
  - **语义记忆 semantic**:实体节点(人物/物体/概念)及其属性,由情景逐步归并推理;
  - **空间记忆 spatial**:环境布局(README 中以 intermediate_outputs/人脸位置间接体现;
    论文页未逐字核实其独立存储结构)。
- **控制进程**:指令→多轮 think-retrieve 循环,从记忆图检索。
- 训练:memorization 与 control 两个模型都用 RL 训练
  (HuggingFace: ByteDance-Seed/M3-Agent-Memorization、M3-Agent-Control)。
- 代码栈依赖:qwen-omni-utils + transformers(fork 版本),ffmpeg 切片,
  ModelScope 声纹模型——**全链路可离线复现**。

### 4.2 Mem0 多模态支持(已核实文档)

- 图像(PNG/JPEG/WebP/GIF)通过 `{"type": "image_url", ...}` 传入 `client.add()`;URL 或 base64。
- `enable_vision: True` 后,VLM 把图像**抽取为文本事实存入标准记忆**,
  与文本记忆同一套向量库/检索——**不做原图级 embedding 检索**。
- `vision_details: auto/low/high` 控制视觉细节档位。
- https://docs.mem0.ai/open-source/features/multimodal-support
- 商业版另有 Multimodal Retrieval cookbook(图文 PDF 混合记忆)。

### 4.3 GUI / computer-use 类截图记忆

- **Claude computer-use(Anthropic)**:官方无独立"截图记忆"模块;社区常见模式是
  每步截图→压缩→保留最近 K 步截图在上下文内,更早步骤蒸馏成文本轨迹
  (未逐字核实,据 anthropic docs computer-use 指南)。
  https://docs.anthropic.com/en/docs/agents-and-tools/computer-use
- **Open Interpreter**:开源(AGPL-3.0,未逐字核实);其 OS mode 每步截图→VLM 决策,
  历史主要靠消息列表,**没有专门的长期视觉记忆库**(据项目 README 印象,未逐字核实)。
  https://github.com/OpenInterpreter/open-interpreter
- **GUI Agent 通用套路**(综合 4.1/4.2):截图(带时间戳+动作前后对比)→ VLM 提取
  "做了什么、界面元素状态" → 存文本记忆 + 缩略图;查询时"最近 N 张原图 + 蒸馏轨迹"。

---

## 5. 检索策略

### 5.1 图文跨模态检索怎么落地

- 同一 CLIP/SigLIP 空间内:text query → 文本编码 → 与图像向量内积/余弦取 top-k。
- 双通道(见 §3.3):文本记忆索引用 BGE-M3/text-embedding,图像索引用 CLIP 系;
  query 分别打两路,合并去重后 rerank。
- **混合检索标配**:dense(向量)+ sparse(BM25/BGE-M3 sparse)融合,时间敏感查询必须带。

### 5.2 时间过滤

- 记忆条目带 `created_at` / `event_time` 字段;检索时先向量 recall 再按时间窗过滤,
  或直接在 SQL/元数据层(如 "last 7 days")预过滤。
- Generative Agents 的 recency 指数衰减见 §6;M3-Agent 的事件节点带视频时间戳,
  检索时可做"发生在指令前"约束(据论文摘要推断,未逐字核实)。

### 5.3 多模态 rerank

- 实用做法:top-50 候选的 **caption 文本**喂给 LLM(或多模态 rerank 模型)重排;
  必要时把原图一并给 VLM 做相关性判断(成本高,只对 top-5~10)。
- 专用跨模态 reranker:Jina reranker v2(未逐字核实)、Cohere Rerank 3.5(未逐字核实);
  开源自托管常见做法是 SigLIP 分数 + BGE-M3 分数加权融合。

---

## 6. 遗忘与巩固

### 6.1 情景→语义蒸馏

- **Generative Agents**(2304.03442,摘要已核实):记忆流(memory stream)+ **reflection**:
  当某类观察累计重要性超过阈值时触发合成,由 LLM 从多条记忆归纳出高阶结论("用户偏好…")。
  https://arxiv.org/abs/2304.03442
- **M3-Agent**:semantic 记忆由 memorization 进程在观看中持续把实体/事件归并、
  推理更新到实体中心图(README+摘要已核实)。
- 落地:每天/每 N 条事件触发一次离线任务——把同实体/同主题的情景条目喂 LLM,
  产出语义条目并回写;情景条目保留但降权。

### 6.2 衰减与评分函数(Generative Agents 公式)

检索分数 = α·recency + β·importance + γ·relevance(三式各自 min-max 归一,论文中权重均等)。

- **recency**:指数衰减 exp(-λ·Δt),λ≈0.995(以小时计,未逐字核实具体常数);
- **importance**:LLM 打分 1~10(poem 得 1, breakup 得 9)写入条目;
- **relevance**:语义余弦相似度。
- 多模态扩展:图像条目的 importance 可由 caption 内容打分;原图访问后 recency 刷新。

### 6.3 遗忘策略(工程实践,综合印象,未逐字核实)

- 原图分层存储:热(最近 7 天,SSD)/冷(对象存储,仅留缩略图+caption);
- 低 importance + 老旧条目降采样帧率或仅保留文本;
- 反事实去重:同一实体/场景反复出现时,合并进语义条目后情景条目可压缩。

---

## 7. 评测 benchmark

| Benchmark | 出处 | 规模/形式 | 用途 |
|---|---|---|---|
| **Video-MME** | https://arxiv.org/abs/2405.21075 (摘要已核实) | 900 视频/254 小时/2700 QA;短(11s~2min)、中(4~15min)、长(16~60min);6 域 30 子类;可选 subtitles+audio | 长视频理解主基准;M3-Agent 在 Video-MME-long 上 +5.3% |
| **EgoSchema** | https://arxiv.org/abs/2308.09126 (摘要已核实) | 5000+ MCQ,3 分钟 egocentric 片段,5 选 1,源自 Ego4D | 极长时序理解;引入"时间证书集"概念 |
| **M3-Bench** | 随 M3-Agent (2508.09736) | M3-Bench-robot:100 段第一视角机器人视频;M3-Bench-web:920 网络视频;开放式 QA,考察人物理解/知识抽取/跨模态推理 | 直接测多模态长期记忆,数据在 HF: ByteDance-Seed/M3-Bench |
| **LoCoMo** | https://arxiv.org/abs/2402.17753 (摘要已核实) | 超长对话(平均 300 turn/9K token/最多 35 会话);**含图像分享回合**,有 multi-modal QA 与 event summarization 任务 | 多模态对话记忆(文本+图像混合)最接近的基准 |
| MLVU / LVBench | 2406.04264 / 2406.08035(未逐字核实) | 长视频理解,更长上下文 | 补充参考 |

---

## 8. 落地配方建议(纯 Python、默认离线可跑)

### 8.1 最小可行技术栈(MVP)

```
存储:    SQLite(元数据/时间戳/importance)+ 本地文件系统(原图,按日期分目录)
向量:    sqlite-vec 或 FAISS(无需服务);文本与图像各一张表
嵌入-文本: BGE-M3 (MIT, 1024 维, sentence-transformers 可跑) 或 bge-small-zh(轻量)
嵌入-图像: SigLIP2-base(开源可离线)或 jina-clip-v2(89 语言但 CC-BY-NC,注意协议)
caption:  Qwen2.5-VL-7B-Instruct(vLLM/transformers 离线)或预留 GPT-4o API 开关
视频分段: ffmpeg 每 30s 切片 + PySceneDetect 关键帧(每片 3~6 帧)
ASR(可选): faster-whisper(离线)
rerank:   BGE-M3 分数 × SigLIP 分数加权融合 + 时间衰减,top-10 给 LLM 重排
```

### 8.2 关键设计决定

1. **条目 schema**(一张表两用):
   `id, modality(image/frame/event), text_caption, image_path, text_emb, image_emb,
   created_at, event_time, importance, source, session_id`
2. **检索两阶段**:recall(dense top-50,双路)→ score(0.5·text_sim + 0.3·image_sim
   + 0.2·recency_exp)→ 时间过滤 → LLM rerank top-10。
3. **先文本后原图**:caption 检索为主,原图嵌入兜底;原图仅在 top-k 命中后按需读。
4. **巩固任务**:每晚对当日条目做 reflection(Generative Agents 阈值触发),
   产出语义条目(用户习惯/实体知识)。
5. **离线优先**:所有默认依赖(ffmpeg、sentence-transformers、sqlite)纯 pip+apt 可装;
   VLM/ASR 模型权重首次下载后即离线。

### 8.3 演进路线

- v0:截图+文本,单用户,bge-small + SigLIP2,sqlite-vec;
- v1:接入视频(30s 切片+diarization),对齐 M3-Agent 的 episodic 图;
- v2:实体中心记忆图(参考 M3-Agent memory graph)+ RL/反思自动化。

---

## 附:本文引用来源清单

- M3-Agent 论文 https://arxiv.org/abs/2508.09736 ;仓库 https://github.com/ByteDance-Seed/m3-agent
- Mem0 多模态文档 https://docs.mem0.ai/open-source/features/multimodal-support ;仓库 https://github.com/mem0ai/mem0
- Video-MME https://arxiv.org/abs/2405.21075
- LoCoMo https://arxiv.org/abs/2402.17753
- EgoSchema https://arxiv.org/abs/2308.09126
- Generative Agents https://arxiv.org/abs/2304.03442
- CLIP https://arxiv.org/abs/2103.00020 ;SigLIP https://arxiv.org/abs/2303.15343
- jina-clip-v2 https://arxiv.org/abs/2412.08802 ;模型卡 https://huggingface.co/jinaai/jina-clip-v2
- BGE-M3 https://arxiv.org/abs/2402.03216
- Qwen-VL https://arxiv.org/abs/2308.12966
- VideoAgent https://arxiv.org/abs/2405.16435 (未逐字核实)
- Anthropic computer-use https://docs.anthropic.com/en/docs/agents-and-tools/computer-use (未逐字核实)
- Open Interpreter https://github.com/OpenInterpreter/open-interpreter (未逐字核实)
