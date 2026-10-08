# MultiMem 设计方案

一个面向多模态智能体的记忆库:情景(episodic)/ 语义(semantic)/ 程序性(procedural)
三类记忆,统一存储文本与图像/视频帧,混合检索,可预演的巩固与遗忘,零硬依赖
(numpy + pillow 之外无强制依赖),并自带一个无 SDK 的 MCP stdio 服务器。

本设计吸收了 2023–2026 年记忆系统的主流结论。调研笔记见
`docs/research/`(学术侧 / 工程侧 / 多模态侧三份),设计决策逐条标注出处。

## 1. 设计原则

1. **ADD-only 写入 + 检索端融合**(mem0 v3 的反转):写入路径不做 LLM 决策的
   UPDATE/DELETE,条目只增;冲突与过期交给双时序字段和遗忘 API。这让存储层
   无脑快,也让审计可回放。
2. **双时序(bi-temporal)字段**(Graphiti 的 `valid_at/invalid_at` 模式):
   `created_at` 记录"何时记住何时知道",`valid_at/invalid_at` 记录"事实何时
   成立/失效"。冲突事实只失效不删除,`search(valid_only=True)` 默认过滤。
3. **文本为纲、二进制为附件**:每条记忆的可检索主体是自然语言文本;图片/视频
   字节存盘仅留路径与 caption。这符合主流系统现状(Mem0/M3-Agent 都是把视觉
   转成文本事实进检索;原图在命中后按需重读)。真正的图文同空间向量(CLIP/
   SigLIP/jina-clip-v2)通过 `EmbeddingProvider` 插槽接入,接口不变。
4. **巩固是显式端点而非后台黑魔法**(MIRIX `auto_dream(dry_run=True)` 的启发):
   `consolidate_semantic()` 可 dry-run 预演,返回将要做出的动作列表,再决定
   是否落盘。情景记忆永远保留,作为语义事实的审计底稿。
5. **遗忘分两级**:`forget(id, hard=False)` 软失效(默认,双时序合规)与
   `hard=True` 物理删除;`decay()` 做时间衰减批处理(Generative Agents 的
   importance 衰减思想)。

## 2. 数据模型

```
MemoryItem
├── kind:        episodic | semantic | procedural
├── content:     str            # 可检索正文(图片记忆 = caption)
├── modality:    text | image | video | audio | multimodal
├── tags:        [str]
├── importance:  float 0..1
├── created_at:  ISO8601        # 知道的时间
├── valid_at / invalid_at: ISO8601 | None   # 事实成立区间(双时序)
├── attachments: [Attachment]   # media_type / path / caption / meta
└── meta:        dict           # 自由字段,如 {"video": "x.mp4", "consolidated_from": [ids]}
```

存储为 SQLite(WAL)+ `embeddings` 表(float32 blob,按 embedder 名隔离,
切换嵌入模型时旧向量保留但不参与检索)。选 SQLite + numpy 暴力余弦是刻意的:
智能体记忆量级在 10^3–10^6,256–4096 维暴力扫描 <50ms,换来零部署依赖;
规模超出时把 `MemoryStore` 换成 faiss/lancedb 后端即可,接口已收敛。

## 3. 检索管线

```
query ──► embed ──► cosine over stored vectors ──► 融合排序 ──► top-k
                                    │
        score = w_rel·relevance + w_rec·recency + w_imp·importance
        (Generative Agents 配方;权重默认 0.65/0.20/0.15,halflife=168h)
```

- `kinds`/`tags` 过滤在向量扫描前裁剪;`valid_only=True` 剔除已失效事实。
- `export_context(query, budget_chars)` 把 top-k 渲染成可直接塞进 prompt 的
  上下文块,带字符预算截断(memory-in-prompt 模式)。
- 多跳扩展(multi-hop expansion, HippoRAG 式)预留为 P2:当前 tags 就是
  实体锚点,后续可做 tag→item 的二跳检索。

## 4. 多模态策略

- **图片**:`add_image(path, caption=...)`。caption 缺省时可用 `caption_fn`
  (接 Qwen-VL/InternVL/GPT-4o)自动生成;离线默认退化成文件名+尺寸伪 caption,
  保证 API 永远可用。检索发生在 caption 文本上。
- **视频**:`add_video(path, max_frames=8)` 用 ffmpeg thumbnail 滤镜抽关键帧,
  每帧一条图片记忆,`meta["video"]` 指回源文件。ffmpeg 缺失时优雅降级为
  单条文本记忆,不抛异常。
- **跨模态向量**:`OpenAICompatEmbedder` 指向任意 /v1/embeddings 端点;
  要图文同空间就接 jina-clip-v2 或自部署 SigLIP(注意 jina-clip-v2 权重是
  CC-BY-NC-4.0,商用走其 API)。图像字节→数据 URI 的转换器
  (`MediaStore.to_data_uri`)已备好,供多模态 VLM 重读。

## 5. 巩固与遗忘

- `consolidate_semantic(min_cluster=2, dry_run=False)`:对情景记忆做嵌入
  近重复聚类,≥min_cluster 的簇生成"3x recurring episode: …"式语义事实,
  `meta["consolidated_from"]` 记录来源 ID(可溯源)。
- `decay(floor_importance, drop_below)`:importance 随时间半衰;跌破阈值
  且刷新无效的条目物理删除。
- 遗忘曲线/Memory-R1 式 RL 决策属 P3:当前动作集(ADD/INVALIDATE/DELETE)
  已与其接口对齐,后续可把决策器换成可学习策略。

## 6. MCP 服务器

`multimem serve ./dir` 启动 stdio JSON-RPC(手工实现 Content-Length 分帧,
零 SDK 依赖)。工具:`add_memory` / `search_memory` / `list_memories` /
`forget_memory`。任何 MCP 客户端(Claude Desktop、ZCode 等)可直接挂载。

## 7. 明确不做(v0.1)

- 不做图记忆数据库(graphiti 的时序边)——tags+meta 覆盖 80% 用例,上图为 P2。
- 不做跨用户/多智能体权限隔离(MemOS 的 cube 模型)——单 agent 单库。
- 不内置 LLM 抽取管线(langmem 式的"从对话自动抽取事实")——那是调用方
  agent 的职责;本库只负责可靠地存取与遗忘。

## 8. 测试与验收

45 条 pytest(types/store/encode/memory/media/mcp/cli 七层),外加
`examples/demo.py` 端到端脚本(文本+图像+巩固+遗忘全链路,exit 0 为过)。
