# MultiMem v0.2 设计方案（跨模态检索 + 检索质量升级）

v0.1 确立了「三类记忆 + 双时序 + 文本为纲」的骨架。v0.2 解决 v0.1 明确留白的
三件事：**真·跨模态检索**（图查文/文查图同空间）、**视频时间戳索引**、
**检索打分与索引的正确性与可扩展性**。设计原则不变：零强制重依赖、
可选依赖优雅降级、每个决策标注出处。

## 0. v0.2 范围与动机

| 能力 | v0.1 现状 | v0.2 目标 |
|---|---|---|
| 图文跨模态 | 只有 caption 文本检索，CLIP 留了插槽 | `ImageEmbedder` 一等公民：图像按视觉向量入库，文本查询跨模态命中 |
| 检索打分 | **有 bug**：relevance 双重除范数（x·q/‖x‖²≠cosine，已修） | 归一化空间内点积=余弦，单测锁死 |
| 视频记忆 | ffmpeg 抽帧、每帧一条、无时间戳 | 帧带 `t_seconds` 时间戳 + 场景边界采样 + 时间过滤检索 |
| 重要性 | 静态 0..1 由调用方给 | 使用计数反馈：命中即 +importance（Generative Agents retrieval→writing 闭环）|
| 索引 | numpy 暴力（10⁶ 量级 <50ms） | 保持默认；可选 sqlite-vec 虚拟表（N>10⁵ 时）|
| embedding 缓存 | 存 SQLite blob（已有） | 补内容寻址去重：相同 caption/hash 不重复调用 API |

## 1. 跨模态检索设计

### 1.1 双空间模型

引入两个 Provider 协议（沿用 v0.1 `EmbeddingProvider` 风格，Protocol 零依赖）：

```python
class TextEmbedder(Protocol):     # name, dim, embed(texts) -> [n, d]
class ImageEmbedder(Protocol):    # name, dim, embed_images(paths) -> [n, d]
```

存储层按 `space` 维度扩展 embeddings 表：`(item_id, embedder, space, dim, vector)`。
- `space="text"`：caption/正文向量（文本塔）。
- `space="vision"`：图像视觉向量（图像塔）。
- **查询时**：文本 query 先在 `text` 空间检索；若库内存在 `vision` 空间向量且
  配置了跨模态（`cross_modal=True`），文本向量与视觉向量**不在同一空间**时
  不直接比——跨模态同空间要求两端用同一个 CLIP 类模型（SigLIP2 / jina-clip-v2 /
  CN-CLIP）。因此 Provider 是**成对的**：`ClipStyleEmbedder` 同时实现两个协议，
  文本塔与图像塔共享 name（`siglip2/`），检索时文本向量直接对视觉向量点积。

### 1.2 检索融合公式（对齐 03-multimodal.md §5.1）

```
final_score = α·max(cos(text_q, text_v), cos(text_q, vision_v))   # 同一 CLIP 模型内
            + w_rec·recency + w_imp·importance
```
即：一条图片记忆带两个向量（caption 文本向量 + 图像视觉向量），query 文本向量
在 CLIP 空间内与两者比、取 max（caption 命中或视觉命中都算数）。这解决了
v0.1 「caption 写不好就永远检索不到」的单通道缺陷，是 M3-Agent/截图记忆系统
的通用做法（03-multimodal.md §3.3：caption+原图双通道）。

### 1.3 降级矩阵（强制）

| 环境 | 行为 |
|---|---|
| 无任何 embedder 配置 | HashingEmbedder 只建 text 空间，行为=v0.1（修 bug 后）|
| 只配 OpenAI 兼容文本端点 | text 空间用远程向量，vision 空间缺省 |
| 配 CLIPStyle（本地 transformers 或 onnxruntime） | text+vision 双空间，真跨模态 |
| 任一调用失败 | 优雅降级到低空间，不抛异常（对齐 v0.1 add_video 的降级纪律）|

## 2. 视频时间戳索引

- `sample_video_keyframes` 升级：优先 `ffmpeg select='gt(scene,0.3)'` 场景切变
  采样（场景边界比等间隔更能代表事件，03-multimodal.md §1.2）；失败退回
  `thumbnail` 等间隔；再失败退回单条文本（v0.1 行为不变）。
- 帧记忆 `meta` 新增：`{"video": "...", "t_seconds": 12.4, "frame_idx": 3}`。
- `search(query, video_within=(start_s, end_s))`：按 meta 时间窗过滤，支撑
  「视频第几分钟发生了什么」式检索；时间过滤在向量扫描前裁剪（与 kinds/tags 同层）。

## 3. 使用反馈闭环（Generative Agents retrieval→importance）

- `MemoryItem.use_count: int = 0`（新列，迁移兼容：旧行缺列按 0 处理）。
- `search(..., feedback=True)` 命中的条目 `use_count += 1` 且 importance
  上调 `min(1.0, importance + 0.02)`（写回 SQLite，Generative Agents 的
  importance 由检索与写入共同塑造的思想）。
- `decay()` 公式加入 use_count 项：频繁被用的记忆衰减慢（0.5+0.5·recency 基础上
  `*min(1, use_count/10)` 的保护因子）。

## 4. 索引与缓存

- **默认**仍是 numpy 暴力（10³–10⁵ 量级最稳妥，零依赖）；预归一化缓存：
  rows_with_vectors 返回前统一 L2 归一化一次，查询端直接点积（消除 v0.1
  每行重复 norm 的浪费）。
- **sqlite-vec 可选**：`Memory(store_backend="sqlite-vec")`，import 失败自动回退
  numpy 并发 warning。仅当 N>10⁵ 且安装了 `sqlite-vec` 时启用。
- **embedding 去重**：`add_image` 若图像 sha1 已存在且 embedder 相同，直接
  复用已存向量不重算（media.py 的 digest 命名已具备内容寻址基础，补查询逻辑）。

## 5. 数据模型增量

```sql
ALTER TABLE memories ADD COLUMN use_count INTEGER NOT NULL DEFAULT 0;
-- embeddings 表空间维度扩展（新库直接建新 schema；旧库检测无 space 列时迁移）
CREATE TABLE embeddings (
    item_id TEXT NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    embedder TEXT NOT NULL,
    space TEXT NOT NULL DEFAULT 'text',
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL,
    PRIMARY KEY (item_id, embedder, space)
);
```

新 API 汇总（全部向后兼容，v0.1 调用代码零改动）：
- `MemoryConfig.cross_modal: bool = False`、`MemoryConfig.feedback: bool = False`
- `Memory.add_image(..., embed_vision=None)`：传 CLIPStyle 时自动双空间
- `Memory.search(query, ..., video_within=None, feedback=False)`
- `ImageEmbedder` / `ClipStyleEmbedder`（可选依赖 transformers/onnx，import 时才要求）

## 6. 测试与验收标准（完成定义）

1. 全量 pytest ≥ 60 条全绿（含：余弦回归锁、双空间写入/检索、时间窗过滤、
   use_count 反馈、sqlite-vec 缺失时优雅回退、旧库自动迁移）。
2. `examples/demo_v2.py` 端到端：文本+图像双空间+视频时间戳+巩固+遗忘，exit 0。
3. 跨模态检索语义断言：用 Hashing 模拟不了的 CLIP 用「合成正交向量」桩测试
   （text 向量与 vision 向量同 name 时取 max 融合正确）。
4. 旧 v0.1 库文件打开后可读可写（迁移路径测试）。

## 7. 明确不做（v0.2）

- 不内置任何权重下载/模型推理（transformers 路径只提供适配器代码，模型由
  调用方装配）——保持库的安装体积为零增长。
- 不做多 agent 共享/权限隔离。
- 不做图记忆数据库（graphiti 式时序边仍留 P3）。
