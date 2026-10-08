# 轻量多模态记忆库:图文 embedding 与向量索引选型调研纪要(2026-10-08)

> 目标场景:依赖仅 numpy+pillow 的 Python 多模态记忆库,重依赖(torch 等)保持可选;部署平台 GB10 aarch64(Linux arm64)。

## 1. 图文对齐 embedding 模型横向对比

| 模型 | 许可证 | 维度 | 大小 | 多语言 | 商用 | 来源 |
|---|---|---|---|---|---|---|
| jina-clip-v2 | **CC-BY-NC-4.0(禁商用)** | 1024(可 Matryoshka 截至最低 64) | ~0.9B 参数,safetensors ~1.73GB(f16) | 89 语言 | 需 API/授权 | https://hf-mirror.com/jinaai/jina-clip-v2 |
| google/siglip2-base-patch16-256 | **Apache-2.0** | 768(base 系列 hidden/projection) | ~375M,f32 单文件 ~1.5GB | 英文为主(WebLI) | 可 | https://hf-mirror.com/google/siglip2-base-patch16-256 |
| OFA-Sys/chinese-clip-vit-base-patch16 | **MIT**(GitHub 仓库确认;HF 卡未标注) | 512(config.projection_dim 实测) | 仓库总存储 ~3.0GB(多格式);bin 约 380M 级 | 中文 | 可 | https://api.github.com/repos/OFA-Sys/Chinese-CLIP |
| openai/clip-vit-base-patch32 | **MIT**(代码仓库);模型卡自述"仅供研究用途",无 license 标签 | 512 | ~151M 参数;usedStorage ~3.6GB(多框架) | 英文 | 代码 MIT;权重卡研究倾向,商用灰色 | https://api.github.com/repos/openai/CLIP |

- jina-clip-v2 细节:文本塔 Jina-XLM-RoBERTa(561M)+ 视觉塔 EVA02-L14(304M),512×512 图输入,8192 token 文本;官方提供 ONNX 版本。CC-BY-NC 是硬伤,**自托管商用违约**。来源:hf-mirror 模型卡。
- SigLIP 2(2025.2 发布,Google):在 sigmoid 损失上统一了解码/全局-局部/掩码目标,新增 NaFlex 变体(动态分辨率,naflex 版在 so400m 系列);base 级 Apache-2.0,是**可商用的最佳默认**。来源:模型卡。
- 2025-2026 轻量新模型补充:
  - SmolVLM2(HuggingFaceTB,Apache-2.0):是 VLM(生成式)而非纯 embedding 模型,做检索要自己取隐藏态,不推荐当向量库用。来源:https://hf-mirror.com/api/models/HuggingFaceTB/SmolVLM2-2.2B-Instruct (license 标签实测)。
  - PaliGemma2(google,自定义 Gemma 许可,可商用但带附加条款):同样是生成式 VLM。来源:HF API license 标签。
  - jina-clip 系列无 v3(2026-10 检索确认 v2 仍是最新,创建于 2024-10-08)。来源:https://hf-mirror.com/api/models?search=jina-clip&author=jinaai
- CPU/aarch64 可行性:上述 4 个均为 Transformer 小模型(<1.5GB 内存),GB10 arm64 可跑;jina-clip-v2 0.9B 在纯 CPU 上编码会明显慢(秒级/张),base 级(150-400M)更合适。ONNX 路径见下节。

## 2. 运行时依赖:transformers+torch vs onnxruntime(CPU)

- `torch` CPU wheel aarch64 约 200MB+,加上 transformers 全家桶安装体积 ~2GB 级;onnxruntime CPU 轮子仅 **14-22MB**(实测 1.30.0 有 `manylinux_2_28_aarch64.whl` 21.3MB,另有 macOS/win_arm64 轮),依赖仅 flatbuffers/numpy/packaging/protobuf(实测 requires_dist)。来源:https://pypi.org/pypi/onnxruntime/json
- **结论:onnxruntime 明显更轻**。jina-clip-v2 官方带 ONNX;Chinese-CLIP 官方仓库提供 ONNX 导出脚本与预训练模型(README 2023.1.15 起支持,来源:https://github.com/OFA-Sys/Chinese-CLIP );SigLIP2 需自己用 optimum 导出(一次性成本)。建议架构:核心库只依赖 onnxruntime(可选),torch+transformers 做为可选 extras。

## 3. 向量索引最小方案对比(N<100k)

| 方案 | 依赖/大小 | arm64 轮子 | 适用 | 来源 |
|---|---|---|---|---|
| numpy 暴力余弦 | 已有依赖,0 增量 | 不需要 | N<100k、dim≤1024 时全量点积 <10ms 级(实测数量级),**首选** | — |
| hnswlib | 源码分发 tar.gz(需 C++ 编译),仅依赖 numpy | 无官方 aarch64 轮,需本机 gcc;header-only C++ | N>100k 或要求亚线性查询时再引入;Apache-2.0 | https://pypi.org/pypi/hnswlib/json ; https://api.github.com/repos/nmslib/hnswlib |
| sqlite-vec | 纯 C 扩展,pip 轮 **0.16-0.17MB**,零依赖 | 有 manylinux2014_aarch64 轮 | 想把向量+元数据+embedding 缓存统一进一个 sqlite 文件时最优;Apache-2.0 | https://pypi.org/pypi/sqlite-vec/json ; https://api.github.com/repos/asg017/sqlite-vec |
| faiss-cpu | MIT;轮子 9-16MB 但总发布 ~190MB | 有 `musllinux_1_2_aarch64.whl`(1.15.1) | N>1M 或需要 GPU/IVF 索引时才值得;小库杀鸡用牛刀 | https://pypi.org/project/faiss-cpu/ |

- 推荐分层:默认 numpy 暴力(纯 Python 可回退);需要持久化向量+元数据 → sqlite-vec;规模或延迟不达标 → hnswlib(需编译)或 faiss-cpu(轮子全)。

## 4. embedding 工程实践的坑

1. **dtype 用 float32**:模型输出即 f32;转 f64 只会加倍内存与 sqlite 体积,精度无收益。numpy 默认 f64,`arr.astype(np.float32)` 显式降。
2. **L2 归一化后点积=余弦**:归一化一次后 `vec @ mat.T` 即余弦相似度,免去每对重算范数;hnswlib 也建议存归一化向量用 `space='ip'`(内积空间等价余弦)。来源:hnswlib README https://github.com/nmslib/hnswlib
3. **sqlite 缓存 schema 常见做法**:`embeddings(model TEXT, id TEXT, dim INT, vec BLOB)`;向量用 `np.float32(arr).tobytes()` 存 BLOB,读回 `np.frombuffer(blob, np.float32)`;以 (model, id) 唯一键——**换模型必须重建/分表**,不同模型向量不可比。sqlite-vec 官方即用 vec0 虚拟表存 float32 blob。来源:https://github.com/asg017/sqlite-vec
4. **截断维度(Matryoshka)省内存**:jina-clip-v2 支持截到 64-1024;若库小,截到 256 可把向量体积降 4 倍,精度损失可接受(需自测 recall)。
5. **批处理**:图像编码按 batch 过模型(ONNX 动态 batch),逐张循环在 CPU 上慢 5-10 倍。

## 5. 最终建议(一句话)

默认 `SigLIP2-base(Apache-2.0,ONNX 导出)+ numpy 暴力余弦 + sqlite(f32 blob)缓存`;中文场景换 `Chinese-CLIP base(MIT,官方 ONNX)`;需要多语言+长文本且不商用才上 jina-clip-v2;>100k 条再引入 hnswlib/faiss-cpu。
