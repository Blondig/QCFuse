# vLLM 0.9.2：ProphetKV + 一阶 attention 影响评分

独立的研究原型，复用 vLLM 已加载的 Llama/Qwen3 权重和分页 KV 缓存，
实现块 KV 复用、query 探测和文档 token 选择性重算。生成和采样继续使用原生
vLLM。此入口不依赖本仓库的 SGLang overlay，也不需要 LMCache。

## 方案与边界

1. **离线块缓存**：prefix 和每个 document 独立前向，保存每层 RoPE 前的 K 和 V。
   Qwen3 的 K 已经过 K norm。复用时按拼接后的全局位置重新应用 RoPE。
2. **Prophet 风格探测**：仅让完整 query suffix 通过所有层，各层读取已缓存文档
   KV 和当前 suffix 的因果 KV。按 query token、query head、层求平均 attention
   得到文档相关性 `A`。不使用最大 query attention 替代均值。
3. **测量浅层漂移**：完整计算前 `p` 层，在第 `p` 层投影所有 token 的 Q/K/V，
   对比独立缓存与当前文档 KV。层号从 0 开始，默认 `p=2`。这得到该层的真实
   上下文漂移；其对深层重要性的预测仍是启发式，需要消融验证。
4. **一阶影响评分**：固定当前测量层的 suffix Q，以独立缓存 prefix/document
   KV 和当前 suffix KV 组成基线，使用完整因果 softmax 分母和基线输出：

   ```text
   delta_o_i = a_i * [delta_v_i + (q dot delta_k_i)/sqrt(d) * (v_i - o)]
   S_i = mean_query sum_head ||delta_o_i||²
   F_i = (1 - weight) * rank(A_i) + weight * rank(S_i)
   ```

   两个 rank 都是平均处理 ties 的百分位排名；常数分数返回 0。
   该公式描述固定 Q 时单 token 修复的局部一阶变化，不等同于最终答案收益，
   也不把多个 token 的独立影响当作联合修复误差的精确分解。
5. **固定集合重算**：在所有文档 token 中选取 `floor(ratio * Ndoc)` 个；
   `ratio>0` 且存在文档时至少 1 个。prefix 和 suffix 全部保留，不占文档预算。
   第 `p` 层使用刚算出的完整 KV，仅对保留 token 做 attention/MLP；之后各层
   仅计算保留 token 的投影/attention/MLP，将其 KV 写入复用的完整块缓存。
6. **原生解码**：每层完整 KV 写回 vLLM 已分配的真实 slot，最后一个 suffix
   token 的 hidden state 交给原生 logits/sampler。后续 decode 恢复原生 forward。

稀疏 attention 使用 PyTorch SDPA 和 token 的真实绝对位置构造因果遮罩。
非连续 query 不能直接套用“连续末尾 query”的 causal mask。当前代码没有移植
QCFuse 的压缩视图、流水线或 Triton 稀疏内核；这是 ProphetKV 思路与数学评分的
融合原型，不能视作论文原版 ProphetKV 的逐项复现。

## 模式和预算

| `--method` | 排序分数 | 公共漂移去除 |
| --- | --- | --- |
| `prophet` | 全层平均 query attention | 无 |
| `prophet_fo` | Prophet + 原始一阶影响，默认模式 | 无 |
| `prophet_fo_residual` | Prophet + 残差一阶影响 | 完全去除首方向 |
| `prophet_fo_mixed` | Prophet + 原始/残差影响分数的均值 | 两种分数各占 50% |

残差模式把各文档 token 的 `[vec(delta_K), vec(delta_V)]` 拼接，用最多 2048
个均匀采样 token、双初值幂迭代近似**未中心化**矩阵的首右奇异方向 `w`，再用
`x_perp = x - (x dot w) * w` 计算残差影响。`residual_weight` 混合原始和残差
影响的平方范数分数；mixed 是 `0.5 * S_raw + 0.5 * S_residual`，再与 Prophet
相关性排名融合。全局漂移也可能携带重要信息，因此默认不开启去除。

`--influence-weight` 控制 FO 在排名融合中的权重，默认 0.5。
`prophet --probe-layer 0` 是从第 0 层起稀疏计算的 attention-only 基线。
若要单独考察评分的收益，应给 `prophet` 和 FO 模式使用相同的 `probe-layer`。
FO 模式要求 `1 <= p < 层数`；第 0 层没有来自此前层的跨块上下文漂移。

`ratio=0` 跳过 query probe/FO 评分，但仍支付指定浅层、prefix、suffix 的成本。
`ratio=1` 跳过评分，执行完整自定义 prefill；它是与原生 full prefill 比较数值
正确性的控制组。100% 预算不要求缓存命中。

对于 `L` 层、`b` 个选中文档 token，FO 模式实际文档工作量为：

```text
attention/MLP token-layers = p * Ndoc + (L - p) * b
KV projection token-layers = (p + 1) * Ndoc + (L - p - 1) * b
```

所以 `ratio=0.2` 不代表整个 prefill 只剩 20% FLOPs。query probe、评分、缓存传输、
全量 KV 写回、必选 prefix/suffix 都有额外开销。

## 环境和运行

从仓库根目录运行，使用已有的 vLLM **0.9.2** 环境；需要新环境时安装：

```bash
python -m pip install -r vllm_repair/requirements.txt
python -m vllm_repair.run --help
```

vLLM 0.9.2 的官方 CUDA 依赖固定 torch 2.7.0，应沿用其依赖组合，避免单独升级
PyTorch。[官方依赖](https://github.com/vllm-project/vllm/blob/v0.9.2/requirements/cuda.txt)

输入 JSON 包含显式 prefix、独立文档块、query/suffix，例如：

```json
{
  "prefix": "Answer the question using the documents.\n\n",
  "documents": [
    "Document 1:\nThe meeting is on Tuesday.\n\n",
    "Document 2:\nThe meeting room is B204.\n\n"
  ],
  "query": "Question: Which room is the meeting in?\nAnswer:"
}
```

将上述内容保存为 `input.json`，使用本地模型路径运行：

```bash
python -m vllm_repair.run \
  --model /path/to/Qwen3-8B \
  --input input.json \
  --method prophet_fo \
  --ratio 0.2 --probe-layer 2 \
  --compare-full --max-tokens 32
```

每段只 tokenize 一次，再拼接同一份 token IDs 用于完整和修复两条路径，避免
文本拼接后的 BPE 边界变化。输入应自行包含需要的聊天模板、BOS、分隔符和
assistant generation prompt；runner 不自动为每个文档添加特殊 token。

先用 `--ratio 1 --compare-full` 检查本机模型/后端，再比较低预算及残差模式。
原生 FlashAttention 与自定义 SDPA 的浮点误差可能使接近并列的 greedy token
不同，单条文本相同也不意味着所有 KV 完全相同。质量评估应使用数据集的答案
指标，不能仅凭输出是否与 full 路径一致。

## 约束和计时

首版限定 vLLM V1、单请求、TP=PP=1、dense Llama/Qwen3、eager、关闭编译、
prefix caching 和 chunked prefill。原生分页缓存使用 FlashAttention 后端。
不支持量化模型/量化 KV、LoRA、sliding-window attention、多模态或 prompt logprobs。
不符合约束时应直接报错，避免把未修复的请求静默当作成功结果。

离线块缓存默认放在 CPU。`prepare_chunks` 保留本次所需的块，并驱逐其余块，
不会随处理样本数无限累积；它不是持久化 SSD 缓存。在线各层搬运计入运行时间。
需要调用者按同一组 prefix/document token IDs 准备并 arm 请求。

runner 将首次输出 token 的 engine-step 墙钟时间记为 TTFT，query probe 在该区间
内。离线块准备单独报告。runtime 的 `query_probe_s`、`repair_s` 是设备同步的诊断
耗时；`fo_score_s` 已包含在 `repair_s` 中，不能重复相加。它们不等于引擎总 TTFT。
native full 与修复路径应同时报告 TTFT、生成总耗时和答案质量，避免用文档预算
或算子减少量代替真实加速比。

SDPA 是否选用高效 CUDA kernel 取决于 GPU、dtype 和 mask；退回 math backend 时
GQA 可能展开 KV。query 分块只限制单次 attention 的 query 规模。本原型优先验证
选择算法和 KV 语义，尚未对传输重叠、异步流水线或专用稀疏 kernel 做性能优化。

## 验证

不依赖 vLLM 的 CPU 数值及 runtime 测试：

```bash
python -B -m unittest test.test_vllm_scoring test.test_vllm_repair -v
```

覆盖完整因果分母、GQA、非连续 query、有限差分、一阶评分、残差边界，以及
小型 Llama/Qwen3 接口模型的 100% 预算与独立 dense 前向等价性、逐层完整 KV
写回、实际稀疏投影/MLP 工作量、固定选择集、缓存复用和异常清理。

开发机无可用 CUDA/vLLM；CPU 测试使用 torch 2.11。真实 vLLM 0.9.2/torch 2.7
GPU 联调、目标模型正确性和 TTFT/质量曲线需要在目标环境运行，尚无实测加速结论。

接口依据：[vLLM worker extension](https://github.com/vllm-project/vllm/blob/v0.9.2/vllm/worker/worker_base.py)、
[V1 FlashAttention KV 写回](https://github.com/vllm-project/vllm/blob/v0.9.2/vllm/v1/attention/backends/flash_attn.py)、
[Qwen3 decoder](https://github.com/vllm-project/vllm/blob/v0.9.2/vllm/model_executor/models/qwen3.py)。
方法参考：[ProphetKV](https://arxiv.org/html/2602.02579v3)。
