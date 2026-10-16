# llm

[![package: vs-package](https://img.shields.io/badge/package-vs--package-f4f4f5?style=flat-square&labelColor=e4e4e7&color=18181b)](https://github.com/vertex-language)
[![devices: metal | cpu](https://img.shields.io/badge/devices-metal%20%7C%20cpu-f4f4f5?style=flat-square&labelColor=e4e4e7&color=18181b)](https://github.com/vertex-language/llm)
[![families: 30 built-in](https://img.shields.io/badge/families-30%20built--in-f4f4f5?style=flat-square&labelColor=e4e4e7&color=18181b)](https://github.com/vertex-language/llm)
[![license: MIT](https://img.shields.io/badge/license-MIT-f4f4f5?style=flat-square&labelColor=e4e4e7&color=18181b)](LICENSE)

Text generation in pure Vertex: language models loaded by name from the model registry (`qwen3`, `llama3.2:3b`, `qwen3:8b-q8_0`) or from local files (`.gguf`, safetensors), running autoregressively on Apple Silicon (Metal) and CPU.

```swift
import "llm"

// Load by name from the model registry (pulled once, pinned, cached)
let m = try await llm.Load("qwen3:0.6b")

// Or load a local quantized GGUF
// let m = try await llm.Load("./stories15M-q4_0.gguf")

let tokens = try await llm.Generate(m, "Once upon a time", tokens: 32)
print(m.Tokenizer.Decode(tokens))
```

---

## Architecture Overview

`llm` provides a driver-style decoupling between model families and generation frontends:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        User Applications & CLI                         │
│             llm.Generate  ·  cmd/generate  ·  cmd/bench                │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                       llm Facade (This Package)                        │
│   llm.Load(ref)  ·  llm.Inspect(ref)  ·  llm.Generate  ·  Registry     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
┌───────────────────────────────────▼────────────────────────────────────┐
│                    llm/arch (The Contract Interface)                   │
│   arch.Model  ·  arch.Batch  ·  arch.Output  ·  arch.Family            │
│   arch.DecoderModel  ·  arch.Standard  ·  arch.DecoderAliases          │
└──────────────┬──────────────────────────────────────────┬──────────────┘
               │                                          │
┌──────────────▼─────────────┐             ┌──────────────▼──────────────┐
│  Dedicated Custom Families │             │  Declarative Spec Families  │
│  llm/llama  ·  llm/qwen    │             │  mistral, gemma, deepseek,  │
│                            │             │  glm, cohere, dbrx, 24 more │
└──────────────┬─────────────┘             └──────────────┬──────────────┘
               │                                          │
┌──────────────▼──────────────────────────────────────────▼──────────────┐
│                 Underlying Neural Network & GPU Layers                 │
│         nn (layers, decoder, moe, cache)  ·  gpu (linalg, neural)      │
└────────────────────────────────────────────────────────────────────────┘
```

A program that needs only a specific family can import it directly (e.g. `import "llm/llama"`) to link only that architecture and eliminate unused code and network dependencies.

---

## Quick Start & CLI Tools

The `cmd/` directory provides pre-built tools for text generation, benchmarking, profiling, and format conversion:

```bash
# Generate text from a local GGUF model
vsc run generate -- testdata/stories15M-q4_0.gguf "Once upon a time" --tokens 64

# Generate text from a model in the registry (vsc run registry -- list llm, in model/)
vsc run generate -- smollm2:135m "The capital of France is"

# Inspect model architecture and tensor configuration without loading weights into GPU
vsc run generate -- qwen3:0.6b --inspect

# Benchmark prompt prefill (pp) and token generation (tg) speed
vsc run bench

# Profile execution latency breakdown across CPU encoding, GPU run, and readback
vsc run profile

# Measure isolated device latency of individual layer operations
vsc run ops

# Convert a Hugging Face snapshot directory into a GGUF file
vsc run convert -- testdata/tiny-llama testdata/tiny-llama.gguf

# Run unit tests and golden verification against llama.cpp
vsc run test-llama       # Greedy token equivalence against llama.cpp
vsc run test-archs       # Validates all families against 50+ llama.cpp architecture models
vsc run test-checkpoint  # Parity between safetensors and GGUF checkpoints
vsc run test-model       # Format resolution and architecture claim checks
```

---

## Packages Summary

| Package | Purpose & Features |
| --- | --- |
| **`llm`** | The public facade: `Load(ref, on:)` resolves registry names (`model/registry`) or local paths, opens the checkpoint, and delegates to the claiming family. `Inspect(ref)` extracts metadata without GPU memory allocation. `Generate(model, prompt, tokens:)` drives the on-device greedy autoregressive loop. `Families()` and `Register(family)` manage the extensible family registry. |
| **`llm/arch`** | The contract implemented by all model families: `arch.Model` (`Decode`, `Context`, `Memory`, `Tokenizer`, `Device`), `arch.Batch` (multi-sequence batch representation with tokens, positions, sequences, and output selection masks), `arch.Output` (`.logits`, `.hidden`), `arch.Family` descriptor, `arch.DecoderModel` (standard `GreedyModel` implementation), and `arch.Standard` spec-family factory. |
| **`llm/llama`** | High-performance implementation of the Llama family (`LlamaForCausalLM`, `llama`), supporting Llama 1/2/3, TinyLlama, SmolLM2, Mixtral, and fine-tunes. Includes `llama.WriteGGUF` for exporting checkpoints to GGUF. |
| **`llm/qwen`** | Full family support for Qwen 1, Qwen 2, Qwen 2.5, Qwen 3, Qwen2-MoE, and Qwen3-MoE across dense and mixture of experts configurations. |
| **`llm/<family>`** | 28 additional built-in families defined concisely via `decoder.Spec`: `arcee`, `bailing` (Ling 1 & 2), `codeshell`, `cohere` (Command R / R+), `dbrx`, `deepseek` (MoE with router bias), `dots`, `ernie` (4.5 & 4.5 MoE), `exaone` (3, 4, K-EXAONE), `gemma` (1, 2, 3), `glm` (ChatGLM, GLM-4, GLM-4.5, GLM-DSA), `granite` (dense, MoE, SWA, hybrid), `hunyuan` (dense, MoE, HY V3/V4, VL), `internlm` (1 & 2), `maincoder`, `minimax` (M2), `mistral` (1 & 3), `nemotron` (H, MoE), `olmo` (1, 2, OLMoE), `orion`, `pangu`, `phi` (2, 3, MoE), `plamo` (1, 2, 3), `seed`, `smollm` (SmolLM3), `stablelm`, `starcoder` (1 & 2), and `xverse`. |

---

## Detailed API Reference

### 1. The `llm` Facade

```vertex
import "llm"
```

#### Functions
- `func Load(_ ref: string, on d: gpu.Device? = nil) async throws -> any Model`:
  Opens a model reference:
  - A registry name: `"qwen3"`, `"qwen3:8b"`, `"qwen3:8b-q8_0"` ([`model/registry`](https://github.com/vertex-language/model/tree/main/registry)). Pulled into the Hugging Face cache (`~/.cache/huggingface/hub`) once, at its pinned commit.
  - Local path: `"./models/model.gguf"` or `"./models/snapshot-dir"`
  Hub references (`hf.co/...`) are refused: the registry is the front door. The checkpoint loads onto `gpu.Default()` (or `d` if specified).
- `func Inspect(_ ref: string) async throws -> Info`:
  Reads configuration without allocating GPU buffers or downloading entire weight tensors. Returns `Info`:
  - `Architecture`: Name declared in checkpoint (e.g. `"LlamaForCausalLM"`).
  - `Family`: Built-in family that claims the checkpoint.
  - `Format`: `.gguf` or `.safetensors`.
  - `Tensors`: Total count of tensors.
  - `Bytes`: Total model file size in bytes.
  - `Config`: Key-value configuration pairs (layer count, head count, dimensions, etc.).
- `func Generate(_ m: any Model, _ prompt: string, tokens n: int) async throws -> [int]`:
  Greedily generates up to `n` tokens from `prompt`.
  - When `m` conforms to `arch.GreedyModel` (such as `arch.DecoderModel`), execution stays entirely on the device: the next token is computed by `neural.Argmax` directly on the GPU and immediately fed into the next step's `Embedding.LookupRow` without copying logits to host memory.
  - Automatically terminates early upon producing `m.Tokenizer.Vocab.Eos`.
- `func Register(_ f: Family)`:
  Registers a third-party or experimental family ahead of the built-in family table.

---

### 2. Architecture Contract (`llm/arch`)

```vertex
import "llm/arch"
```

#### `arch.Model` Protocol
The protocol implemented by all language models:
```vertex
public protocol Model: AnyObject {
    var Device: gpu.Device { get }
    var Memory: cache.Memory { get }
    var Context: int { get }
    var Tokenizer: tokenizer.Tokenizer { get }
    func Decode(_ b: Batch, _ out: Output) async throws -> gpu.Buffer<float32>
}
```

#### `arch.Batch` Struct
Encapsulates batched inference across arbitrary sequences:
- `Tokens: [int32]`: Token IDs for each row in the batch.
- `Embeddings: Buffer<float32>?`: Optional explicit embedding inputs (for multimodal adapters or image tokens) in place of `Tokens`.
- `Positions: [int32]`: Positional coordinate for rotary position embeddings.
- `Sequences: [int32]`: Target sequence index within `Memory` (supporting up to 256 concurrent sequences).
- `Outputs: [bool]`: Boolean mask specifying which rows should output logits or hidden states.
- `Batch.Prompt(tokens, from: 0, sequence: 0, all: false)`: Helper constructing a standard prompt batch where only the final row produces logits (or all rows if `all == true`).

#### `arch.DecoderModel`
Standard implementation of `arch.GreedyModel`:
- Wraps `decoder.Network`, `tokenizer.Tokenizer`, and `cache.KV`.
- Implements `Generate(prompt, tokens:)` with on-device greedy selection.

---

### 3. Supported Model Families (30 Built-in)

Every model family is verified against llama.cpp reference implementations.

| Family | Architecture Aliases Claimed | Key Architectural Features |
| --- | --- | --- |
| **`llama`** | `LlamaForCausalLM`, `llama` | Standard SwiGLU, RMSNorm, GQA, Llama 3 `rope_freqs` scaling. Supports export via `llama.WriteGGUF`. |
| **`qwen`** | `QWenLMHeadModel`, `qwen`, `Qwen2ForCausalLM`, `qwen2`, `Qwen3ForCausalLM`, `qwen3`, `Qwen2MoeForCausalLM`, `qwen2moe`, `Qwen3MoeForCausalLM`, `qwen3moe` | QKV biases (Qwen 2/2.5), QK-norm (Qwen 3), shared gated experts (Qwen2-MoE, Qwen3-MoE). |
| **`mistral`** | `MistralForCausalLM`, `mistral`, `Mistral3ForConditionalGeneration`, `Ministral3ForCausalLM`, `mistral3` | Sliding window attention, GQA, high RoPE theta. |
| **`granite`** | `GraniteForCausalLM`, `granite`, `GraniteMoeForCausalLM`, `GraniteMoeSharedForCausalLM`, `granitemoe` | Residual multipliers, logit scale division, shared experts in MoE. |
| **`olmo`** | `OlmoForCausalLM`, `olmo`, `Olmo2ForCausalLM`, `olmo2`, `OlmoeForCausalLM`, `olmoe` | OLMo 1 (no norm weights), OLMo 2 (post-attention & post-FFN norms), OLMoE. |
| **`exaone`** | `ExaoneForCausalLM`, `exaone`, `Exaone4ForCausalLM`, `exaone4`, `ExaoneMoEForCausalLM`, `exaone-moe` | SWA with dense first layer, `RopeSlidingOnly` (EXAONE 4 turns only sliding layers). |
| **`internlm`** | `InternLMForCausalLM`, `internlm`, `InternLM2ForCausalLM`, `internlm2` | QKV bias variations, custom scaling factors. |
| **`smollm`** | `SmolLM3ForCausalLM`, `smollm3` | `NoRopeEvery: 4` (every 4th layer omits rotary embeddings). |
| **`seed`** | `SeedOssForCausalLM`, `seed_oss` | Standard decoder with LayerNorm options. |
| **`arcee`** | `ArceeForCausalLM`, `arcee` | Specialized merged transformer architecture. |
| **`pangu`** | `PanguEmbeddedForCausalLM`, `pangu-embedded` | Lightweight embedded causal decoder. |
| **`orion`** | `OrionForCausalLM`, `orion` | Scaled GQA transformer. |
| **`xverse`** | `XverseForCausalLM`, `xverse` | Custom vocab and scaling. |
| **`maincoder`**| `MaincoderForCausalLM`, `maincoder` | `NormAfterRope`: head normalization evaluated after rotation. |
| **`gemma`** | `GemmaForCausalLM`, `gemma`, `Gemma2ForCausalLM`, `gemma2`, `Gemma3ForCausalLM`, `gemma3` | Embedding scaled by $\sqrt{d}$, post-norm branches, head norms. |
| **`phi`** | `PhiForCausalLM`, `phi2`, `Phi3ForCausalLM`, `phi3`, `PhiMoeForCausalLM`, `phimoe` | Phi-2 parallel attention & MLP layers; Phi-3 fused GateUp weights. |
| **`stablelm`** | `StableLmForCausalLM`, `StableLMEpochForCausalLM`, `stablelm` | Parallel layers, QK-norm with per-head scaling. |
| **`starcoder`**| `Starcoder2ForCausalLM`, `starcoder2`, `starcoder` | Code generation architecture with sliding window options. |
| **`cohere`** | `CohereForCausalLM`, `cohere`, `Cohere2ForCausalLM`, `cohere2`, `Cohere2MoeForCausalLM`, `cohere2moe`, `CommandRForCausalLM`, `command-r` | Parallel decoder layers, logit scale multipliers, QK head norms. |
| **`glm`** | `ChatGLMModel`, `chatglm`, `Glm4ForCausalLM`, `glm4`, `Glm4MoeForCausalLM`, `glm4moe`, `GLM-DSA`, `glm-dsa` | Fused GateUp projections, sigmoid MoE gating. |
| **`nemotron`** | `NemotronForCausalLM`, `nemotron`, `NemotronHForCausalLM`, `nemotron_h`, `NemotronHMoeForCausalLM`, `nemotron_h_moe` | High parameter scale decoders and hybrid MoE layers. |
| **`plamo`** | `PlamoForCausalLM`, `plamo`, `Plamo2ForCausalLM`, `plamo2`, `Plamo3ForCausalLM`, `plamo3` | Dense transformers with custom rotary scalings. |
| **`hunyuan`** | `HunYuanDenseV1ForCausalLM`, `hunyuan-dense`, `HunYuanVLForConditionalGeneration`, `hunyuan_vl`, `HY_V3`, `hy_v3`, `HY_V4`, `hy_v4` | Dense, MoE, and multimodal causal architectures. |
| **`codeshell`**| `CodeShellForCausalLM`, `codeshell` | Pre-norm code generation network. |
| **`ernie`** | `Ernie4_5_ForCausalLM`, `Ernie4_5ForCausalLM`, `ernie4_5`, `Ernie4_5_MoeForCausalLM`, `ernie4_5-moe` | Dense and mixture of experts configurations. |
| **`deepseek`** | `DeepseekForCausalLM`, `deepseek`, `Deepseek2ForCausalLM`, `deepseek2`, `Deepseek32ForCausalLM`, `deepseek32`, `Deepseek4ForCausalLM`, `deepseek4` | Router selection bias correction, sigmoid and $\sqrt{\text{softplus}}$ gating. |
| **`dots`** | `Dots1ForCausalLM`, `dots1`, `Dots3NoteForCausalLM`, `dots3note` | File-driven gating and normalization options. |
| **`dbrx`** | `DbrxForCausalLM`, `dbrx` | MegaBlocks MoE with QKV activation clamping (`Clamp = 6.0`). |
| **`minimax`** | `MiniMax01ForCausalLM`, `minimax-01`, `MiniMaxM2ForCausalLM`, `minimax-m2`, `MiniMaxM3ForCausalLM`, `minimax-m3` | Deep sparse MoE architecture with specialized routing. |
| **`bailing`** | `BailingMoeForCausalLM`, `bailingmoe`, `BailingMoeV2ForCausalLM`, `bailingmoe2`, `BailingMoe3ForCausalLM`, `bailingmoe3` | Ling 1 & Ling 2 sparse MoE decoders. |

---

## Testing & Verification

Verification is conducted against reference fixtures generated by llama.cpp:

- `cmd/test-archs`: Evaluates each architecture fixture in `testdata/archs` against llama.cpp logits for identical 48-token sequences. Tests:
  1. Full batch prompt (all rows checked).
  2. Split prompt (10 tokens batched, then 6 tokens single-step).
  3. Two sequences concurrently evaluated in one memory.
  Validation requires normalized mean squared error (NMSE) within $\le 10^{-4}$.
- `cmd/test-llama`: Verifies that greedy token outputs match llama.cpp character-for-character across Float32, Q4_0, and Q4_K_M GGUF models.
- `cmd/test-checkpoint`: Validates that identical weights saved as safetensors and GGUF yield bit-equivalent logits.
- `cmd/test-model`: Validates format detection, checkpoint claiming, and rejection of unrecognized architectures.

---

## Performance & Speed

On an Apple M1 (MacBook Air, 7-core GPU), greedy generation with `llm.Generate` runs without host synchronization stalls:

| Model / Quantization | Metal Generation (`tg`) | llama.cpp (Same Machine) |
| --- | --- | --- |
| **stories15M Q4_0** | **~1,200 tokens/s** | ~585 tokens/s |
| **stories110M Q4_K_M** | **~330 tokens/s** | ~320 tokens/s |

### Why Vertex Generation is Faster
Standard inference engines synchronize each step with the host CPU to retrieve the next token ID before dispatching the next step. In Vertex, `llm.Generate` and `arch.DecoderModel` keep the entire loop on the device:
1. `neural.Argmax` finds the maximum logit row index in GPU registers.
2. `Embedding.LookupRow` reads that device index directly to decode the next token's embedding into shared memory.
3. The host thread only interacts once to read back the final sequence tokens upon completion or EOS.

---

## License

[MIT](LICENSE)
