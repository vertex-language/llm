# llm

[![package: vs-package](https://img.shields.io/badge/package-vs--package-f4f4f5?style=flat-square&labelColor=e4e4e7&color=18181b)](https://github.com/vertex-language)
[![families: llama](https://img.shields.io/badge/families-llama-f4f4f5?style=flat-square&labelColor=e4e4e7&color=18181b)](https://github.com/vertex-language/llm)

Text generation: language models loaded by reference, from a local file or the Hugging Face Hub, onto Metal or the CPU, in pure Vertex.

```swift
import "llm"

let m = try await llm.Load("hf.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0")
let tokens = try await llm.Generate(m, "Once upon a time", tokens: 32)
print(m.Tokenizer.Decode(tokens))
```

---

## Quick Start

```bash
vsc run generate -- testdata/stories15M-q4_0.gguf "Once upon a time" --tokens 64
vsc run generate -- hf.co/HuggingFaceTB/SmolLM2-135M "The capital of France is"
vsc run bench                 # prompt and generation tokens/second on each device
vsc run test-llama            # greedy tokens against llama.cpp's
vsc run test-archs            # every family against llama.cpp's random-weight model of each architecture
```

---

## Packages

| Package | What it is |
| --- | --- |
| **`llm`** | The facade: `llm.Load(ref)` fetches a reference (`model/fetch`), opens it as a checkpoint, and loads it with the first family that claims it. `llm.Families()`, `llm.Register(family)` for one not built in, `llm.Inspect(ref)`, and a greedy `llm.Generate` |
| **`llm/arch`** | The contract families implement: `arch.Model` -- `Decode(batch, output)`, a `Batch` of tokens (or embeddings) at positions of sequences with the rows whose logits or hidden states are wanted, over the model's `Memory` (`nn/cache`) -- and `arch.Family` (a name, `Claims(checkpoint)`, `Load`). `arch.Forward(m, token, position)` is a one-row Decode. Families import this, never `llm` |
| **`llm/llama`** | The Llama family for generation (`LlamaForCausalLM`, GGUF `llama`: Llama 1 to 3, TinyLlama, SmolLM2, Mixtral and their fine-tunes) over the `nn/llama` network: a prompt is one batch, then greedy decoding runs on the device. `llama.WriteGGUF` writes one as llama.cpp's converter would |
| **`llm/qwen`** | Qwen 1, 2, 2.5 and 3, Qwen2-MoE and Qwen3-MoE, over the `nn/qwen` network |
| **`llm/<family>`** | A family a `decoder.Spec` describes, one package each: `arcee`, `bailing` (Ling 1 and 2), `codeshell`, `cohere`, `dbrx`, `deepseek` (DeepSeek-MoE), `dots`, `ernie` (4.5 and 4.5 MoE), `exaone` (3, 4, K-EXAONE), `gemma`, `glm` (ChatGLM, GLM-4, GLM-4.5), `granite` (and Granite MoE), `hunyuan` (dense, MoE, HY V3), `internlm`, `maincoder`, `minimax` (M2), `mistral` (Mistral 3), `nemotron`, `olmo` (OLMo, OLMo 2, OLMoE), `orion`, `pangu`, `phi`, `plamo`, `seed`, `smollm`, `stablelm`, `starcoder`, `xverse`. `vsc run test-archs` says which of llama.cpp's architectures match: 50 of 127 |

A program that wants one family imports it and links nothing else:

```swift
import "llm/llama"
let m = try await llama.Model.Load(fs.Path("stories15M-q4_0.gguf"), on: gpu.Default())
```

---

## Testing

llama.cpp is the oracle, built from source. `testdata/oracle/llama_run.cpp`
runs a prompt through libllama a token at a time and prints each step's top
logits and the greedy continuation (`cmd/test-llama/golden`).
`testdata/oracle/gen-archs.sh` runs llama.cpp's `test-llama-archs` to write
a random-weight model of every architecture it can save (127 of them, every
optional tensor its graph reads set), and `arch_logits.cpp` writes
llama.cpp's logits for 48 tokens through each: `testdata/archs`.

| Program | Checks |
| --- | --- |
| `test-archs` | each `testdata/archs` model through the family that claims it, on the CPU device and Metal, as one batch, as a batch and single tokens, and as two sequences in one memory: normalized squared error within llama.cpp's own bound, 1e-4. It lists the architectures no family claims yet |
| `test-llama` | "Once upon a time" through three GGUF llamas (f32, Q4_0, Q4_K_M) on the CPU device and Metal: the greedy tokens equal llama.cpp's |
| `test-checkpoint` | the same weights as safetensors and as GGUF give the same logits, and llama.cpp's |
| `test-model` | format choice from a Hub file list, families claiming checkpoints, local references loaded, an unknown architecture refused |

The test models live in `testdata/` and are not committed; see the `model`
repository's README for where they come from, and `vsc run convert` for
`testdata/tiny-llama.gguf`.

## Speed

`vsc run bench` loads `stories15M-q4_0` (or `MODEL=`), reads a prompt of
`PROMPT` tokens as one batch, and generates 100 tokens on each device; `vsc run profile` splits a token's time; `vsc run ops`
times each operation alone; `VERTEX_GPU_PROFILE=3` before any of them says
each kernel's device time as it runs in the model. On an Apple M1 (MacBook
Air, 7-core GPU), 2026-10-14, llama.cpp's `llama-simple` run alternately:

| Metal, tokens/s | Vertex | llama.cpp (same machine) |
| --- | --- | --- |
| stories15M Q4_0 | ~1,200 | ~585 |
| stories110M Q4_K_M | ~330 | ~320 |

`Generate` runs the greedy loop on the device: each token's greatest logit
is chosen there and read by the next step's embedding lookup, so the host
never waits to learn a token (llama.cpp samples each token on the host).
Driving `Forward` token by token instead, reading each token back, is ~860
and ~300 tokens/s. The CPU device is ~65 tokens/s on stories15M, against
llama.cpp's 3,433 (as measured 2026-09-25).

Reading a prompt is where llama.cpp is still far ahead (`llama-bench`'s
pp): ~3,700 tokens/s on stories15M against its ~35,000, and ~420 on
stories110M against ~6,900. A prompt is one batch through `linalg.Gemm`
and tiled attention, but llama.cpp's matrix products run on Apple's
`simdgroup_matrix` units, which Vertex kernels cannot reach until the IR
has wave-scope matrix multiply-accumulate.

---

## License

[MIT](LICENSE)
