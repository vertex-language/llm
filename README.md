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
vsc run bench                 # tokens/second on each device
vsc run test-llama            # greedy tokens against llama.cpp's
```

---

## Packages

| Package | What it is |
| --- | --- |
| **`llm`** | The facade: `llm.Load(ref)` fetches a reference (`model/fetch`), opens it as a checkpoint, and loads it with the first family that claims it. `llm.Families()`, `llm.Register(family)` for one not built in, `llm.Inspect(ref)`, and a greedy `llm.Generate` |
| **`llm/arch`** | The contract families implement: `arch.Model` (`Forward(token, position)` to logits, with a KV cache and a tokenizer) and `arch.Family` (a name, `Claims(checkpoint)`, `Load`). Also llama.cpp's decoder tables (`DecoderAliases`, `GGUFName`), which read a GGUF in Hugging Face's names. Families import this, never `llm` |
| **`llm/llama`** | The Llama family (`LlamaForCausalLM`, GGUF `llama`: Llama 1/2, TinyLlama, SmolLM2 and their fine-tunes), from safetensors or GGUF: f32, f16, bf16, Q4_0, Q8_0, Q4_K and Q6_K weights, grouped-query attention, both RoPE layouts. `llama.WriteGGUF` writes one as llama.cpp's converter would |

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

| Program | Checks |
| --- | --- |
| `test-llama` | "Once upon a time" through three GGUF llamas (f32, Q4_0, Q4_K_M) on the CPU device and Metal: the greedy tokens equal llama.cpp's |
| `test-checkpoint` | the same weights as safetensors and as GGUF give the same logits, and llama.cpp's |
| `test-model` | format choice from a Hub file list, families claiming checkpoints, local references loaded, an unknown architecture refused |

The test models live in `testdata/` and are not committed; see the `model`
repository's README for where they come from, and `vsc run convert` for
`testdata/tiny-llama.gguf`.

## Speed

`vsc run bench` loads `stories15M-q4_0` and generates 100 tokens on each
device; `vsc run profile` splits a token's time; `vsc run ops` times each
operation alone. On an Apple M-series Mac (2026-09-25):

| | Vertex | llama.cpp (same machine) |
| --- | --- | --- |
| Metal | ~320 tokens/s | 615 tokens/s |
| CPU | ~70 tokens/s | 3,433 tokens/s |

---

## License

[MIT](LICENSE)
