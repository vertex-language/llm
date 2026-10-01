# llm/llama

The Llama family: Llama 1 and 2, and the models that share their architecture (LlamaForCausalLM on the Hugging Face Hub, general.architecture "llama" in GGUF: TinyLlama, SmolLM2, most fine-tunes) -- an embedding, blocks of RMSNorm, grouped-query attention with rotary position embeddings.

```vertex
import "llm/llama"
```

## Types

- **`LoadError`** (enum): LoadError is a checkpoint that is not a llama this can run, and says why.
- **`Config`** (struct): Config is a llama's shape.
- **`Block`** (class): Block is one transformer layer.
- **`Model`** (class): Model is a llama on a device, with a KV cache for one sequence.

## Functions

- `func WriteGGUF(_ ck: model.Checkpoint, to path: fs.Path) throws`: WriteGGUF writes a llama checkpoint as a GGUF file llama.cpp loads: its config under llama.*, its SentencePiece vocabulary under tokenizer.ggml.*, and its tensors under GGUF's names, weights in their own dtype (f32, f16 or bf16) and norms as f32.
- `func Argmax(_ xs: [float32]) -> int`: Argmax is the index of the largest value, the first of equals.

Part of the [`llm`](https://github.com/vertex-language/llm) repository.
