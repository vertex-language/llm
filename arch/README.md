# llm/arch

The contract every text-generation family implements, standard or third-party: a Family says whether it claims a checkpoint and loads one onto a device, as a Model the facade (llm) and the generation loop drive without knowing the family.

```vertex
import "llm/arch"
```

## Types

- **`Model`** (protocol): Model is a family's model loaded on a device, with its KV cache: what runs, whatever the family.
- **`Family`** (struct): Family is a model family: its name, whether it claims a checkpoint, and how it loads one.

## Functions

- `func GGUFName(_ hf: string) -> string?`: GGUFName is the GGUF name llama.cpp's converter gives a Hugging Face tensor name, by the table most decoders share (gguf-py's tensor_mapping.py): "model.layers.3.self_attn.q_proj.weight" is "blk.3.attn_q.weight".

Part of the [`llm`](https://github.com/vertex-language/llm) repository.
