// Package smollm is SmolLM3: Llama's layers, every fourth without rotary
// embeddings. SmolLM2 loads as llama.
package smollm

import (
    "llm/arch"
    "nn/decoder"
)

public let Names = ["SmolLM3ForCausalLM", "smollm3"]

public let Family = arch.Standard(name: "smollm", names: Names, spec: { _ in
    var s = decoder.Spec()
    s.NoRopeEvery = 4
    s.FFNBias = true
    s.AttentionScale = true
    return s
})
