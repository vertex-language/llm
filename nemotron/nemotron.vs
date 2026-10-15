// Package nemotron is NVIDIA's Nemotron-4 (nemotron) so far: LayerNorm
// with biases, partial rotary embeddings, a squared-ReLU MLP with biases.
// Nemotron-H and the Llama-Nemotron (Deci) join with their mechanisms.
package nemotron

import (
    "llm/arch"
    "nn/decoder"
)

public let Names = ["NemotronForCausalLM", "nemotron"]

public let Family = arch.Standard(name: "nemotron", names: Names, spec: { _ in
    var s = decoder.Spec()
    s.Rope = .neox
    s.Norm = .layer
    s.NormBias = true
    s.FFNGated = false
    s.FFNActivation = .ReLUSquared
    s.FFNBias = true
    return s
})
