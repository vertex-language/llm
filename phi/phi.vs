// Package phi is Microsoft's Phi: Phi-2 so far -- LayerNorm, attention and
// a GELU MLP side by side from one norm, partial rotary embeddings. Phi-3,
// Phi-4 and Phi-MoE join with sliding windows, LongRoPE and experts.
package phi

import (
    "llm/arch"
    "nn/decoder"
)

public let Names = ["PhiForCausalLM", "phi2"]

public let Family = arch.Standard(name: "phi", names: Names, spec: { _ in
    var s = decoder.Spec()
    s.Rope = .neox
    s.Norm = .layer
    s.NormBias = true
    s.Parallel = true
    s.FFNNorm = nil
    s.FFNGated = false
    s.FFNActivation = .GELUTanh
    s.FFNBias = true
    s.OutputBias = true
    return s
})
