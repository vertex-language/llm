// Package orion is OrionStar's Orion: LayerNorm with biases, NeoX rotary
// embeddings.
package orion

import (
    "llm/arch"
    "nn/decoder"
)

public let Names = ["OrionForCausalLM", "orion"]

public let Family = arch.Standard(name: "orion", names: Names, spec: { _ in
    var s = decoder.Spec()
    s.Rope = .neox
    s.Norm = .layer
    s.NormBias = true
    s.AttnOutBias = false
    return s
})
