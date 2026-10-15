// Package granite is IBM's Granite 3 (granite): Llama's layers with
// multipliers on the embedding, each branch, the attention scores and
// the logits. Its MoE, hybrid and SWA generations join with their
// mechanisms.
package granite

import (
    "llm/arch"
    "nn/decoder"
)

public let Names = ["GraniteForCausalLM", "granite"]

public let Family = arch.Standard(name: "granite", names: Names, spec: { _ in
    var s = decoder.Spec()
    s.RopeFactors = true
    s.FFNBias = true
    s.AttentionScale = true
    s.ResidualScale = true
    s.LogitScale = .divide
    return s
})
