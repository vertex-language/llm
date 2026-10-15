// Package granite is IBM's Granite 3 (granite): Llama's layers with
// multipliers on the embedding, each branch, the attention scores and
// the logits; and Granite 3 MoE, with a shared expert where it has one.
// Its hybrid and SWA generations join with their mechanisms.
package granite

import (
    "llm/arch"
    "nn/decoder"
)

public let Names = ["GraniteForCausalLM", "granite", "GraniteMoeForCausalLM", "GraniteMoeSharedForCausalLM", "granitemoe"]

public let Family = arch.Standard(name: "granite", names: Names, spec: { _ in
    var s = decoder.Spec()
    s.RopeFactors = true
    s.FFNBias = true
    s.AttentionScale = true
    s.ResidualScale = true
    s.LogitScale = .divide
    s.MoE = true
    s.MoESharedExpert = true
    return s
})
