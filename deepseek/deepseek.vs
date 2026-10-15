// Package deepseek is DeepSeek's language models: DeepSeek-MoE (deepseek),
// Llama's attention over leading dense layers and then mixtures of
// experts with shared experts. DeepSeek V2 and V3 (deepseek2) join with
// latent attention.
package deepseek

import (
    "llm/arch"
    "nn/decoder"
)

public let Names = ["DeepseekForCausalLM", "deepseek"]

public let Family = arch.Standard(name: "deepseek", names: Names, spec: { _ in
    var s = decoder.Spec()
    s.RopeFactors = true
    s.AttentionScale = true
    s.MoE = true
    s.MoENorm = false
    s.MoESharedExpert = true
    return s
})
