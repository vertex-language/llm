// Package minimax is MiniMax's language models: MiniMax-M2 (minimax-m2),
// Q and K normalized over all their heads, and a mixture of experts in
// every layer chosen by the file's gating with a selection bias.
// MiniMax-01's lightning attention and M3 join with their mechanisms.
package minimax

import (
    "llm/arch"
    "nn/decoder"
)

public let Names = ["MiniMaxM2ForCausalLM", "minimax-m2"]

public let Family = arch.Standard(name: "minimax", names: Names, spec: { _ in
    var s = decoder.Spec()
    s.Rope = .neox
    s.QKNorm = .beforeRope
    s.AttnOutBias = false
    s.MoE = true
    s.MoEFileGating = true
    return s
})
