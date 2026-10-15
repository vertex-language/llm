// Package cohere is Cohere's Command R and R+ (command-r): LayerNorm with
// no bias, attention and the MLP side by side from one norm, a LayerNorm
// over each of Q's and K's heads in R+, and scaled logits. Command A
// (cohere2) and Cohere2-MoE join with sliding windows and experts.
package cohere

import (
    "llm/arch"
    "nn/decoder"
)

public let Names = ["CohereForCausalLM", "command-r"]

public let Family = arch.Standard(name: "cohere", names: Names, spec: { ck in
    var s = decoder.Spec()
    s.Norm = .layer
    s.Parallel = true
    s.FFNNorm = nil
    s.LogitScale = .multiply
    if ck.Has("blk.0.attn_q_norm.weight", gguf: "blk.0.attn_q_norm.weight") {
        s.QKNorm = .beforeRope
        s.QKNormKind = .layer
        s.QKNormEach = true
    }
    return s
})
