// Package stablelm is Stability's StableLM 2 and Stable Code: LayerNorm,
// partial rotary embeddings, and a LayerNorm over each of Q's and K's
// heads where the model has them.
package stablelm

import (
    "llm/arch"
    "nn/decoder"
)

public let Names = ["StableLmForCausalLM", "StableLMEpochForCausalLM", "stablelm"]

public let Family = arch.Standard(name: "stablelm", names: Names, spec: { ck in
    var s = decoder.Spec()
    s.Rope = .neox
    s.Norm = .layer
    s.NormBias = true
    s.AttnOutBias = false
    if ck.Has("blk.0.attn_q_norm.weight", gguf: "blk.0.attn_q_norm.weight") {
        s.QKNorm = .beforeRope
        s.QKNormKind = .layer
        s.QKNormEach = true
    }
    return s
})
