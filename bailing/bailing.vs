// Package bailing is Ant Group's Ling (Bailing) MoE: Ling 1 (bailingmoe),
// Llama's attention with experts and a shared expert; and Ling 2
// (bailingmoe2), one QKV projection normalized per head, leading dense
// layers, and experts chosen by the file's gating with a selection bias.
// Ling 2.5's linear attention (bailingmoe3) joins with nn/delta.
package bailing

import (
    "llm/arch"
    "nn/decoder"
)

public let Names = ["BailingMoeForCausalLM", "bailingmoe", "BailingMoeV2ForCausalLM", "bailingmoe2"]

public let Family = arch.Standard(name: "bailing", names: Names, spec: { ck in
    var s = decoder.Spec()
    s.MoE = true
    s.MoEFileGating = true
    s.MoEFileNorm = true
    s.MoESharedExpert = true
    if ck.Architecture == "bailingmoe" || ck.Architecture == "BailingMoeForCausalLM" {
        s.RopeFactors = true
        return s
    }
    s.Rope = .neox
    s.QKNorm = .beforeRope
    return s
})
