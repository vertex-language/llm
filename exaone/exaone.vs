// Package exaone is LG's EXAONE: EXAONE 3 and 3.5 (exaone); EXAONE 4
// (exaone4), norms after each branch and Q and K normalized per head,
// and in the 32B only the sliding-window layers turn; and K-EXAONE (exaone-moe),
// EXAONE 4's attention before norms, with leading dense layers and
// experts chosen by the file's gating with a shared expert.
package exaone

import (
    "llm/arch"
    "nn/decoder"
)

public let Names = ["ExaoneForCausalLM", "exaone", "Exaone4ForCausalLM", "exaone4", "ExaoneMoEForCausalLM", "exaone-moe"]

public let Family = arch.Standard(name: "exaone", names: Names, spec: { ck in
    var s = decoder.Spec()
    s.Rope = .neox
    s.RopeFactors = true
    if ck.Architecture == "exaone" || ck.Architecture == "ExaoneForCausalLM" {
        return s
    }
    s.SWA = true
    s.RopeSlidingOnly = true
    s.QKNorm = .beforeRope
    s.AttnOutBias = false
    if ck.Architecture == "exaone4" || ck.Architecture == "Exaone4ForCausalLM" {
        // llama.cpp gives only the 32B (64 layers) sliding windows, three
        // to a full layer unless the file says.
        s.SWA = ck.Config.Int("num_hidden_layers") == 64
        s.SWAPattern = 4
        s.PreNorms = false
        s.PostAttnNorm = "post_attention_norm"
        s.PostFFNNorm = "post_ffw_norm"
        return s
    }
    s.MoE = true
    s.MoEFileGating = true
    s.MoEFileNorm = true
    s.MoESharedExpert = true
    return s
})
