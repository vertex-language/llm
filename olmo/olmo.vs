// Package olmo is AI2's OLMo: OLMo (1), LayerNorm with no weights; and
// OLMo 2 and 3, norms only after each branch and over all of Q's and K's
// heads, and OLMo 3's sliding-window layers. OLMoE joins with experts.
package olmo

import (
    "llm/arch"
    "nn/decoder"
)

public let Names = ["OlmoForCausalLM", "OLMoForCausalLM", "olmo", "Olmo2ForCausalLM", "olmo2"]

public let Family = arch.Standard(name: "olmo", names: Names, spec: { ck in
    var s = decoder.Spec()
    s.AttnOutBias = false
    if ["olmo", "OlmoForCausalLM", "OLMoForCausalLM"].contains(ck.Architecture) {
        s.Norm = .layer
        s.NormWeights = false
        return s
    }
    // OLMo 3 is OLMo 2 with three sliding-window layers to a full one.
    s.SWA = true
    s.SWAPattern = 4
    s.Rope = .neox
    s.PreNorms = false
    s.QKNorm = .beforeRope
    s.PostAttnNorm = "post_attention_norm"
    s.PostFFNNorm = "post_ffw_norm"
    return s
})
