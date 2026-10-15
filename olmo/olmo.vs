// Package olmo is AI2's OLMo: OLMo (1), LayerNorm with no weights; and
// OLMo 2, norms only after each branch and over all of Q's and K's heads.
// OLMo 3's sliding window and OLMoE join with their mechanisms.
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
    if ck.Config.Int("sliding_window", gguf: "attention.sliding_window") != nil {
        throw decoder.DecoderError.invalid("OLMo 3's sliding window is not run yet")
    }
    s.Rope = .neox
    s.PreNorms = false
    s.QKNorm = .beforeRope
    s.PostAttnNorm = "post_attention_norm"
    s.PostFFNNorm = "post_ffw_norm"
    return s
})
