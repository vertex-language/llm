// Package glm is Zhipu's GLM: ChatGLM3 and GLM-4 (chatglm), and GLM-4-0414
// (glm4) with norms after each branch -- both an MLP whose gate and up are
// one weight. GLM-4.5 and GLM-5 join with experts and latent attention.
package glm

import (
    "llm/arch"
    "nn/decoder"
)

public let Names = ["ChatGLMModel", "ChatGLMForConditionalGeneration", "chatglm", "Glm4ForCausalLM", "glm4"]

public let Family = arch.Standard(name: "glm", names: Names, spec: { ck in
    var s = decoder.Spec()
    s.AttnOutBias = false
    s.FFNFused = true
    if ck.Architecture == "glm4" || ck.Architecture == "Glm4ForCausalLM" {
        if decoder.MRope(ck) {
            s.Rope = .neox
        }
        s.PostAttnNorm = "post_attention_norm"
        s.PostFFNNorm = "post_ffw_norm"
    }
    return s
})
