// Package glm is Zhipu's GLM: ChatGLM3 and GLM-4 (chatglm), and GLM-4-0414
// (glm4) with norms after each branch -- both an MLP whose gate and up are
// one weight; and GLM-4.5 (glm4moe), Q and K normalized per head, leading
// dense layers, then experts chosen by the file's gating with a
// selection bias and a shared expert. GLM-5 joins with sparse attention.
package glm

import (
    "llm/arch"
    "nn/decoder"
)

public let Names = ["ChatGLMModel", "ChatGLMForConditionalGeneration", "chatglm", "Glm4ForCausalLM", "glm4", "Glm4MoeForCausalLM",
                    "glm4moe"]

public let Family = arch.Standard(name: "glm", names: Names, spec: { ck in
    var s = decoder.Spec()
    s.AttnOutBias = false
    if ck.Architecture == "glm4moe" || ck.Architecture == "Glm4MoeForCausalLM" {
        // Its graph turns halves of a head, M-RoPE or not.
        s.Rope = .neox
        s.QKNorm = .beforeRope
        s.FFNNorm = "post_attention_norm"
        s.MoE = true
        s.MoEFileGating = true
        s.MoEFileNorm = true
        s.MoESharedExpert = true
        return s
    }
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
