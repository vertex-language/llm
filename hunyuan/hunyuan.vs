// Package hunyuan is Tencent's Hunyuan: Hunyuan dense V1 (hunyuan-dense),
// Hunyuan-VL's text (hunyuan_vl) and Hunyuan-MoE (hunyuan-moe, a shared
// expert beside the mixture), Q and K normalized after they turn; and HY
// V3 (hy_v3), normalized before, with the file's gating, a selection
// bias, and the experts' and shared expert's scales. HY V4 joins with
// hyper-connections.
package hunyuan

import (
    "llm/arch"
    "nn/decoder"
)

public let Names = ["HunYuanDenseV1ForCausalLM", "hunyuan-dense", "HunYuanVLForConditionalGeneration", "hunyuan_vl",
                    "HunYuanMoEV1ForCausalLM", "hunyuan-moe", "HYV3ForCausalLM", "hy_v3"]

public let Family = arch.Standard(name: "hunyuan", names: Names, spec: { ck in
    var s = decoder.Spec()
    s.Rope = .neox
    s.RopeFactors = true
    if ck.Architecture == "hy_v3" || ck.Architecture == "HYV3ForCausalLM" {
        s.QKNorm = .beforeRope
        s.FFNBias = true
        s.FFNScale = true
        s.MoE = true
        s.MoEFileGating = true
        s.MoEFileNorm = true
        s.MoEExpertScales = true
        s.MoESharedExpert = true
        return s
    }
    s.QKNorm = .afterRope
    s.MoE = true
    s.MoESharedExpert = true
    return s
})
