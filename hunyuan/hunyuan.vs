// Package hunyuan is Tencent's Hunyuan: Hunyuan dense V1 (hunyuan-dense),
// Hunyuan-VL's text (hunyuan_vl) and Hunyuan-MoE (hunyuan-moe, a shared
// expert beside the mixture), Q and K normalized after they turn. HY V3
// and V4 join with their routing.
package hunyuan

import (
    "llm/arch"
    "nn/decoder"
)

public let Names = ["HunYuanDenseV1ForCausalLM", "hunyuan-dense", "HunYuanVLForConditionalGeneration", "hunyuan_vl",
                    "HunYuanMoEV1ForCausalLM", "hunyuan-moe"]

public let Family = arch.Standard(name: "hunyuan", names: Names, spec: { _ in
    var s = decoder.Spec()
    s.Rope = .neox
    s.RopeFactors = true
    s.QKNorm = .afterRope
    s.MoE = true
    s.MoESharedExpert = true
    return s
})
