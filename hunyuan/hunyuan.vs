// Package hunyuan is Tencent's Hunyuan, dense so far: Hunyuan dense V1
// (hunyuan-dense) and Hunyuan-VL's text (hunyuan_vl), Q and K normalized
// after they turn. Hunyuan-MoE and HY V3/V4 join with experts.
package hunyuan

import (
    "llm/arch"
    "nn/decoder"
)

public let Names = ["HunYuanDenseV1ForCausalLM", "hunyuan-dense", "HunYuanVLForConditionalGeneration", "hunyuan_vl"]

public let Family = arch.Standard(name: "hunyuan", names: Names, spec: { _ in
    var s = decoder.Spec()
    s.Rope = .neox
    s.RopeFactors = true
    s.QKNorm = .afterRope
    return s
})
