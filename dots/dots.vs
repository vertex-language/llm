// Package dots is rednote's dots.llm1 (dots1): Qwen3's attention, leading
// dense layers, then mixtures of experts chosen by sigmoid with a
// selection bias, and a shared expert.
package dots

import (
    "llm/arch"
    "nn/decoder"
)

public let Names = ["Dots1ForCausalLM", "dots1"]

public let Family = arch.Standard(name: "dots", names: Names, spec: { _ in
    var s = decoder.Spec()
    s.Rope = .neox
    s.QKNorm = .beforeRope
    s.AttnOutBias = false
    s.MoE = true
    s.MoEFileGating = true
    s.MoEFileNorm = true
    s.MoESharedExpert = true
    return s
})
