// Package ernie is Baidu's ERNIE 4.5 (ernie4_5) and ERNIE 4.5 MoE
// (ernie4_5-moe): dense layers, then every n-th layer a mixture of
// experts with a selection bias and a shared expert. PaddleOCR-VL joins
// with M-RoPE.
package ernie

import (
    "llm/arch"
    "nn/decoder"
)

public let Names = ["Ernie4_5_ForCausalLM", "Ernie4_5ForCausalLM", "ernie4_5", "Ernie4_5_MoeForCausalLM", "ernie4_5-moe"]

public let Family = arch.Standard(name: "ernie", names: Names, spec: { _ in
    var s = decoder.Spec()
    s.AttnOutBias = false
    s.MoE = true
    s.MoESharedExpert = true
    return s
})
