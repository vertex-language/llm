// Package ernie is Baidu's ERNIE 4.5, dense so far (ernie4_5). ERNIE 4.5
// MoE and PaddleOCR-VL join with experts and M-RoPE.
package ernie

import (
    "llm/arch"
    "nn/decoder"
)

public let Names = ["Ernie4_5_ForCausalLM", "Ernie4_5ForCausalLM", "ernie4_5"]

public let Family = arch.Standard(name: "ernie", names: Names, spec: { _ in
    var s = decoder.Spec()
    s.AttnOutBias = false
    return s
})
