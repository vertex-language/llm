// Package codeshell is CodeShell: LayerNorm with biases and a GELU MLP
// with biases.
package codeshell

import (
    "llm/arch"
    "nn/decoder"
)

public let Names = ["CodeShellForCausalLM", "codeshell"]

public let Family = arch.Standard(name: "codeshell", names: Names, spec: { _ in
    var s = decoder.Spec()
    s.Rope = .neox
    s.Norm = .layer
    s.NormBias = true
    s.FFNGated = false
    s.FFNActivation = .GELUTanh
    s.FFNBias = true
    return s
})
