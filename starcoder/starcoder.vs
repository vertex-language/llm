// Package starcoder is BigCode's StarCoder 2 so far: LayerNorm with biases
// and a GELU MLP with biases. StarCoder (1), GPTBigCode, joins with
// learned positions.
package starcoder

import (
    "llm/arch"
    "nn/decoder"
)

public let Names = ["Starcoder2ForCausalLM", "starcoder2"]

// llama.cpp's StarCoder 2 reads no sliding window, and neither does this.
public let Family = arch.Standard(name: "starcoder", names: Names, spec: { _ in
    var s = decoder.Spec()
    s.Rope = .neox
    s.Norm = .layer
    s.NormBias = true
    s.FFNGated = false
    s.FFNActivation = .GELUTanh
    s.FFNBias = true
    return s
})
