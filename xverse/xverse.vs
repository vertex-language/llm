// Package xverse is XVERSE: Llama's layers, no output-projection bias.
package xverse

import (
    "llm/arch"
    "nn/decoder"
)

public let Names = ["XverseForCausalLM", "xverse"]

public let Family = arch.Standard(name: "xverse", names: Names, spec: { _ in
    var s = decoder.Spec()
    s.AttnOutBias = false
    return s
})
