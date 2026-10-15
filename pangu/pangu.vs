// Package pangu is openPangu Embedded: MLP biases and a head bias.
package pangu

import (
    "llm/arch"
    "nn/decoder"
)

public let Names = ["PanguEmbeddedForCausalLM", "pangu-embedded"]

public let Family = arch.Standard(name: "pangu", names: Names, spec: { _ in
    var s = decoder.Spec()
    s.Rope = .neox
    s.FFNBias = true
    s.OutputBias = true
    return s
})
