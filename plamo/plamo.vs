// Package plamo is Preferred Networks' PLaMo (1) so far: attention and
// the MLP side by side from one norm. PLaMo 2 and 3 join with their
// hybrid and windowed layers.
package plamo

import (
    "llm/arch"
    "nn/decoder"
)

public let Names = ["PlamoForCausalLM", "plamo"]

public let Family = arch.Standard(name: "plamo", names: Names, spec: { _ in
    var s = decoder.Spec()
    s.Rope = .neox
    s.Parallel = true
    s.FFNNorm = nil
    s.AttnOutBias = false
    return s
})
