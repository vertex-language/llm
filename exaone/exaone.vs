// Package exaone is LG's EXAONE: EXAONE 3 and 3.5 (exaone) so far.
package exaone

import (
    "llm/arch"
    "nn/decoder"
)

public let Names = ["ExaoneForCausalLM", "exaone"]

public let Family = arch.Standard(name: "exaone", names: Names, spec: { _ in
    var s = decoder.Spec()
    s.Rope = .neox
    s.RopeFactors = true
    return s
})
