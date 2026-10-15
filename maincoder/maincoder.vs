// Package maincoder is Maincoder: Q and K normalized after they turn.
package maincoder

import (
    "llm/arch"
    "nn/decoder"
)

public let Names = ["MaincoderForCausalLM", "maincoder"]

public let Family = arch.Standard(name: "maincoder", names: Names, spec: { _ in
    var s = decoder.Spec()
    s.QKNorm = .afterRope
    return s
})
