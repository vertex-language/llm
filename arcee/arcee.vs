// Package arcee is Arcee's AFM: Llama's layers with an MLP of squared
// ReLU and no gate.
package arcee

import (
    "llm/arch"
    "nn/decoder"
)

public let Names = ["ArceeForCausalLM", "arcee"]

public let Family = arch.Standard(name: "arcee", names: Names, spec: { _ in
    var s = decoder.Spec()
    s.RopeFactors = true
    s.FFNGated = false
    s.FFNActivation = .ReLUSquared
    s.AttentionScale = true
    return s
})
