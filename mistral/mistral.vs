// Package mistral is the Mistral family llama.cpp runs as its own
// architectures: Mistral 3 and Ministral 3 (mistral3). Mistral 7B and
// Mixtral load as llama.
package mistral

import (
    "llm/arch"
    "nn/decoder"
)

public let Names = ["Mistral3ForConditionalGeneration", "Ministral3ForCausalLM", "mistral3"]

public let Family = arch.Standard(name: "mistral", names: Names, spec: { ck in
    if decoder.Key(ck, "attention.temperature_scale") != 0 {
        throw decoder.DecoderError.invalid("Mistral 3's attention temperature scaling is not run yet")
    }
    var s = decoder.Spec()
    s.RopeFactors = true
    s.FFNBias = true
    s.FFNScale = true
    s.AttentionScale = true
    s.MoE = true
    s.MoEExpertScales = true
    return s
})
