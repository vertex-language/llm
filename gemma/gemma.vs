// Package gemma is Google's Gemma for text generation: Gemma (1) so far,
// √width-scaled embeddings and a GELU-gated MLP. Gemma 2, 3, 3n and 4 join
// with sliding windows, softcapping and their encoders.
package gemma

import (
    "llm/arch"
    "nn/decoder"
)

public let Names = ["GemmaForCausalLM", "gemma"]

public let Family = arch.Standard(name: "gemma", names: Names, spec: { _ in
    var s = decoder.Spec()
    s.Rope = .neox
    s.EmbedSqrtDim = true
    s.AttnOutBias = false
    s.FFNActivation = .GELUTanh
    return s
})
