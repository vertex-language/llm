// Package dbrx is Databricks' DBRX: LayerNorms without biases, one QKV
// projection clamped to ±clamp_kqv, NeoX rotary embeddings, and a
// mixture of experts in every layer.
package dbrx

import (
    "llm/arch"
    "nn/decoder"
)

public let Names = ["DbrxForCausalLM", "dbrx"]

public let Family = arch.Standard(name: "dbrx", names: Names, spec: { _ in
    var s = decoder.Spec()
    s.Rope = .neox
    s.Norm = .layer
    s.FFNNorm = "attn_output_norm"
    s.AttnOutBias = false
    s.MoE = true
    return s
})
