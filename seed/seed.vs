// Package seed is ByteDance's Seed-OSS: NeoX rotary embeddings, the norm
// before the MLP called the post-attention norm.
package seed

import (
    "llm/arch"
    "nn/decoder"
)

public let Names = ["SeedOssForCausalLM", "seed_oss"]

public let Family = arch.Standard(name: "seed", names: Names, spec: { _ in
    var s = decoder.Spec()
    s.Rope = .neox
    s.FFNNorm = "post_attention_norm"
    s.AttentionScale = true
    return s
})
