// Package qwen is the Qwen family for text generation: the nn/qwen
// network -- Qwen, Qwen2 and Qwen2.5, Qwen3 -- with a tokenizer and a
// memory, as llm's contract asks.
package qwen

import (
    "llm/arch"
    net "nn/qwen"
)

/// Names are the architecture names this family loads.
public let Names = net.Names

/// Family is the Qwen family, for llm.Register and the facade's list.
public let Family = arch.Family(name: "qwen", names: Names, load: { ck, d in
    return try arch.DecoderModel(try await net.Load(ck, on: d), try ck.Tokenizer())
})
