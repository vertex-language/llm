package arch

import (
    "model"
    "nn/decoder"
)

/// Standard is a family whose network is nn/decoder Standard layers, read
/// by a Spec: its name, the architecture names it claims, and the spec for
/// a checkpoint (one a generation, where the family has several).
public func Standard(name: string, names: [string], spec: @escaping (model.Checkpoint) throws -> decoder.Spec) -> Family {
    return Family(name: name, names: names, load: { ck, d in
        return try DecoderModel(try await decoder.Build(ck, on: d, try spec(ck)), try ck.Tokenizer())
    })
}
