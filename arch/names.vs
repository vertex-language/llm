package arch

import (
    "model"
    "nn/decoder"
)

/// GGUFName is the GGUF name llama.cpp's converter gives a Hugging Face
/// tensor name, by the table most decoders share (decoder.GGUFName).
public func GGUFName(_ hf: string) -> string? {
    return decoder.GGUFName(hf)
}

/// DecoderAliases are llama.cpp's spellings of the names every decoder
/// shares (decoder.Aliases).
public let DecoderAliases: model.Aliases = decoder.Aliases
