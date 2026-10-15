// Package llm is text generation: a language model by reference -- a
// local path or a Hugging Face Hub reference -- loaded onto a device by
// whichever of its families claims the checkpoint.
//
//     let m = try await llm.Load("hf.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0")
//     let m = try await llm.Load("hf.co/unsloth/Qwen3-0.6B-GGUF:Q4_K_M")
//     let m = try await llm.Load("./stories15M-q4_0.gguf")
//
// It is the facade: it imports the standard families and model/fetch.
// The families import llm/arch, the contract, and never this package, so a
// program that wants one family imports it (llm/llama) and links nothing
// else and no network.
package llm

import (
    "gpu"
    "gpu/neural"
    "llm/arch"
    "llm/llama"
    "model"
    "model/fetch"
)

/// Model is a loaded language model, whatever its family.
public typealias Model = arch.Model

/// Family is a text-generation family.
public typealias Family = arch.Family

/// LoadError is a reference that could not be loaded, and why.
public enum LoadError: Error, CustomStringConvertible {
    case unknownFamily(string, [string])
    case unsupported(string)

    public var description: string {
        switch self {
        case .unknownFamily(let a, let known):
            let what = a.isEmpty ? "a checkpoint that names no architecture" : a
            return "llm: \(what) is not a known family; known: \(known.joined(separator: ", "))"
        case .unsupported(let why):
            return "llm: " + why
        }
    }
}

/// The families built in, in the order they are asked.
let builtin: [Family] = [llama.Family]
var registered: [Family] = []

/// Register adds a family -- a third-party one, or one not built in --
/// ahead of the built-in ones.
public func Register(_ f: Family) {
    registered.append(f)
}

/// Families are the families this program can load: registered, then
/// built in.
public func Families() -> [Family] {
    return registered.reversed() + builtin
}

/// FamilyOf is the first family that claims a checkpoint, or nil.
public func FamilyOf(_ ck: model.Checkpoint) -> Family? {
    for f in Families() where f.Claims(ck) {
        return f
    }
    return nil
}

/// Load reads a model onto device d (the default device when nil): a local
/// file or directory, or a Hub reference downloaded into the Hugging Face
/// cache first.
public func Load(_ ref: string, on d: gpu.Device? = nil) async throws -> any Model {
    return try await Load(try await fetch.Open(ref), on: d)
}

/// Load reads an opened checkpoint onto device d, with the family that
/// claims it.
public func Load(_ ck: model.Checkpoint, on d: gpu.Device? = nil) async throws -> any Model {
    guard let family = FamilyOf(ck) else {
        var known: [string] = []
        for f in Families() {
            known += f.Names.isEmpty ? [f.Name] : f.Names
        }
        throw LoadError.unknownFamily(ck.Architecture, known)
    }
    return try await family.Load(ck, d ?? gpu.Default())
}

/// Info is what a reference is, read without loading it.
public struct Info {
    public var Architecture: string
    public var Family: string?
    public var Format: model.Format
    public var Tensors: int
    public var Bytes: int64
    public var Config: [(string, string)]
}

/// Inspect opens a reference -- fetching its files if it is on the Hub --
/// and says what it is: architecture, family, format, size and shape,
/// with nothing put on a device.
public func Inspect(_ ref: string) async throws -> Info {
    let keys = ["num_hidden_layers", "hidden_size", "num_attention_heads", "num_key_value_heads", "intermediate_size",
                "vocab_size", "max_position_embeddings", "rope_theta"]
    let i = try await fetch.Inspect(ref, keys: keys, aliases: arch.DecoderAliases)
    var family: string? = nil
    for f in Families() where f.Names.contains(i.Architecture) {
        family = f.Name
        break
    }
    return Info(Architecture: i.Architecture, Family: family, Format: i.Format, Tensors: i.Tensors, Bytes: i.Bytes,
                Config: i.Config)
}

/// Generate continues prompt greedily for at most n tokens, stopping at
/// the end-of-sequence token, and is the tokens it made. (llm/generate,
/// with sampling and streaming, replaces this.)
public func Generate(_ m: any Model, _ prompt: string, tokens n: int) async throws -> [int] {
    if let g = m as? any arch.GreedyModel {
        return try await g.Generate(prompt, tokens: n)
    }
    let ids = m.Tokenizer.Encode(prompt, addSpecial: true)
    if ids.count + n > m.Context {
        throw LoadError.unsupported("\(ids.count) prompt tokens and \(n) more are past the context of \(m.Context)")
    }
    // The next token is the greatest logit, found on the device: one int
    // read back a token, and none for the prompt's tokens but the last.
    var best: gpu.Buffer<int32>? = nil
    func greatest(_ logits: gpu.Buffer<float32>) async throws -> int {
        // A float's bytes viewed as an int: CreateBuffer(of: int32.self)
        // specialized here fails the compiler's SIL verifier.
        if best == nil { best = try logits.Device.CreateBuffer(of: float32.self, count: 1).View(as: int32.self) }
        try await neural.Argmax(logits, into: best!, rows: 1, cols: logits.count)
        return int(try await best!.Download()[0])
    }
    var next = 0
    var position = 0
    for i in 0..<ids.count {
        let logits = try await m.Forward(ids[i], position: position)
        position += 1
        if i == ids.count - 1 {
            next = try await greatest(logits)
        }
    }
    var out: [int] = []
    while out.count < n {
        out.append(next)
        if next == m.Tokenizer.Vocab.Eos || out.count == n {
            break
        }
        next = try await greatest(try await m.Forward(next, position: position))
        position += 1
    }
    return out
}
