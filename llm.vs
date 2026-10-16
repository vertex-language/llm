// Package llm is text generation: a language model by reference -- a
// name from the model registry, or a local path -- loaded onto a device
// by whichever of its families claims the checkpoint.
//
//     let m = try await llm.Load("qwen3")              // the latest tag, its default quant
//     let m = try await llm.Load("qwen3:8b-q8_0")      // a tag and a quant
//     let m = try await llm.Load("./stories15M-q4_0.gguf")
//
// A name is pulled into the Hugging Face cache once, pinned to a commit
// (model/registry: `vsc run registry -- list llm`). Hub references
// (hf.co/...) are not loaded directly: the registry is the front door.
//
// It is the facade: it imports the standard families and model/registry.
// The families import llm/arch, the contract, and never this package, so a
// program that wants one family imports it (llm/llama) and links nothing
// else and no network.
package llm

import (
    "fs"
    "gpu"
    "gpu/neural"
    "llm/arch"
    "llm/plamo"
    "llm/nemotron"
    "llm/hunyuan"
    "llm/glm"
    "llm/ernie"
    "llm/codeshell"
    "llm/dbrx"
    "llm/deepseek"
    "llm/dots"
    "llm/starcoder"
    "llm/stablelm"
    "llm/phi"
    "llm/gemma"
    "llm/cohere"
    "llm/arcee"
    "llm/bailing"
    "llm/exaone"
    "llm/granite"
    "llm/internlm"
    "llm/llama"
    "llm/maincoder"
    "llm/minimax"
    "llm/mistral"
    "llm/olmo"
    "llm/orion"
    "llm/pangu"
    "llm/qwen"
    "llm/seed"
    "llm/smollm"
    "llm/xverse"
    "model"
    "model/registry"
)

/// Model is a loaded language model, whatever its family.
public typealias Model = arch.Model

/// Batch is rows of tokens at positions of sequences: what Model.Decode
/// takes.
public typealias Batch = arch.Batch

/// Output is what Decode writes for each wanted row: logits, or hidden
/// states.
public typealias Output = arch.Output

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
let builtin: [Family] = [
    llama.Family, qwen.Family, mistral.Family, granite.Family, olmo.Family, exaone.Family, internlm.Family,
    smollm.Family, seed.Family, arcee.Family, pangu.Family, orion.Family, xverse.Family, maincoder.Family,
    gemma.Family, phi.Family, stablelm.Family, starcoder.Family, cohere.Family, glm.Family, nemotron.Family,
    plamo.Family, hunyuan.Family, codeshell.Family, ernie.Family, deepseek.Family, dots.Family,
    dbrx.Family, minimax.Family, bailing.Family,
]
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

/// Load reads a model onto device d (the default device when nil): a name
/// from the model registry, pulled into the Hugging Face cache first, or a
/// local file or directory.
public func Load(_ ref: string, on d: gpu.Device? = nil) async throws -> any Model {
    return try await Load(try model.Open(try await Locate(ref)), on: d)
}

/// Locate is the local path a reference names: a path as it is, or a
/// registry name's file, pulled if the cache does not have it.
public func Locate(_ ref: string) async throws -> fs.Path {
    if isLocal(ref) {
        return fs.Path(ref)
    }
    let lower = ref.lowercased()
    if lower.hasPrefix("hf.co/") || lower.hasPrefix("huggingface.co/") || lower.hasPrefix("https://") || lower.hasPrefix("http://") {
        throw LoadError.unsupported("\(ref): Hub references are not loaded directly; use a name from the model registry (vsc run registry -- list llm) or a local path")
    }
    let p = try registry.Find(ref)
    if p.Entry.Task != "llm" {
        throw LoadError.unsupported("\(p.ID) is a \(p.Entry.Task) model, not text generation")
    }
    return try await registry.Pull(p).Path
}

// isLocal is whether a reference is a path on this machine rather than a
// registry name: it starts with "/", "./", "../" or "~", or names
// something that exists.
func isLocal(_ ref: string) -> bool {
    if ref.hasPrefix("/") || ref.hasPrefix("./") || ref.hasPrefix("../") || ref.hasPrefix("~") {
        return true
    }
    return (try? fs.Metadata(fs.Path(ref))) != nil
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

/// Inspect opens a reference -- pulling a registry name's file first --
/// and says what it is: architecture, family, format, size and shape,
/// with nothing put on a device.
public func Inspect(_ ref: string) async throws -> Info {
    let keys = ["num_hidden_layers", "hidden_size", "num_attention_heads", "num_key_value_heads", "intermediate_size",
                "vocab_size", "max_position_embeddings", "rope_theta"]
    let ck = try model.Open(try await Locate(ref))
    ck.Aliases = arch.DecoderAliases
    var bytes: int64 = 0
    for n in ck.Names {
        bytes += int64(ck.Info(n, gguf: n)?.Size ?? 0)
    }
    var cfg: [(string, string)] = []
    for key in keys {
        if let n = ck.Config.Int(key) {
            cfg.append((key, "\(n)"))
        } else if let x = ck.Config.Double(key) {
            cfg.append((key, "\(x)"))
        }
    }
    return Info(Architecture: ck.Architecture, Family: FamilyOf(ck)?.Name, Format: ck.Format, Tensors: ck.Names.count,
                Bytes: bytes, Config: cfg)
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
    // The prompt is one batch; then a token a step.
    var next = 0
    var position = ids.count
    if !ids.isEmpty {
        next = try await greatest(try await m.Decode(arch.Batch.Prompt(ids), .logits))
    }
    var out: [int] = []
    while out.count < n {
        out.append(next)
        if next == m.Tokenizer.Vocab.Eos || out.count == n {
            break
        }
        next = try await greatest(try await arch.Forward(m, next, position: position))
        position += 1
    }
    return out
}
