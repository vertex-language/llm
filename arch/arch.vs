// Package arch is the contract every text-generation family implements,
// standard or third-party: a Family says whether it claims a checkpoint and
// loads one onto a device, as a Model the facade (llm) and the generation
// loop drive without knowing the family.
//
// Families import this package, never the facade, so the facade can
// import the families: database/sql's driver split. It also holds the
// decoder conventions a family reads a GGUF through: llama.cpp's config
// and tensor-name tables (DecoderAliases).
package arch

import (
    "gpu"
    "model"
    "text/tokenizer"
)

/// Model is a family's model loaded on a device, with its KV cache: what
/// runs, whatever the family.
public protocol Model: AnyObject {
    var Tokenizer: tokenizer.Tokenizer { get }
    var Device: gpu.Device { get }
    /// Context is how many positions the cache holds.
    var Context: int { get }
    /// Forward runs token at position through the model, adding it to the
    /// cache, and is its logits over the vocabulary, on the device.
    func Forward(_ token: int, position: int) async throws -> gpu.Buffer<float32>
}

/// Family is a model family: its name, whether it claims a checkpoint,
/// and how it loads one.
public struct Family {
    /// Name is the family's own: "llama".
    public let Name: string
    /// Names are the architecture names it claims, where it claims by name
    /// (Hugging Face's "LlamaForCausalLM", GGUF's "llama"): what an error
    /// lists as known.
    public let Names: [string]
    public let Claims: (model.Checkpoint) -> bool
    public let Load: (model.Checkpoint, gpu.Device) async throws -> any Model

    /// A family that claims the checkpoints whose architecture is one of
    /// names.
    public init(name: string, names: [string], load: @escaping (model.Checkpoint, gpu.Device) async throws -> any Model) {
        self.Name = name
        self.Names = names
        self.Claims = { ck in names.contains(ck.Architecture) }
        self.Load = load
    }

    /// A family that decides by what else a checkpoint holds.
    public init(name: string, claims: @escaping (model.Checkpoint) -> bool,
                load: @escaping (model.Checkpoint, gpu.Device) async throws -> any Model) {
        self.Name = name
        self.Names = []
        self.Claims = claims
        self.Load = load
    }
}
