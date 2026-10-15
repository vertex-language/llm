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
    "nn/cache"
    "text/tokenizer"
)

/// BatchError is a batch a model cannot decode, and why.
public enum BatchError: Error, CustomStringConvertible {
    case invalid(string)

    public var description: string {
        switch self {
        case .invalid(let why): return "llm: " + why
        }
    }
}

/// Batch is what a model decodes in one call: rows of tokens -- or of
/// embeddings, an image's or audio's from an adapter -- each at a position
/// of a sequence, and which rows' outputs are wanted. A prompt is one
/// sequence's run of positions; decoding several conversations is a row of
/// each; verifying a draft is the draft's rows. A sequence's rows are
/// consecutive, at consecutive positions from where its memory has
/// reached or before it (earlier positions are dropped and decoded again).
public struct Batch {
    public var Tokens: [int32]
    /// Embeddings, when set, are the rows' inputs in place of Tokens:
    /// count x the model's width.
    public var Embeddings: gpu.Buffer<float32>? = nil
    public var Positions: [int32]
    public var Sequences: [int32]
    public var Outputs: [bool]

    public init(tokens: [int32], positions: [int32], sequences: [int32], outputs: [bool]) {
        self.Tokens = tokens
        self.Positions = positions
        self.Sequences = sequences
        self.Outputs = outputs
    }

    /// Prompt is tokens as one sequence's run from position, the last
    /// row's output wanted -- or with all, every row's.
    public static func Prompt(_ tokens: [int], from position: int = 0, sequence: int = 0, all: bool = false) -> Batch {
        let n = tokens.count
        var outputs = [bool](repeating: all, count: n)
        if n > 0 { outputs[n - 1] = true }
        return Batch(tokens: tokens.map { int32($0) }, positions: (0..<n).map { int32(position + $0) },
                     sequences: [int32](repeating: int32(sequence), count: n), outputs: outputs)
    }

    /// Count is how many rows it has.
    public var Count: int { return Positions.count }

    /// Wanted is how many rows' outputs are wanted.
    public var Wanted: int {
        var n = 0
        for o in Outputs where o { n += 1 }
        return n
    }

    /// Runs are the batch's sequences in order: each one's first row, row
    /// count, first position and sequence -- checked: the rows' arrays
    /// agree, a sequence's rows are consecutive at consecutive positions,
    /// and no sequence comes twice.
    public func Runs() throws -> [cache.Run] {
        let n = Positions.count
        if Sequences.count != n || Outputs.count != n || (Embeddings == nil && Tokens.count != n) {
            throw BatchError.invalid("a batch of \(Tokens.count) tokens, \(n) positions, \(Sequences.count) sequences, \(Outputs.count) outputs")
        }
        var runs: [cache.Run] = []
        var seen: [int32] = []
        var i = 0
        while i < n {
            let s = Sequences[i]
            if seen.contains(s) {
                throw BatchError.invalid("sequence \(s)'s rows are not consecutive")
            }
            seen.append(s)
            var j = i + 1
            while j < n && Sequences[j] == s {
                if Positions[j] != Positions[j - 1] + 1 {
                    throw BatchError.invalid("sequence \(s)'s positions \(Positions[j - 1]) then \(Positions[j])")
                }
                j += 1
            }
            runs.append(cache.Run(row: i, count: j - i, position: int(Positions[i]), sequence: int(s)))
            i = j
        }
        return runs
    }
}

/// Output is what Decode writes for each wanted row.
public enum Output {
    /// The logits over the vocabulary.
    case logits
    /// The final hidden state, normalized: what an embedding pools.
    case hidden
}

/// Model is a family's model loaded on a device, with its memory: what
/// runs, whatever the family.
public protocol Model: AnyObject {
    var Tokenizer: tokenizer.Tokenizer { get }
    var Device: gpu.Device { get }
    /// Context is how many positions a sequence may hold.
    var Context: int { get }
    /// Memory is what it keeps between batches, by sequence.
    var Memory: cache.Memory { get }
    /// Decode runs a batch through the model, adding its rows to their
    /// sequences' memory, and is the output of each wanted row, one after
    /// another, on the device: wanted x vocabulary logits, or wanted x
    /// width hidden states. The buffer is the model's, good until the next
    /// Decode.
    func Decode(_ b: Batch, _ out: Output) async throws -> gpu.Buffer<float32>
}

/// Forward runs token at position of sequence 0 through m and is its
/// logits: Decode of a one-row batch.
public func Forward(_ m: any Model, _ token: int, position: int) async throws -> gpu.Buffer<float32> {
    return try await m.Decode(Batch.Prompt([token], from: position), .logits)
}

/// GreedyModel is a Model that runs greedy decoding itself -- on its
/// device, choosing each next token there, where the generic loop reads
/// every token back -- and that llm.Generate leaves it to.
public protocol GreedyModel: Model {
    /// Generate continues prompt greedily for at most n tokens, stopping at
    /// the end-of-sequence token, and is the tokens it made.
    func Generate(_ prompt: string, tokens n: int) async throws -> [int]
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
