// Package llama is the Llama family for text generation: the nn/llama
// network -- Llama 1 to 3, TinyLlama, SmolLM2, Mistral 7B, most fine-tunes
// -- with a tokenizer and a memory, as llm's contract asks.
//
// Everything about the network is nn/llama's, so that the other tasks
// that run a Llama share it; this package is what generating text with
// one adds: the family's names and Load, Decode over llm's batches, and
// greedy decoding run on the device.
package llama

import (
    "fs"
    "gpu"
    "gpu/neural"
    "llm/arch"
    "model"
    "nn/cache"
    "nn/decoder"
    "text/tokenizer"
    net "nn/llama"
)

/// LoadError is a checkpoint that is not a llama this can run, and says why.
public typealias LoadError = net.LoadError

/// Config is a llama's shape.
public typealias Config = net.Config

/// Names are the architecture names this family loads: Hugging Face's and
/// GGUF's.
public let Names = net.Names

/// Family is the Llama family, for llm.Register and the facade's list.
public let Family = arch.Family(name: "llama", names: Names, load: { ck, d in
    return try await Model.Load(ck, on: d)
})

/// Model is a llama on a device, with a memory of one sequence (more with
/// Load's sequences).
public final class Model: arch.GreedyModel {
    public let Config: net.Config
    public let Network: decoder.Network
    public let Tokenizer: tokenizer.Tokenizer
    public let KV: cache.KV
    // The sequence's tokens, as the device writes and reads them when it
    // runs the decoding loop itself (Generate).
    let _tokens: gpu.Buffer<int32>

    init(_ config: net.Config, _ network: decoder.Network, _ tok: tokenizer.Tokenizer, sequences: int) throws {
        self.Config = config
        self.Network = network
        self.Tokenizer = tok
        self.KV = network.Memory(sequences: sequences)
        self._tokens = try network.Tokens(count: network.Context + 1)
    }


    public var Device: gpu.Device { return Network.Device }
    public var Memory: cache.Memory { return KV }

    /// Context is how many positions a sequence holds.
    public var Context: int { return Network.Context }

    /// Load reads the llama at path -- a GGUF file, or a Hugging Face
    /// snapshot directory -- onto device d.
    public static func Load(_ path: fs.Path, on d: gpu.Device) async throws -> Model {
        return try await Load(try model.Open(path), on: d)
    }

    /// Load reads a llama checkpoint onto device d, with a memory of
    /// sequences sequences.
    public static func Load(_ ck: model.Checkpoint, on d: gpu.Device, sequences: int = 1) async throws -> Model {
        let network = try await net.Load(ck, on: d)
        return try Model(try net.Config.Read(ck), network, try ck.Tokenizer(), sequences: sequences)
    }

    /// Decode runs a batch through the model (llm/arch's Model.Decode).
    public func Decode(_ b: arch.Batch, _ out: arch.Output) async throws -> gpu.Buffer<float32> {
        let runs = try b.Runs()
        var wanted: [int] = []
        for i in 0..<b.Count where b.Outputs[i] {
            wanted.append(i)
        }
        var hidden = false
        switch out {
        case .logits: hidden = false
        case .hidden: hidden = true
        }
        if let e = b.Embeddings {
            return try await Network.Forward(decoder.Input.embeddings(e), runs: runs, wanted: wanted, hidden: hidden, memory: KV)
        }
        return try await Network.Forward(decoder.Input.tokens(b.Tokens), runs: runs, wanted: wanted, hidden: hidden, memory: KV)
    }

    /// Forward runs token at position of sequence 0 through the model and
    /// is its logits: a one-row Decode.
    public func Forward(_ token: int, position: int) async throws -> gpu.Buffer<float32> {
        return try await arch.Forward(self, token, position: position)
    }

    /// Generate continues prompt greedily for at most n tokens, stopping at
    /// the end-of-sequence token, and is the tokens it made.
    ///
    /// The prompt is one batch. Then the device runs the loop: each token's
    /// greatest logit is written where the next step's embedding lookup
    /// reads it, so the host queues step after step without waiting to
    /// learn a token, and reads the tokens back a few steps at a time --
    /// the end-of-sequence token is found up to that many steps late, and
    /// what was made past it dropped.
    public func Generate(_ prompt: string, tokens n: int) async throws -> [int] {
        let ids = Tokenizer.Encode(prompt, addSpecial: true)
        if ids.count + n > Context {
            throw LoadError.unsupported("\(ids.count) prompt tokens and \(n) more are past the context of \(Context)")
        }
        if n == 0 {
            return []
        }
        // _tokens holds the prompt and then the tokens made, the first of
        // them at `first`. With no prompt, the first token is 0.
        let first = max(ids.count, 1)
        let start: [int] = ids.isEmpty ? [0] : ids
        try await _tokens.Slice(from: 0, count: start.count).Upload(start.map { int32($0) })
        let logits = try await Decode(arch.Batch.Prompt(start), .logits)
        try await neural.Argmax(logits, into: _tokens.Slice(from: first, count: 1), rows: 1, cols: logits.count)
        var made = 1
        var read = 0
        var out: [int] = []
        while true {
            var steps = 0
            while made < n && steps < 8 {
                let at = first + made - 1
                let l = try await Network.Step(tokens: _tokens, at: at, position: at, memory: KV)
                try await neural.Argmax(l, into: _tokens.Slice(from: at + 1, count: 1), rows: 1, cols: l.count)
                made += 1
                steps += 1
            }
            for t in try await _tokens.Slice(from: first + read, count: made - read).Download() {
                out.append(int(t))
                if int(t) == Tokenizer.Vocab.Eos {
                    return out
                }
            }
            read = made
            if made == n {
                return out
            }
        }
    }
}

/// Argmax is the index of the largest value, the first of equals.
public func Argmax(_ xs: [float32]) -> int {
    var best = 0
    for i in 1..<max(1, xs.count) where xs[i] > xs[best] {
        best = i
    }
    return best
}
