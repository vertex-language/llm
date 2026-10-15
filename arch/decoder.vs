package arch

import (
    "gpu"
    "gpu/neural"
    "nn/cache"
    "nn/decoder"
    "text/tokenizer"
)

/// DecoderModel is a family's model over an nn/decoder network: the
/// network, its tokenizer and a memory, with Decode and greedy decoding on
/// the device written once for every family whose network is a
/// decoder.Network. A family's Load is its network's, wrapped:
///
///     arch.Family(name: "qwen", names: names, load: { ck, d in
///         return try arch.DecoderModel(try await qwen.Load(ck, on: d), try ck.Tokenizer())
///     })
public final class DecoderModel: GreedyModel {
    public let Network: decoder.Network
    public let Tokenizer: tokenizer.Tokenizer
    public let KV: cache.KV
    let _tokens: gpu.Buffer<int32>

    public init(_ network: decoder.Network, _ tok: tokenizer.Tokenizer, sequences: int = 1) throws {
        self.Network = network
        self.Tokenizer = tok
        self.KV = network.Memory(sequences: sequences)
        self._tokens = try network.Tokens(count: network.Context + 1)
    }

    public var Device: gpu.Device { return Network.Device }
    public var Memory: cache.Memory { return KV }
    public var Context: int { return Network.Context }

    public func Decode(_ b: Batch, _ out: Output) async throws -> gpu.Buffer<float32> {
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

    /// Generate continues prompt greedily for at most n tokens, stopping at
    /// the end-of-sequence token: the prompt as one batch, then a loop the
    /// device runs, each token's greatest logit written where the next
    /// step's lookup reads it, read back a few steps at a time.
    public func Generate(_ prompt: string, tokens n: int) async throws -> [int] {
        let ids = Tokenizer.Encode(prompt, addSpecial: true)
        if ids.count + n > Context {
            throw BatchError.invalid("\(ids.count) prompt tokens and \(n) more are past the context of \(Context)")
        }
        if n == 0 {
            return []
        }
        let first = max(ids.count, 1)
        let start: [int] = ids.isEmpty ? [0] : ids
        try await _tokens.Slice(from: 0, count: start.count).Upload(start.map { int32($0) })
        let logits = try await Decode(Batch.Prompt(start), .logits)
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
