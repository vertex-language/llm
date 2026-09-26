// Package llama is the Llama family: Llama 1 and 2, and the models that
// share their architecture (LlamaForCausalLM on the Hugging Face Hub,
// general.architecture "llama" in GGUF: TinyLlama, SmolLM2, most
// fine-tunes) -- an embedding, blocks of RMSNorm, grouped-query attention
// with rotary position embeddings, RMSNorm and a SwiGLU MLP, each added
// back to what went in, then a final RMSNorm and the head to logits.
//
// The family is written once, against model.Checkpoint: the same code
// loads a Hugging Face snapshot (safetensors) or a GGUF file. The one thing
// it must tell apart is the layout of Q and K: llama.cpp permutes their
// rows for interleaved RoPE when it writes a llama GGUF, and Hugging
// Face's are in halves.
//
// It runs one token at a time, greedily; batching and sampling
// (llm/decoder, llm/generate) grow out of it as more families come.
package llama

import (
    "fs"
    "gpu"
    "gpu/neural"
    "llm/arch"
    "model"
    "nn"
    "tensor"
    "text/tokenizer"
)

/// LoadError is a checkpoint that is not a llama this can run, and says why.
public enum LoadError: Error, CustomStringConvertible {
    case unsupported(string)

    public var Message: string {
        switch self {
        case .unsupported(let why): return "llama: " + why
        }
    }

    public var description: string { return Message }
}

/// Names are the architecture names this family loads: Hugging Face's and
/// GGUF's.
public let Names = ["LlamaForCausalLM", "llama"]

/// Family is the Llama family, for llm.Register and the facade's list.
public let Family = arch.Family(name: "llama", names: Names, load: { ck, d in
    return try await Model.Load(ck, on: d)
})

/// Config is a llama's shape.
public struct Config {
    public var Layers: int
    public var Dim: int
    public var Hidden: int
    public var Heads: int
    public var KVHeads: int
    public var HeadDim: int
    public var Vocab: int
    public var Context: int
    public var Eps: float32
    public var RopeBase: float32
    /// TiedEmbeddings is whether the head is the embedding.
    public var TiedEmbeddings: bool

    /// Read is a checkpoint's llama config, asked in config.json's keys.
    public static func Read(_ ck: model.Checkpoint) throws -> Config {
        ck.Aliases = arch.DecoderAliases
        if !Names.contains(ck.Architecture) {
            throw LoadError.unsupported("\(ck.Architecture) is not a llama")
        }
        let c = ck.Config
        func need(_ key: string) throws -> int {
            guard let n = c.Int(key) else {
                throw LoadError.unsupported("no \(key) in the config")
            }
            return n
        }
        let dim = try need("hidden_size")
        let heads = try need("num_attention_heads")
        var cfg = Config(
            Layers: try need("num_hidden_layers"),
            Dim: dim,
            Hidden: try need("intermediate_size"),
            Heads: heads,
            KVHeads: c.Int("num_key_value_heads") ?? heads,
            HeadDim: c.Int("head_dim") ?? dim / heads,
            Vocab: try need("vocab_size"),
            Context: c.Int("max_position_embeddings") ?? 2048,
            Eps: float32(c.Double("rms_norm_eps") ?? 1e-5),
            RopeBase: float32(c.Double("rope_theta") ?? 10000),
            TiedEmbeddings: false)
        if ck.Format == .gguf {
            cfg.TiedEmbeddings = !ck.Has("lm_head.weight")
            if let rope = c.Int("rope_dimension_count", gguf: "rope.dimension_count"), rope != cfg.HeadDim {
                throw LoadError.unsupported("rotary embeddings over \(rope) of a head's \(cfg.HeadDim) dimensions")
            }
        } else {
            cfg.TiedEmbeddings = c.Bool("tie_word_embeddings") ?? !ck.Has("lm_head.weight")
            if let scaling = c.Raw?["rope_scaling"], !scaling.IsNull {
                let kind = scaling["rope_type"]?.String ?? scaling["type"]?.String ?? "?"
                if kind != "default" {
                    throw LoadError.unsupported("rope_scaling '\(kind)' (Llama 3.1's) is not run yet")
                }
            }
            if let act = c.String("hidden_act"), act != "silu" {
                throw LoadError.unsupported("a '\(act)' MLP; llama's is silu")
            }
            if c.Bool("attention_bias") == true || c.Bool("mlp_bias") == true {
                throw LoadError.unsupported("biases in attention or the MLP")
            }
        }
        return cfg
    }
}

/// Block is one transformer layer.
public final class Block {
    public let AttnNorm: nn.RMSNorm
    public let Attention: nn.Attention
    public let FFNNorm: nn.RMSNorm
    public let MLP: nn.GatedMLP

    init(attnNorm: nn.RMSNorm, attention: nn.Attention, ffnNorm: nn.RMSNorm, mlp: nn.GatedMLP) {
        self.AttnNorm = attnNorm
        self.Attention = attention
        self.FFNNorm = ffnNorm
        self.MLP = mlp
    }
}

/// Model is a llama on a device, with a KV cache for one sequence.
public final class Model: arch.Model {
    public let Config: Config
    public let Tokenizer: tokenizer.Tokenizer
    public let Device: gpu.Device
    public let Embed: nn.Embedding
    public let Blocks: [Block]
    public let Norm: nn.RMSNorm
    public let Output: nn.Linear
    let _caches: [nn.Cache]
    let _x: gpu.Buffer<float32>
    let _h: gpu.Buffer<float32>
    let _logits: gpu.Buffer<float32>

    init(_ config: Config, _ tok: tokenizer.Tokenizer, _ d: gpu.Device, _ embed: nn.Embedding, _ blocks: [Block],
         _ norm: nn.RMSNorm, _ output: nn.Linear) throws {
        self.Config = config
        self.Tokenizer = tok
        self.Device = d
        self.Embed = embed
        self.Blocks = blocks
        self.Norm = norm
        self.Output = output
        var caches: [nn.Cache] = []
        for _ in 0..<config.Layers {
            caches.append(try nn.Cache(on: d, kvHeads: config.KVHeads, headDim: config.HeadDim, capacity: config.Context))
        }
        self._caches = caches
        self._x = try d.CreateBuffer(of: float32.self, count: config.Dim)
        self._h = try d.CreateBuffer(of: float32.self, count: config.Dim)
        self._logits = try d.CreateBuffer(of: float32.self, count: output.Out)
    }

    /// Context is how many positions the cache holds.
    public var Context: int { return Config.Context }

    /// Load reads the llama at path -- a GGUF file, or a Hugging Face
    /// snapshot directory -- onto device d.
    public static func Load(_ path: fs.Path, on d: gpu.Device) async throws -> Model {
        return try await Load(try model.Open(path), on: d)
    }

    /// Load reads a llama checkpoint onto device d. Weights stay in the
    /// file's format -- f32, f16, bf16 or quantized -- and where the device
    /// reads the file's mapping in place, nothing is copied.
    public static func Load(_ ck: model.Checkpoint, on d: gpu.Device) async throws -> Model {
        let c = try llama.Config.Read(ck)
        // llama.cpp permutes Q and K for interleaved RoPE when it writes a
        // llama GGUF; Hugging Face's weights are in halves.
        let rope = nn.Rope(base: c.RopeBase, layout: ck.Format == .gguf ? neural.RopeLayout.interleaved : neural.RopeLayout.halves)
        func weight(_ name: string) async throws -> tensor.Tensor {
            return try await ck.Tensor(name, on: d)
        }
        func norm(_ name: string) async throws -> nn.RMSNorm {
            return nn.RMSNorm(try await ck.Float32(name, on: d), eps: c.Eps)
        }
        let embed = try nn.Embedding(try await weight("model.embed_tokens.weight"))
        var blocks: [Block] = []
        for i in 0..<c.Layers {
            let p = "model.layers.\(i)."
            let attention = try nn.Attention(
                qkv: try nn.Stacked([try nn.Linear(try await weight(p + "self_attn.q_proj.weight")),
                                     try nn.Linear(try await weight(p + "self_attn.k_proj.weight")),
                                     try nn.Linear(try await weight(p + "self_attn.v_proj.weight"))]),
                o: try nn.Linear(try await weight(p + "self_attn.o_proj.weight")),
                heads: c.Heads, kvHeads: c.KVHeads, headDim: c.HeadDim, rope: rope)
            let mlp = try nn.GatedMLP.Fused(
                gate: try nn.Linear(try await weight(p + "mlp.gate_proj.weight")),
                up: try nn.Linear(try await weight(p + "mlp.up_proj.weight")),
                down: try nn.Linear(try await weight(p + "mlp.down_proj.weight")),
                activation: .SiLU)
            blocks.append(Block(attnNorm: try await norm(p + "input_layernorm.weight"), attention: attention,
                                ffnNorm: try await norm(p + "post_attention_layernorm.weight"), mlp: mlp))
        }
        let final = try await norm("model.norm.weight")
        let output = c.TiedEmbeddings ? try nn.Linear(embed.Weight) : try nn.Linear(try await weight("lm_head.weight"))
        return try Model(c, try ck.Tokenizer(), d, embed, blocks, final, output)
    }

    /// Forward runs token at position through the model, adding it to the
    /// cache, and is its logits over the vocabulary.
    public func Forward(_ token: int, position: int) async throws -> gpu.Buffer<float32> {
        try await Embed.Lookup(token, into: _x)
        for i in 0..<Blocks.count {
            let b = Blocks[i]
            // Each sublayer's output is added to the residual stream by the
            // projection that makes it.
            try await b.AttnNorm.Forward(_x, into: _h)
            try await b.Attention.Forward(_h, position: position, cache: _caches[i], into: _x, accumulate: true)
            try await b.FFNNorm.Forward(_x, into: _h)
            try await b.MLP.Forward(_h, into: _x, accumulate: true)
        }
        try await Norm.Forward(_x, into: _h)
        try await Output.Forward(_h, into: _logits)
        return _logits
    }

    /// Generate continues prompt greedily for at most n tokens, stopping at
    /// the end-of-sequence token, and is the tokens it made.
    public func Generate(_ prompt: string, tokens n: int) async throws -> [int] {
        let ids = Tokenizer.Encode(prompt, addSpecial: true)
        if ids.count + n > Config.Context {
            throw LoadError.unsupported("\(ids.count) prompt tokens and \(n) more are past the context of \(Config.Context)")
        }
        var logits: [float32] = []
        var position = 0
        for id in ids {
            logits = try await Forward(id, position: position).Download()
            position += 1
        }
        var out: [int] = []
        while out.count < n {
            let next = Argmax(logits)
            out.append(next)
            if next == Tokenizer.Vocab.Eos || out.count == n {
                break
            }
            logits = try await Forward(next, position: position).Download()
            position += 1
        }
        return out
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
