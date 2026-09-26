// A llama written as GGUF, as llama.cpp's convert_hf_to_gguf.py writes it.
package llama

import (
    "fs"
    "llm/arch"
    "model"
    "model/gguf"
)

/// WriteGGUF writes a llama checkpoint as a GGUF file llama.cpp loads:
/// its config under llama.*, its SentencePiece vocabulary under
/// tokenizer.ggml.*, and its tensors under GGUF's names, weights in their
/// own dtype (f32, f16 or bf16) and norms as f32. The rows of Q and K are
/// permuted for interleaved RoPE, as llama.cpp's converter permutes them.
/// A GGUF checkpoint is written as it is.
public func WriteGGUF(_ ck: model.Checkpoint, to path: fs.Path) throws {
    ck.Aliases = arch.DecoderAliases
    let c = try Config.Read(ck)
    let tok = try ck.Tokenizer()
    let v = tok.Vocab
    var meta: [(string, gguf.Value)] = [
        ("general.architecture", .Text("llama")),
        ("llama.context_length", .U32(uint32(c.Context))),
        ("llama.embedding_length", .U32(uint32(c.Dim))),
        ("llama.block_count", .U32(uint32(c.Layers))),
        ("llama.feed_forward_length", .U32(uint32(c.Hidden))),
        ("llama.rope.dimension_count", .U32(uint32(c.HeadDim))),
        ("llama.attention.head_count", .U32(uint32(c.Heads))),
        ("llama.attention.head_count_kv", .U32(uint32(c.KVHeads))),
        ("llama.attention.layer_norm_rms_epsilon", .F32(c.Eps)),
        ("llama.rope.freq_base", .F32(c.RopeBase)),
        ("llama.vocab_size", .U32(uint32(c.Vocab))),
    ]
    switch tok.Algorithm {
    case .sentencePiece:
        meta.append(("tokenizer.ggml.model", .Text("llama")))
        meta.append(("tokenizer.ggml.tokens", .Array(.Text, v.Tokens.map { gguf.Value.Text($0) })))
        meta.append(("tokenizer.ggml.scores", .Array(.F32, v.Scores.map { gguf.Value.F32($0) })))
        meta.append(("tokenizer.ggml.token_type", .Array(.I32, v.Kinds.map { gguf.Value.I32(int32($0.rawValue)) })))
    case .bpe(let b):
        // The pre-tokenizer by the name llama.cpp gives its split.
        var pre = "default"
        switch b.Split {
        case .gpt2: pre = "gpt-2"
        case .digitsThenGPT2: pre = "smollm"
        case .llama3: pre = "llama-bpe"
        case .qwen2: pre = "qwen2"
        }
        meta.append(("tokenizer.ggml.model", .Text("gpt2")))
        meta.append(("tokenizer.ggml.pre", .Text(pre)))
        meta.append(("tokenizer.ggml.tokens", .Array(.Text, v.Tokens.map { gguf.Value.Text($0) })))
        meta.append(("tokenizer.ggml.token_type", .Array(.I32, v.Kinds.map { gguf.Value.I32(int32($0.rawValue)) })))
        meta.append(("tokenizer.ggml.merges", .Array(.Text, v.Merges.map { gguf.Value.Text($0) })))
    }
    if let b = v.Bos { meta.append(("tokenizer.ggml.bos_token_id", .U32(uint32(b)))) }
    if let e = v.Eos { meta.append(("tokenizer.ggml.eos_token_id", .U32(uint32(e)))) }
    if let u = v.Unknown { meta.append(("tokenizer.ggml.unknown_token_id", .U32(uint32(u)))) }
    meta.append(("tokenizer.ggml.add_bos_token", .Bool(v.AddBos)))
    meta.append(("tokenizer.ggml.add_eos_token", .Bool(v.AddEos)))
    if case .sentencePiece = tok.Algorithm {
        meta.append(("tokenizer.ggml.add_space_prefix", .Bool(v.AddSpacePrefix)))
    }
    if let t = ck.ChatTemplate { meta.append(("tokenizer.chat_template", .Text(t))) }

    // Each tensor's layout now; its bytes when it is written.
    var layouts: [gguf.Layout] = []
    var makers: [() throws -> [uint8]] = []
    func add(_ hf: string, norm: bool = false, permuteHeads: int = 0) throws {
        guard let t = ck.Info(hf), let g = arch.GGUFName(hf) else {
            throw LoadError.unsupported("no tensor \(hf)")
        }
        var type: gguf.TensorType
        switch t.SourceType {
        case "F32", "f32": type = .F32
        case "F16", "f16": type = .F16
        case "BF16", "bf16": type = .BF16
        default:
            if ck.Format == .gguf, let raw = ck.GGUF?.Tensor(t.Name) {
                type = raw.Type
            } else {
                throw LoadError.unsupported("\(hf) is \(t.SourceType)")
            }
        }
        let source = type
        let widened = norm && type != .F32
        if widened {
            type = .F32
        }
        let permute = permuteHeads > 0 && ck.Format == .safetensors
        // GGUF lists dimensions innermost first.
        layouts.append(gguf.Layout(name: g, type: type, shape: t.Shape.reversed()))
        makers.append({
            var bytes = ck.Copy(t)
            if widened {
                bytes = widen(bytes, source)
            }
            if permute {
                bytes = permuteRows(bytes, rows: t.Shape[0], heads: permuteHeads, rowBytes: bytes.count / t.Shape[0])
            }
            return bytes
        })
    }
    try add("model.embed_tokens.weight")
    try add("model.norm.weight", norm: true)
    if !c.TiedEmbeddings {
        try add("lm_head.weight")
    }
    for i in 0..<c.Layers {
        let p = "model.layers.\(i)."
        try add(p + "input_layernorm.weight", norm: true)
        try add(p + "self_attn.q_proj.weight", permuteHeads: c.Heads)
        try add(p + "self_attn.k_proj.weight", permuteHeads: c.KVHeads)
        try add(p + "self_attn.v_proj.weight")
        try add(p + "self_attn.o_proj.weight")
        try add(p + "post_attention_layernorm.weight", norm: true)
        try add(p + "mlp.gate_proj.weight")
        try add(p + "mlp.up_proj.weight")
        try add(p + "mlp.down_proj.weight")
    }
    try gguf.Save(metadata: meta, tensors: layouts, to: path, bytes: { i in try makers[i]() })
}

// permuteRows reorders each head's rows from halves to interleaved: row
// s·(d/2) + i of a head becomes row 2i + s -- convert_hf_to_gguf.py's
// permute(), reshape(heads, 2, d/2, …).swapaxes(1, 2).
func permuteRows(_ b: [uint8], rows: int, heads: int, rowBytes: int) -> [uint8] {
    var out = [uint8](repeating: 0, count: b.count)
    let d = rows / heads
    let half = d / 2
    for h in 0..<heads {
        for s in 0..<2 {
            for i in 0..<half {
                let from = (h * d + s * half + i) * rowBytes
                let to = (h * d + 2 * i + s) * rowBytes
                for k in 0..<rowBytes {
                    out[to + k] = b[from + k]
                }
            }
        }
    }
    return out
}

// widen is f16 or bf16 bytes as f32 bytes.
func widen(_ b: [uint8], _ type: gguf.TensorType) -> [uint8] {
    var out: [uint8] = []
    out.reserveCapacity(b.count * 2)
    var i = 0
    while i + 1 < b.count {
        let h = uint16(b[i]) | uint16(b[i + 1]) << 8
        let f = type == .BF16 ? float32(bitPattern: uint32(h) << 16) : float32(float16(bitPattern: h))
        let bits = f.bitPattern
        for k in 0..<4 {
            out.append(uint8(truncatingIfNeeded: bits >> uint32(8 * k)))
        }
        i += 2
    }
    return out
}
