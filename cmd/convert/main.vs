// convert: a llama checkpoint -- a Hugging Face snapshot directory -- as a
// GGUF file llama.cpp loads, as its convert_hf_to_gguf.py writes it.
//
//     vsc run convert -- testdata/tiny-llama testdata/tiny-llama.gguf
package main

import (
    "fs"
    "model"
    "llm/llama"
)

func main() -> int32 {
    let args = CommandLine.arguments
    if args.count != 3 {
        print("usage: convert <checkpoint> <out.gguf>")
        return 2
    }
    do {
        let ck = try model.Open(fs.Path(args[1]))
        try llama.WriteGGUF(ck, to: fs.Path(args[2]))
        print("\(ck.Architecture) (\(ck.Format.Name), \(ck.Names.count) tensors) -> \(args[2])")
    } catch {
        print("\(error)")
        return 1
    }
    return 0
}
