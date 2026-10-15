// Package internlm is InternLM2 and 2.5. InternLM3 loads as llama.
package internlm

import (
    "llm/arch"
    "nn/decoder"
)

public let Names = ["InternLM2ForCausalLM", "internlm2"]

public let Family = arch.Standard(name: "internlm", names: Names, spec: { _ in
    return decoder.Spec()
})
