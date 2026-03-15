import ExecuTorchLLM

let runner = TextRunner(modelPath: "llama.pte", tokenizerPath: "tiktoken.bin")
try runner.generate("Hello, how are you?", Config {
    $0.sequenceLength = 128
}) { token in
    print(token, terminator: "")
}
