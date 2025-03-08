#
## Start R1 with Ollama native
* Pull image with Ollama
```bash
ollama pull hf.co/unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF:Q4_K_M
```
The model hf.co/unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF:Q4_K_M works nice on a M1 with 32 GB of RAM.

  
## Start R1 with Ollama in docker
No TPU support on Mac yet.

docker run -d -v ~/.ollama:/root/.ollama -p 11434:11434 -e OLLAMA_MODEL="hf.co/unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF:Q4_K_M"  --name ollama ollama/ollama

## Curl
curl  http://127.0.0.1:11434/api/chat -d '{
  "model": "hf.co/unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF:Q4_K_M",
  "messages": [
    { "role": "user", "content": "What is the following code doing? print(\"hello world\"" }
  ],
  "stream": false
}'

## Ideas
- Reviewing and improving docstrings for clarity.
- Suggesting edge cases for test coverage.
- Tracking breaking changes and auto-generating changelogs.
- Recommending how to split large functions into smaller parts.
- Adding context-aware comments.
- Detecting and suggesting updates for deprecated code.
