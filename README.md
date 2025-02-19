## Start R1 

docker run -d -v ~/.ollama:/root/.ollama -p 11434:11434 -e OLLAMA_MODEL="hf.co/unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF:Q4_K_M"  --name ollama ollama/ollama

## Curl
curl  http://127.0.0.1:11434/api/chat -d '{
  "model": "hf.co/unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF:Q4_K_M",
  "messages": [
    { "role": "user", "content": "What is the following code doing? print(\"hello world\"" }
  ],
  "stream": false
}'

