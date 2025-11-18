# Network Request Logging - 网络请求日志

## 问题 (Issue)
当执行以下命令时，由于缺乏网络请求的可见性，难以调试错误：
```bash
cd sparrow-ml/llm
./sparrow.sh "*" --pipeline "sparrow-parse" --options qwen --options qwen-plus --file-path "../llm/data/invoice_1.pdf"
```

When executing the above command, it was difficult to debug errors due to lack of visibility into network requests.

## 解决方案 (Solution)
添加了全面的网络请求日志记录功能，包括：
- 连接尝试和计时
- 文件处理信息
- 请求/响应时间
- 详细的错误信息和错误类型

Added comprehensive network request logging, including:
- Connection attempts and timing
- File processing information
- Request/response timing
- Detailed error messages and error types

## 日志示例 (Logging Examples)

### HuggingFace Inference 日志
当使用 HuggingFace Space 进行推理时，你会看到类似以下的日志：

```
[2025-11-18 12:50:15] [sparrow_parse.vllm.huggingface_inference] [INFO] ================================================================================
[2025-11-18 12:50:15] [sparrow_parse.vllm.huggingface_inference] [INFO] Starting HuggingFace inference
[2025-11-18 12:50:15] [sparrow_parse.vllm.huggingface_inference] [INFO] HuggingFace Space: qwen-plus
[2025-11-18 12:50:15] [sparrow_parse.vllm.huggingface_inference] [INFO] HuggingFace Token: ***
[2025-11-18 12:50:15] [sparrow_parse.vllm.huggingface_inference] [INFO] Text input: retrieve document data. return response in JSON format
[2025-11-18 12:50:15] [sparrow_parse.vllm.huggingface_inference] [INFO] Connecting to HuggingFace Space: qwen-plus
[2025-11-18 12:50:17] [sparrow_parse.vllm.huggingface_inference] [INFO] Successfully connected to HuggingFace Space (took 2.34s)
[2025-11-18 12:50:17] [sparrow_parse.vllm.huggingface_inference] [INFO] Processing 1 file(s)
[2025-11-18 12:50:17] [sparrow_parse.vllm.huggingface_inference] [INFO]   File 1: /home/user/sparrow/sparrow-ml/llm/data/invoice_1.pdf
[2025-11-18 12:50:17] [sparrow_parse.vllm.huggingface_inference] [INFO] Preparing files for upload...
[2025-11-18 12:50:18] [sparrow_parse.vllm.huggingface_inference] [INFO] Successfully prepared 1 file(s) for upload
[2025-11-18 12:50:18] [sparrow_parse.vllm.huggingface_inference] [INFO] Sending prediction request to HuggingFace Space...
[2025-11-18 12:50:18] [sparrow_parse.vllm.huggingface_inference] [INFO] API endpoint: /run_inference
[2025-11-18 12:50:25] [sparrow_parse.vllm.huggingface_inference] [INFO] Received response from HuggingFace Space (took 7.12s)
[2025-11-18 12:50:25] [sparrow_parse.vllm.huggingface_inference] [INFO] Response type: str
[2025-11-18 12:50:25] [sparrow_parse.vllm.huggingface_inference] [INFO] Response length: 1234 characters
[2025-11-18 12:50:25] [sparrow_parse.vllm.huggingface_inference] [INFO] Successfully parsed 1 result(s)
[2025-11-18 12:50:25] [sparrow_parse.vllm.huggingface_inference] [INFO] Processing result 1/1
[2025-11-18 12:50:25] [sparrow_parse.vllm.huggingface_inference] [INFO] Successfully completed inference for 1 page(s)
[2025-11-18 12:50:25] [sparrow_parse.vllm.huggingface_inference] [INFO] ================================================================================
```

### 错误日志示例 (Error Logging Example)
当发生错误时，你会看到详细的错误信息：

```
[2025-11-18 12:50:15] [sparrow_parse.vllm.huggingface_inference] [ERROR] Failed to connect to HuggingFace Space: invalid-space
[2025-11-18 12:50:15] [sparrow_parse.vllm.huggingface_inference] [ERROR] Error: Space 'invalid-space' does not exist
[2025-11-18 12:50:15] [sparrow_parse.vllm.huggingface_inference] [ERROR] Error type: ValueError
```

### Ollama Inference 日志
当使用 Ollama 进行推理时：

```
[2025-11-18 12:50:15] [sparrow_parse.vllm.ollama_inference] [INFO] ================================================================================
[2025-11-18 12:50:15] [sparrow_parse.vllm.ollama_inference] [INFO] Starting Ollama image inference
[2025-11-18 12:50:15] [sparrow_parse.vllm.ollama_inference] [INFO] Model: qwen2-vl:7b
[2025-11-18 12:50:15] [sparrow_parse.vllm.ollama_inference] [INFO] Processing 1 image(s)
[2025-11-18 12:50:15] [sparrow_parse.vllm.ollama_inference] [INFO] Processing image 1/1: /path/to/image.png
[2025-11-18 12:50:15] [sparrow_parse.vllm.ollama_inference] [INFO] Sending request to Ollama for image 1...
[2025-11-18 12:50:20] [sparrow_parse.vllm.ollama_inference] [INFO] Received response from Ollama (took 5.43s)
[2025-11-18 12:50:20] [sparrow_parse.vllm.ollama_inference] [INFO] Inference completed successfully for: /path/to/image.png
[2025-11-18 12:50:20] [sparrow_parse.vllm.ollama_inference] [INFO] Successfully processed 1/1 image(s)
[2025-11-18 12:50:20] [sparrow_parse.vllm.ollama_inference] [INFO] ================================================================================
```

## 使用方法 (Usage)
日志会自动输出到控制台。如果你想将日志保存到文件，可以使用重定向：

The logs are automatically output to the console. To save logs to a file, use redirection:

```bash
./sparrow.sh "*" --pipeline "sparrow-parse" --options huggingface --options qwen-plus --file-path "data/invoice_1.pdf" 2>&1 | tee sparrow.log
```

## 验证 (Verification)
运行验证测试以确认日志功能正常工作：

Run the verification test to confirm logging is working:

```bash
cd sparrow-data/parse
python verify_logging.py
```

预期输出 (Expected output):
```
✓ All tests passed (2/2)

Logging has been successfully added to:
  - HuggingFace inference (network requests via gradio_client)
  - Ollama inference (network requests via ollama API)

The logging includes:
  - Connection attempts and timing
  - File processing information
  - Request/response timing
  - Error handling with detailed error types
```

## 修改的文件 (Modified Files)
1. `sparrow-data/parse/sparrow_parse/vllm/huggingface_inference.py` - 添加了网络请求日志
2. `sparrow-data/parse/sparrow_parse/vllm/ollama_inference.py` - 添加了网络请求日志
3. `sparrow-data/parse/verify_logging.py` - 验证测试脚本

## 注意事项 (Notes)
- 日志级别设置为 INFO，可以通过环境变量调整
- 敏感信息（如 HuggingFace Token）会被遮蔽显示
- 网络请求时间以秒为单位显示
- 每个请求都有开始和结束的标记线（80个等号）便于区分

- Log level is set to INFO and can be adjusted via environment variables
- Sensitive information (like HuggingFace Token) is masked
- Network request times are shown in seconds
- Each request has start/end marker lines (80 equals signs) for easy identification
