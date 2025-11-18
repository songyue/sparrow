# Model Adapter Usage Guide

This guide explains how to use the various model adapters available in Sparrow, including the newly added **Alibaba Qwen (阿里千问)** and **DeepSeek** adapters.

## Table of Contents

- [Overview](#overview)
- [Available Adapters](#available-adapters)
- [Alibaba Qwen Adapter](#alibaba-qwen-adapter)
- [DeepSeek Adapter](#deepseek-adapter)
- [Other Adapters](#other-adapters)
- [Usage Examples](#usage-examples)

## Overview

Sparrow supports multiple LLM backends through a unified adapter interface. Each adapter provides a consistent way to interact with different model providers, allowing you to easily switch between services.

## Available Adapters

| Adapter | Provider | API Type | Environment Variable |
|---------|----------|----------|---------------------|
| `qwen` | Alibaba Cloud | DashScope API | `DASHSCOPE_API_KEY` or `QWEN_API_KEY` |
| `deepseek` | DeepSeek | OpenAI-compatible | `DEEPSEEK_API_KEY` |
| `openai` | OpenAI | Native | `OPENAI_API_KEY` |
| `hfapi` | Hugging Face | Inference API | `HF_API_KEY` or `HF_TOKEN` |
| `hf_local` | Hugging Face | Local transformers | N/A |
| `llamacpp` | Local | llama.cpp | N/A |

## Alibaba Qwen Adapter

### Setup

1. **Get API Key**: Sign up for Alibaba Cloud DashScope service at https://dashscope.aliyun.com/
2. **Set Environment Variable**:
   ```bash
   export DASHSCOPE_API_KEY="your-api-key-here"
   # or
   export QWEN_API_KEY="your-api-key-here"
   ```

### Supported Models

- `qwen-turbo` - Fast, cost-effective
- `qwen-plus` - Balanced performance
- `qwen-max` - Best performance
- `qwen-max-longcontext` - Extended context window

### CLI Usage

```bash
# Basic usage
./sparrow.sh "your query text" \
  --pipeline "sparrow-parse" \
  --options qwen \
  --options qwen-plus \
  --file-path "document.pdf"

# With additional options
./sparrow.sh '[{"field":"str"}]' \
  --pipeline "sparrow-parse" \
  --options qwen \
  --options qwen-max \
  --options validation_off \
  --file-path "data/invoice.pdf"
```

### API Usage

```bash
curl -X POST 'http://localhost:8002/api/v1/sparrow-llm/inference' \
  -H 'Content-Type: multipart/form-data' \
  -F 'query=your query' \
  -F 'pipeline=sparrow-parse' \
  -F 'options=qwen,qwen-plus' \
  -F 'file=@document.pdf'
```

### Python Usage

```python
from model_adapters import QwenAdapter

# Initialize adapter
adapter = QwenAdapter(model="qwen-plus", api_key="your-api-key")

# Generate response
result = adapter.generate("What is the total amount in this invoice?")
print(result["text"])
```

## DeepSeek Adapter

### Setup

1. **Get API Key**: Sign up at https://platform.deepseek.com/
2. **Set Environment Variable**:
   ```bash
   export DEEPSEEK_API_KEY="your-api-key-here"
   ```

### Supported Models

- `deepseek-chat` - General chat model
- `deepseek-coder` - Specialized for code

### CLI Usage

```bash
# Basic usage
./sparrow.sh "your query text" \
  --pipeline "sparrow-parse" \
  --options deepseek \
  --options deepseek-chat \
  --file-path "document.pdf"

# For code-related queries
./sparrow.sh "extract code snippets" \
  --pipeline "sparrow-parse" \
  --options deepseek \
  --options deepseek-coder \
  --file-path "code_document.pdf"
```

### API Usage

```bash
curl -X POST 'http://localhost:8002/api/v1/sparrow-llm/inference' \
  -H 'Content-Type: multipart/form-data' \
  -F 'query=your query' \
  -F 'pipeline=sparrow-parse' \
  -F 'options=deepseek,deepseek-chat' \
  -F 'file=@document.pdf'
```

### Python Usage

```python
from model_adapters import DeepSeekAdapter

# Initialize adapter
adapter = DeepSeekAdapter(model="deepseek-chat", api_key="your-api-key")

# Generate response
result = adapter.generate("Summarize this document")
print(result["text"])
```

## Other Adapters

### OpenAI Adapter

```bash
export OPENAI_API_KEY="your-api-key"

./sparrow.sh "query" \
  --pipeline "sparrow-parse" \
  --options openai \
  --options gpt-4o \
  --file-path "document.pdf"
```

### Hugging Face Inference API

```bash
export HF_API_KEY="your-hf-token"

./sparrow.sh "query" \
  --pipeline "sparrow-parse" \
  --options hfapi \
  --options "username/model-name" \
  --file-path "document.pdf"
```

## Usage Examples

### Example 1: Invoice Extraction with Qwen

```bash
# Extract specific fields using JSON schema
./sparrow.sh '[{"invoice_number":"str","date":"str","total":"str"}]' \
  --pipeline "sparrow-parse" \
  --options qwen \
  --options qwen-max \
  --file-path "data/invoice.pdf"
```

### Example 2: Table Extraction with DeepSeek

```bash
./sparrow.sh '[{"item":"str", "amount":0}]' \
  --pipeline "sparrow-parse" \
  --options deepseek \
  --options deepseek-chat \
  --options tables_only \
  --file-path "data/financial_table.png"
```

### Example 3: Instruction-based Query with Qwen

```bash
./sparrow.sh "check if total amount exceeds $1000" \
  --pipeline "sparrow-parse" \
  --instruction \
  --options qwen \
  --options qwen-turbo \
  --file-path "invoice.jpg"
```

### Example 4: Document Analysis with Qwen

```bash
# Use specific query with validation
./sparrow.sh '[{"company_name":"str","address":"str"}]' \
  --pipeline "sparrow-parse" \
  --options qwen \
  --options qwen-plus \
  --file-path "document.pdf"
```

## Troubleshooting

### Common Issues

**1. API Key Not Found**
```
RuntimeError: DASHSCOPE_API_KEY not set in environment
```
**Solution**: Make sure to set the appropriate environment variable before running.

**2. Empty Prompt Error**
```
ValueError: Qwen adapter requires a non-empty prompt. The adapter does not support document/image inputs directly.
```
**Solution**: The text-based adapters (Qwen, OpenAI, DeepSeek, HFInferenceAPI) do not support the `*` (query all data) option as they require explicit text prompts. Use a specific JSON schema query or instruction instead:
```bash
# ❌ This will fail with text-based adapters
./sparrow.sh '*' --pipeline "sparrow-parse" --options qwen --options qwen-plus --file-path "doc.pdf"

# ✅ Use a specific query instead
./sparrow.sh '[{"field":"str"}]' --pipeline "sparrow-parse" --options qwen --options qwen-plus --file-path "doc.pdf"

# ✅ Or use an instruction query
./sparrow.sh "extract invoice details" --pipeline "sparrow-parse" --instruction --options qwen --options qwen-plus --file-path "doc.pdf"
```

**3. Rate Limiting**
Both Qwen and DeepSeek have rate limits. If you encounter rate limit errors, consider:
- Adding delays between requests
- Upgrading your API plan
- Implementing retry logic with exponential backoff

**4. Request Timeout**
The default timeout is 60 seconds. For large documents or complex queries, you may need to increase this in the adapter code.

## Notes

- **Qwen Adapter**: Uses Alibaba's DashScope API format with nested message structure
- **DeepSeek Adapter**: Uses OpenAI-compatible API format for easy migration
- Both adapters support JSON-formatted responses for structured data extraction
- The adapters are designed to work seamlessly with Sparrow's existing pipeline infrastructure
- **Important**: Text-based adapters (Qwen, OpenAI, DeepSeek, HFInferenceAPI) do not support vision/document inputs and require explicit text prompts. Use MLX, Hugging Face, or Ollama backends for document vision capabilities.

## Support

For issues or questions:
- GitHub Issues: https://github.com/katanaml/sparrow/issues
- Email: abaranovskis@redsamuraiconsulting.com
