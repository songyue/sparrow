from sparrow_parse.vllm.inference_base import ModelInference
import ollama
import os
import json
import re
import logging
from datetime import datetime


class OllamaInference(ModelInference):
    """
        A class for performing inference using the Ollama model.
        Handles image preprocessing, response formatting, and model interaction.
        """

    def __init__(self, model_name):
        """
        Initialize the inference class with the given model name.

        :param model_name: Name of the model to load.
        """
        self.model_name = model_name
        print(f"Ollama initialized for model: {model_name}")
        # Configure logging for network requests
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        
    def process_response(self, output_text):
        """
        Process and clean the model's raw output to format as JSON.
        """
        try:
            # Check if we have markdown code block markers
            if "```" in output_text:
                # Handle markdown-formatted output
                json_start = output_text.find("```json")
                if json_start != -1:
                    # Extract content between ```json and ```
                    content = output_text[json_start + 7:]
                    json_end = content.rfind("```")
                    if json_end != -1:
                        content = content[:json_end].strip()
                        formatted_json = json.loads(content)
                        return json.dumps(formatted_json, indent=2, ensure_ascii=False)

            # Handle raw JSON (no markdown formatting)
            # First try to find JSON array or object patterns
            for pattern in [r'\[\s*\{.*\}\s*\]', r'\{.*\}']:
                matches = re.search(pattern, output_text, re.DOTALL)
                if matches:
                    potential_json = matches.group(0)
                    try:
                        formatted_json = json.loads(potential_json)
                        return json.dumps(formatted_json, indent=2, ensure_ascii=False)
                    except:
                        pass

            # Last resort: try to parse the whole text as JSON
            formatted_json = json.loads(output_text.strip())
            return json.dumps(formatted_json, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"Failed to parse JSON: {e}")
            return output_text


    def inference(self, input_data, apply_annotation=False, precision_callback=None, mode=None):
        """
        Perform inference on input data using the specified model.

        :param input_data: A list of dictionaries containing image file paths and text inputs.
        :param apply_annotation: Optional flag to apply annotations to the output.
        :param precision_callback: Optional callback function to modify input data before inference.
        :param mode: Optional mode for inference ("static" for simple JSON output).
        :return: List of processed model responses.
        """

        # Validate input_data
        if not input_data or not isinstance(input_data, list) or len(input_data) == 0:
            raise ValueError("input_data must be a non-empty list")

        # Ollama backend doesn't support annotations yet
        apply_annotation = False

        # Handle static mode
        if mode == "static":
            return [self.get_simple_json()]

        # Determine if we're doing text-only or image-based inference
        is_text_only = input_data[0].get("file_path") is None

        if is_text_only:
            # Text-only inference
            messages = input_data[0]["text_input"]
            response = self._generate_text_response(messages)
            results = [response]
        else:
            # Image-based inference
            file_paths = self._extract_file_paths(input_data)
            results = self._process_images(file_paths, input_data, apply_annotation, precision_callback)

        return results


    def _generate_text_response(self, messages):
        """
        Generate a text response for text-only inputs.

        :param messages: Input messages
        :return: Generated response
        """
        self.logger.info("=" * 80)
        self.logger.info("Starting Ollama text inference")
        self.logger.info(f"Model: {self.model_name}")
        self.logger.info(f"Message length: {len(str(messages))} characters")
        
        try:
            self.logger.info("Sending request to Ollama...")
            request_start = datetime.now()
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': messages
                    }
                ]
            )
            request_time = (datetime.now() - request_start).total_seconds()
            self.logger.info(f"Received response from Ollama (took {request_time:.2f}s)")
            self.logger.info("Inference completed successfully")
            self.logger.info("=" * 80)
            return self.process_response(response['message']['content'])
        except Exception as e:
            self.logger.error(f"Error during text inference: {e}")
            self.logger.error(f"Error type: {type(e).__name__}")
            self.logger.error("=" * 80)
            raise


    def _process_images(self, file_paths, input_data, apply_annotation, precision_callback):
        """
        Process images and generate responses for each.
        """
        self.logger.info("=" * 80)
        self.logger.info("Starting Ollama image inference")
        self.logger.info(f"Model: {self.model_name}")
        self.logger.info(f"Processing {len(file_paths)} image(s)")
        
        results = []
        for idx, file_path in enumerate(file_paths, 1):
            try:
                self.logger.info(f"Processing image {idx}/{len(file_paths)}: {file_path}")
                
                # Check if file exists
                if not os.path.exists(file_path):
                    self.logger.warning(f"File does not exist: {file_path}")
                    continue

                # Prepare messages based on model type
                messages = self._prepare_messages(file_path, input_data, apply_annotation, precision_callback)

                # Handle different message formats for Ollama API
                if isinstance(messages, list):
                    # For Qwen: messages is a list of message dicts, add images to the last user message
                    ollama_messages = messages.copy()
                    # Find the last user message and add images
                    for msg in reversed(ollama_messages):
                        if msg['role'] == 'user':
                            msg['images'] = [file_path]
                            break
                else:
                    # For Mistral: messages is a string, wrap in standard message format
                    ollama_messages = [
                        {
                            'role': 'user',
                            'content': messages,
                            'images': [file_path]
                        }
                    ]

                # Make the multimodal request to Ollama
                self.logger.info(f"Sending request to Ollama for image {idx}...")
                request_start = datetime.now()
                response = ollama.chat(
                    model=self.model_name,
                    messages=ollama_messages
                )
                request_time = (datetime.now() - request_start).total_seconds()
                self.logger.info(f"Received response from Ollama (took {request_time:.2f}s)")

                # Process the raw response
                processed_response = self.process_response(response['message']['content'])

                results.append(processed_response)
                self.logger.info(f"Inference completed successfully for: {file_path}")

            except Exception as e:
                self.logger.error(f"Error processing image {file_path}: {e}")
                self.logger.error(f"Error type: {type(e).__name__}")
                # Continue processing other images instead of failing completely
                continue

        self.logger.info(f"Successfully processed {len(results)}/{len(file_paths)} image(s)")
        self.logger.info("=" * 80)
        return results


    def _prepare_messages(self, file_path, input_data, apply_annotation, precision_callback):
        """
        Prepare the appropriate messages based on the model type.

        :param input_data: Original input data
        :param apply_annotation: Flag to apply annotations
        :return: Properly formatted messages
        """
        if "mistral" or "olmocr" or "gemma"in self.model_name.lower():
            if precision_callback is not None:
                input_data = precision_callback(file_path, input_data)

            return input_data[0]["text_input"]
        elif "qwen" in self.model_name.lower():
            if precision_callback is not None:
                input_data = precision_callback(file_path, input_data)

            return input_data[0]["text_input"]
        else:
            raise ValueError("Unsupported model type. Please use either Mistral or Qwen.")


    @staticmethod
    def _extract_file_paths(input_data):
        """
        Extract and resolve absolute file paths from input data.

        :param input_data: List of dictionaries containing image file paths.
        :return: List of absolute file paths.
        """
        return [
            os.path.abspath(file_path)
            for data in input_data
            for file_path in data.get("file_path", [])
        ]