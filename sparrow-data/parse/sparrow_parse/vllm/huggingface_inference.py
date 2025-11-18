from gradio_client import Client, handle_file
from sparrow_parse.vllm.inference_base import ModelInference
import json
import os
import ast
import logging
from datetime import datetime


class HuggingFaceInference(ModelInference):
    def __init__(self, hf_space, hf_token):
        self.hf_space = hf_space
        self.hf_token = hf_token
        # Configure logging for network requests
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)


    def process_response(self, output_text):
        json_string = output_text

        json_string = json_string.strip("[]'")
        json_string = json_string.replace("```json\n", "").replace("\n```", "")
        json_string = json_string.replace("'", "")

        try:
            formatted_json = json.loads(json_string)
            return json.dumps(formatted_json, indent=2)
        except json.JSONDecodeError as e:
            print("Failed to parse JSON:", e)
            return output_text


    def inference(self, input_data, apply_annotation=False, precision_callback=None, mode=None):
        if mode == "static":
            simple_json = self.get_simple_json()
            return [simple_json]

        self.logger.info("=" * 80)
        self.logger.info("Starting HuggingFace inference")
        self.logger.info(f"HuggingFace Space: {self.hf_space}")
        self.logger.info(f"HuggingFace Token: {'***' if self.hf_token else 'Not provided'}")
        self.logger.info(f"Text input: {input_data[0].get('text_input', 'N/A')}")
        
        try:
            self.logger.info(f"Connecting to HuggingFace Space: {self.hf_space}")
            start_time = datetime.now()
            client = Client(self.hf_space, hf_token=self.hf_token)
            connection_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"Successfully connected to HuggingFace Space (took {connection_time:.2f}s)")
        except Exception as e:
            self.logger.error(f"Failed to connect to HuggingFace Space: {self.hf_space}")
            self.logger.error(f"Error: {str(e)}")
            self.logger.error(f"Error type: {type(e).__name__}")
            raise

        # Extract and prepare the absolute paths for all file paths in input_data
        file_paths = [
            os.path.abspath(file_path)
            for data in input_data
            for file_path in data["file_path"]
        ]
        
        self.logger.info(f"Processing {len(file_paths)} file(s)")
        for idx, path in enumerate(file_paths, 1):
            self.logger.info(f"  File {idx}: {path}")

        # Validate file existence and prepare files for the Gradio client
        missing_files = [path for path in file_paths if not os.path.exists(path)]
        if missing_files:
            self.logger.error(f"Missing files: {missing_files}")
            raise FileNotFoundError(f"The following files do not exist: {missing_files}")
        
        self.logger.info("Preparing files for upload...")
        try:
            image_files = [handle_file(path) for path in file_paths]
            self.logger.info(f"Successfully prepared {len(image_files)} file(s) for upload")
        except Exception as e:
            self.logger.error(f"Failed to prepare files for upload: {str(e)}")
            raise

        self.logger.info("Sending prediction request to HuggingFace Space...")
        self.logger.info(f"API endpoint: /run_inference")
        try:
            request_start = datetime.now()
            results = client.predict(
                input_imgs=image_files,
                text_input=input_data[0]["text_input"],  # Single shared text input for all images
                api_name="/run_inference"  # Specify the Gradio API endpoint
            )
            request_time = (datetime.now() - request_start).total_seconds()
            self.logger.info(f"Received response from HuggingFace Space (took {request_time:.2f}s)")
            self.logger.info(f"Response type: {type(results).__name__}")
            self.logger.info(f"Response length: {len(str(results))} characters")
        except Exception as e:
            self.logger.error(f"Failed to get prediction from HuggingFace Space")
            self.logger.error(f"Error: {str(e)}")
            self.logger.error(f"Error type: {type(e).__name__}")
            raise

        # Convert the string into a Python list
        try:
            parsed_results = ast.literal_eval(results)
            self.logger.info(f"Successfully parsed {len(parsed_results)} result(s)")
        except Exception as e:
            self.logger.error(f"Failed to parse results: {str(e)}")
            self.logger.error(f"Raw results: {results[:500]}...")  # Show first 500 chars
            raise

        results_array = []
        for idx, page_output in enumerate(parsed_results, 1):
            self.logger.info(f"Processing result {idx}/{len(parsed_results)}")
            page_result = self.process_response(page_output)
            results_array.append(page_result)

        self.logger.info(f"Successfully completed inference for {len(results_array)} page(s)")
        self.logger.info("=" * 80)
        return results_array