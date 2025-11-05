from transforms.api import transform, Input, Output, TransformContext, TransformInput, TransformOutput, incremental, configure
from palantir_models.transforms import (
    OpenAiGptChatWithVisionLanguageModelInput,
    GenericVisionCompletionLanguageModelInput
)
from palantir_models.models import (
    OpenAiGptChatWithVisionLanguageModel,
    GenericVisionCompletionLanguageModel
)
from language_model_service_api.languagemodelservice_api_completion_v3 import (
    GptChatWithVisionCompletionRequest,
    GenericVisionCompletionRequest,
    GenericChatCompletionResponse
)
from language_model_service_api.languagemodelservice_api import (
    ChatMessage,
    ChatMessageRole,
    ChatMessageContent,
    Base64ImageContent,
    MultiContentChatMessage,
    ImageDetail,
    GenericMessageContent,
    GenericMessage,
    GenericMediaContent,
    MimeType
)
import logging
import json
import re
from typing import Optional, Dict, List, Any
from difflib import SequenceMatcher
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from pyspark.sql import Row
from pyspark.sql.functions import col, lit, when, udf
from pyspark.sql.types import StringType, BooleanType, StructType, StructField

logging.basicConfig(level=logging.INFO)

def create_table_extraction_prompt(markdown_tables: List[Dict] = None) -> str:
    """Create table extraction prompt with optional structured markdown tables as helper context"""
    base_prompt = """
FIRST : Check if photo is a logo, find out the company name and output "Logo" and skip it and move on to the following instructions.
SECOND : Check if photo is a header/banner, if it is give a short description of the visual and skip it and move on to the following instructions.
THIRD : Check if photo is a person/place/object/animal, if it is just give a short description of the photo and skip it and move on to the following instructions.
FOURTH : If the image has any chart/graph/plot, skip it and move on to the following instructions.
THEN : You are an expert at extracting structured data from image page. Analyze the provided image and extract ALL table content you can find."""

    if markdown_tables and len(markdown_tables) > 0:
        markdown_section = """

STRUCTURED TABLE CONTEXT (as helper for understanding the document):
The following tables were extracted from the document's markdown content to help you understand the structure and content:

"""
        for idx, table in enumerate(markdown_tables, 1):
            table_title = table.get('title', f'Table {idx}')
            table_headers = table.get('headers', [])
            table_rows = table.get('rows', [])

            markdown_section += f"**{table_title}:**\n"
            if table_headers:
                markdown_section += f"Headers: {', '.join(table_headers)}\n"
            if table_rows:
                markdown_section += f"Sample rows: {len(table_rows)} rows of data\n"
            markdown_section += "\n"

        markdown_section += "Use this structured table information as reference context to help you better understand and extract tables from the image. This context shows what tables exist in the document."
        base_prompt += markdown_section

    rules_and_format = """

Rules:
1. Only look for all tables in the image.
2. Extract the data in a structured JSON format for each of the tables identified.
3. Be consistent and provide impeccable quality extractions for all evaluations.
4. Include column headers, row data, and any metadata you can identify.
5. Be precise and comprehensive - don't miss any data points.
6. If you see multiple tables, extract all of them.
7. Include any table titles, captions, or contextual information.

Return the extracted data as a JSON object with this exact structure:
{
  "tables": [
    {
      "title": "table title or description",
      "headers": ["column1", "column2", "column3"],
      "rows": [
        ["value1", "value2", "value3"],
        ["value4", "value5", "value6"]
      ],
      "metadata": {
        "source": "description of where this table was found in the image",
        "notes": "any additional context"
      }
    }
  ],
  "confidence": "high/medium/low",
  "extraction_notes": "any challenges or observations about the extraction"
}

Be thorough, consistent and accurate in your extraction."""

    return base_prompt + rules_and_format


def call_gpt4_vision_llm(model, prompt: str, base64_content: str) -> str:
    """Call GPT-4 Vision with proper message structure for table extraction"""
    try:
        if not base64_content:
            raise ValueError("No image content provided")

        message = MultiContentChatMessage(
            contents=[
                ChatMessageContent(text=prompt),
                ChatMessageContent(
                    image=Base64ImageContent(
                        image_url=f"data:image/jpeg;base64,{base64_content}",
                        detail=ImageDetail.HIGH
                    )
                )
            ],
            role=ChatMessageRole.USER,
        )

        request = GptChatWithVisionCompletionRequest(
            [message],
            max_tokens=8000,
            temperature=0
        )

        response = model.create_chat_completion(request)
        return response.choices[0].message.content

    except Exception as e:
        logging.error(f"GPT-4 Vision API error: {str(e)}")
        return f"Error calling GPT-4 Vision API: {str(e)}"


def call_claude4_vision_llm(model, prompt: str, base64_content: str) -> str:
    """Call Claude-4 Vision LLM with vision-specific API structure for table extraction"""
    try:
        if not base64_content:
            raise ValueError("No image content provided")

        prompt_content = GenericMessageContent(text=prompt)
        image_content = GenericMessageContent(
            generic_media=GenericMediaContent(
                content=base64_content,
                mime_type=MimeType.IMAGE_PNG
            )
        )

        request: GenericVisionCompletionRequest = GenericVisionCompletionRequest([
            GenericMessage(
                contents=[prompt_content, image_content],
                role=ChatMessageRole.USER
            ),
        ], max_tokens=8000, temperature=0)

        response: GenericChatCompletionResponse = model.create_vision_completion(request)
        return response.completion

    except Exception as e:
        logging.error(f"Claude-4 Vision API error: {str(e)}")
        return f"Error calling Claude-4 Vision API: {str(e)}"


def call_gemini_vision_llm(model, prompt: str, base64_content: str) -> str:
    """Call Gemini 2.5 Pro Vision LLM with vision-specific API structure for table extraction"""
    try:
        if not base64_content:
            raise ValueError("No image content provided")

        prompt_content = GenericMessageContent(text=prompt)
        image_content = GenericMessageContent(
            generic_media=GenericMediaContent(
                content=base64_content,
                mime_type=MimeType.IMAGE_PNG
            )
        )

        request: GenericVisionCompletionRequest = GenericVisionCompletionRequest([
            GenericMessage(
                contents=[prompt_content, image_content],
                role=ChatMessageRole.USER
            ),
        ], max_tokens=8000, temperature=0)

        response: GenericChatCompletionResponse = model.create_vision_completion(request)
        return response.completion

    except Exception as e:
        logging.error(f"Gemini 2.5 Pro Vision API error: {str(e)}")
        return f"Error calling Gemini 2.5 Pro Vision API: {str(e)}"


def call_claude4_markdown_llm(model, markdown_text: str) -> str:
    """Call Claude-4 to detect and convert markdown tables to JSON format, replacing tables in-place"""
    try:
        if not markdown_text or not markdown_text.strip():
            return markdown_text

        markdown_table_prompt = """
You are an expert at detecting and converting markdown tables to structured JSON format.

Analyze the provided markdown text and:
1. Identify any markdown tables (tables with | separators)
2. Convert each table to the exact JSON format specified below
3. Replace the original markdown table text with the JSON representation, surrounded by tags
4. Keep ALL other markdown content exactly as it is (headers, paragraphs, lists, etc.)

Return the updated markdown text where ONLY markdown tables are replaced with JSON objects in this exact format:
JSON TABLE START
{
  "tables": [
    {
      "title": "table title or description",
      "headers": ["column1", "column2", "column3"],
      "rows": [
        ["value1", "value2", "value3"],
        ["value4", "value5", "value6"]
      ],
      "metadata": {
        "source": "converted from markdown table",
        "notes": "any additional context"
      }
    }
  ],
  "confidence": "high/medium/low",
  "extraction_notes": "conversion from markdown format"
}
JSON TABLE END

CRITICAL REQUIREMENTS:
- If no tables are found, return the original markdown text EXACTLY unchanged
- If tables are found, replace ONLY the markdown table portions with JSON format
- Keep all other markdown content (text, headers, lists, etc.) exactly as provided
- Do NOT modify any content that is not a markdown table
- Preserve all spacing, line breaks, and formatting of non-table content
"""

        prompt_content = GenericMessageContent(text=f"{markdown_table_prompt}\n\nMarkdown content to process:\n{markdown_text}")

        request: GenericVisionCompletionRequest = GenericVisionCompletionRequest([
            GenericMessage(
                contents=[prompt_content],
                role=ChatMessageRole.USER
            ),
        ], max_tokens=8000, temperature=0)

        response: GenericChatCompletionResponse = model.create_vision_completion(request)
        return response.completion

    except Exception as e:
        logging.error(f"Claude-4 Markdown API error: {str(e)}")
        return markdown_text


def extract_tables_from_processed_markdown(processed_markdown: str) -> List[Dict]:
    """Extract JSON table objects from processed markdown text"""
    tables = []
    try:
        table_block_pattern = r'JSON TABLE START\s*(.*?)\s*JSON TABLE END'
        table_blocks = re.findall(table_block_pattern, processed_markdown, re.DOTALL | re.IGNORECASE)

        for block in table_blocks:
            try:
                json_str = block.strip()
                json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
                parsed_json = json.loads(json_str)

                if isinstance(parsed_json, dict) and "tables" in parsed_json:
                    for table in parsed_json["tables"]:
                        table["source_llm"] = "markdown_conversion"
                        tables.append(table)

            except json.JSONDecodeError as e:
                logging.warning(f"Failed to parse JSON table block: {e}")
                continue

        if not tables:
            logging.info("No JSON TABLE START/END blocks found, trying general JSON pattern")
            json_pattern = r'\{[^{}]*"tables"[^{}]*\[.*?\].*?\}'
            matches = re.findall(json_pattern, processed_markdown, re.DOTALL)

            for match in matches:
                try:
                    json_str = match.strip()
                    json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
                    parsed_json = json.loads(json_str)

                    if isinstance(parsed_json, dict) and "tables" in parsed_json:
                        for table in parsed_json["tables"]:
                            table["source_llm"] = "markdown_conversion"
                            tables.append(table)

                except json.JSONDecodeError:
                    continue

    except Exception as e:
        logging.warning(f"Error extracting tables from markdown: {e}")

    return tables


def parse_llm_response(response_text: str) -> Dict[str, Any]:
    """Parse LLM response and extract structured table data"""
    try:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group()
            return json.loads(json_str)
        else:
            return {
                "tables": [],
                "confidence": "low",
                "extraction_notes": f"Could not parse structured data from response: {response_text[:200]}..."
            }
    except json.JSONDecodeError as e:
        logging.warning(f"Failed to parse JSON from LLM response: {e}")
        return {
            "tables": [],
            "confidence": "low",
            "extraction_notes": f"JSON parsing error: {str(e)}"
        }


def calculate_table_similarity(table1: Dict, table2: Dict) -> float:
    """Calculate similarity between two extracted tables based ONLY on content values matching"""
    try:
        rows1 = table1.get("rows", [])
        rows2 = table2.get("rows", [])

        if not rows1 and not rows2:
            return 1.0

        if not rows1 or not rows2:
            return 0.0

        content_similarity = calculate_content_similarity(rows1, rows2)

        return content_similarity

    except Exception as e:
        logging.warning(f"Error calculating table similarity: {e}")
        return 0.0


def calculate_content_similarity(rows1: List[List], rows2: List[List]) -> float:
    """Calculate detailed content similarity between table rows with exact value matching priority"""
    try:
        if not rows1 or not rows2:
            return 0.0

        normalized_rows1 = normalize_table_rows(rows1)
        normalized_rows2 = normalize_table_rows(rows2)

        exact_matches = 0
        total_cells = 0

        max_rows = max(len(normalized_rows1), len(normalized_rows2))
        min_rows = min(len(normalized_rows1), len(normalized_rows2))

        for i in range(min_rows):
            row1 = normalized_rows1[i] if i < len(normalized_rows1) else []
            row2 = normalized_rows2[i] if i < len(normalized_rows2) else []

            max_cols = max(len(row1), len(row2))
            min_cols = min(len(row1), len(row2))

            for j in range(min_cols):
                total_cells += 1
                if row1[j] == row2[j]:
                    exact_matches += 1

            total_cells += (max_cols - min_cols)

        total_cells += (max_rows - min_rows) * max(
            len(normalized_rows1[0]) if normalized_rows1 else 0,
            len(normalized_rows2[0]) if normalized_rows2 else 0
        )

        exact_match_ratio = exact_matches / total_cells if total_cells > 0 else 0.0

        return exact_match_ratio

    except Exception as e:
        logging.warning(f"Error calculating content similarity: {e}")
        return 0.0


def normalize_table_rows(rows: List[List]) -> List[List]:
    """Normalize table rows for better comparison by cleaning and standardizing values"""
    normalized = []
    for row in rows:
        normalized_row = []
        for cell in row:
            if cell is None:
                normalized_cell = ""
            else:
                cell_str = str(cell).strip()
                cell_str = ' '.join(cell_str.split())
                cell_str = cell_str.lower()
                cell_str = cell_str.replace(',', '').replace('$', '').replace('%', '')
                normalized_cell = cell_str
            normalized_row.append(normalized_cell)
        normalized.append(normalized_row)
    return normalized


def find_consensus_tables(llm_responses: Dict[str, Dict]) -> Dict[str, Any]:
    """Find consensus among LLM responses for extracted tables"""
    consensus_tables = []
    consensus_metadata = {
        "llm_count": len(llm_responses),
        "consensus_threshold": 0.7,
        "validated_tables": 0,
        "total_unique_tables": 0
    }

    try:
        all_tables = []
        for llm_name, response in llm_responses.items():
            tables = response.get("tables", [])
            for table in tables:
                table["source_llm"] = llm_name
                all_tables.append(table)

        consensus_metadata["total_unique_tables"] = len(all_tables)

        processed_indices = set()

        for i, table1 in enumerate(all_tables):
            if i in processed_indices:
                continue

            similar_tables = [table1]
            similar_indices = {i}

            for j, table2 in enumerate(all_tables):
                if j <= i or j in processed_indices:
                    continue

                similarity = calculate_table_similarity(table1, table2)
                if similarity >= consensus_metadata["consensus_threshold"]:
                    similar_tables.append(table2)
                    similar_indices.add(j)

            processed_indices.update(similar_indices)

            if len(similar_tables) >= 2:
                consensus_table = merge_consensus_table(similar_tables)
                consensus_table["validation_info"] = {
                    "llm_consensus_count": len(similar_tables),
                    "participating_llms": [t["source_llm"] for t in similar_tables],
                    "confidence_level": "high" if len(similar_tables) >= 3 else "medium"
                }
                consensus_tables.append(consensus_table)
                consensus_metadata["validated_tables"] += 1

        return {
            "consensus_tables": consensus_tables,
            "metadata": consensus_metadata,
            "extraction_successful": len(consensus_tables) > 0
        }

    except Exception as e:
        logging.error(f"Error finding consensus: {e}")
        return {
            "consensus_tables": [],
            "metadata": consensus_metadata,
            "extraction_successful": False,
            "error": str(e)
        }


def merge_consensus_table(similar_tables: List[Dict]) -> Dict:
    """Merge multiple similar table extractions into a consensus table with LLM priority"""

    max_rows = max(len(t.get("rows", [])) for t in similar_tables)
    tables_with_max_rows = [t for t in similar_tables if len(t.get("rows", [])) == max_rows]

    if len(tables_with_max_rows) == 1:
        base_table = tables_with_max_rows[0]
    else:
        llm_priority = {
            "claude4_vision": 3,
            "gemini_vision": 2,
            "gpt4_vision": 1
        }

        base_table = max(tables_with_max_rows,
                        key=lambda t: llm_priority.get(t.get("source_llm", ""), 0))

    merged_table = base_table.copy()

    all_titles = [t.get("title", "") for t in similar_tables if t.get("title")]
    all_notes = [t.get("metadata", {}).get("notes", "") for t in similar_tables if t.get("metadata", {}).get("notes")]

    merged_table["title"] = all_titles[0] if all_titles else "Extracted Table"
    merged_table["metadata"] = merged_table.get("metadata", {})
    merged_table["metadata"]["consensus_notes"] = " | ".join(filter(None, all_notes))

    return merged_table


def get_fallback_tables(llm_responses: Dict[str, Dict]) -> List[Dict]:
    """Get fallback tables when no consensus is achieved using priority order: Claude-4 > Gemini 2.5 Pro > GPT-4"""
    try:
        priority_order = ["claude4_vision", "gemini_vision", "gpt4_vision"]

        for llm_name in priority_order:
            if llm_name in llm_responses:
                llm_result = llm_responses[llm_name]
                tables = llm_result.get("tables", [])

                if tables:
                    logging.info(f"Using fallback tables from {llm_name} (extracted {len(tables)} tables)")
                    for table in tables:
                        table["fallback_source"] = llm_name
                        table["fallback_reason"] = "no_committee_consensus"
                    return tables

        logging.warning("No tables found in any LLM responses for fallback")
        return []

    except Exception as e:
        logging.error(f"Error in fallback table selection: {e}")
        return []


def process_single_row_with_llms(row_dict: Dict, llm_models: Dict) -> Dict:
    """
    Process a single row with all LLM calls.
    This function is designed to run on executor nodes.

    CRITICAL: This processes ONE row at a time to minimize memory usage per executor.
    Each executor processes its partition rows sequentially to avoid OOM.
    """
    try:
        base64_content = row_dict.get('pageImageBase64', '')
        if not base64_content:
            logging.warning(f"No pageImageBase64 content for row")
            return create_error_result(row_dict, "No image content")

        converted_markdown = row_dict.get('converted_markdown', '')
        markdown_tables = []
        updated_converted_markdown = converted_markdown

        if converted_markdown and converted_markdown.strip():
            try:
                processed_markdown = call_claude4_markdown_llm(
                    llm_models['claude4_vision'],
                    converted_markdown
                )
                updated_converted_markdown = processed_markdown
                markdown_tables = extract_tables_from_processed_markdown(processed_markdown)

                if markdown_tables:
                    logging.info(f"Extracted {len(markdown_tables)} structured tables from markdown")
            except Exception as e:
                logging.error(f"Error processing markdown: {e}")
                updated_converted_markdown = converted_markdown

        enhanced_prompt = create_table_extraction_prompt(markdown_tables)

        llm_responses = {}

        def call_llm_wrapper(llm_name, model, prompt, content):
            """Wrapper for calling LLMs"""
            try:
                if llm_name == "gpt4_vision":
                    return llm_name, call_gpt4_vision_llm(model, prompt, content)
                elif llm_name == "claude4_vision":
                    return llm_name, call_claude4_vision_llm(model, prompt, content)
                elif llm_name == "gemini_vision":
                    return llm_name, call_gemini_vision_llm(model, prompt, content)
                else:
                    return llm_name, f"Error: Unknown LLM {llm_name}"
            except Exception as e:
                logging.error(f"Error calling {llm_name}: {e}")
                return llm_name, f"Error: {str(e)}"

        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(call_llm_wrapper, llm_name, model, enhanced_prompt, base64_content): llm_name
                for llm_name, model in llm_models.items()
            }

            for future in as_completed(futures):
                try:
                    llm_name, raw_response = future.result(timeout=90)

                    if raw_response.startswith("Error:"):
                        llm_responses[llm_name] = {
                            "tables": [],
                            "confidence": "low",
                            "extraction_notes": raw_response
                        }
                    else:
                        parsed_response = parse_llm_response(raw_response)
                        llm_responses[llm_name] = parsed_response

                except Exception as e:
                    failed_llm = futures[future]
                    logging.error(f"Error processing {failed_llm} result: {e}")
                    llm_responses[failed_llm] = {
                        "tables": [],
                        "confidence": "low",
                        "extraction_notes": f"Processing failed: {str(e)}"
                    }

        consensus_result = find_consensus_tables(llm_responses)

        if consensus_result["consensus_tables"]:
            final_validated_tables = consensus_result["consensus_tables"]
        else:
            final_validated_tables = get_fallback_tables(llm_responses)

        if final_validated_tables:
            try:
                validated_tables_json = {
                    "tables": final_validated_tables,
                    "confidence": "high" if consensus_result["consensus_tables"] else "medium",
                    "extraction_notes": "Committee validated tables from vision LLMs"
                }

                json_table_content = json.dumps(validated_tables_json, indent=2)
                formatted_validated_tables = f"JSON TABLE START\n{json_table_content}\nJSON TABLE END"

                if updated_converted_markdown and "JSON TABLE START" in updated_converted_markdown:
                    table_block_pattern = r'JSON TABLE START\s*.*?\s*JSON TABLE END'
                    updated_converted_markdown = re.sub(
                        table_block_pattern,
                        '',
                        updated_converted_markdown,
                        flags=re.DOTALL | re.IGNORECASE
                    )

                    if updated_converted_markdown and updated_converted_markdown.strip():
                        updated_converted_markdown = f"{updated_converted_markdown.rstrip()}\n\n{formatted_validated_tables}"
                    else:
                        updated_converted_markdown = formatted_validated_tables
                else:
                    if updated_converted_markdown and updated_converted_markdown.strip():
                        updated_converted_markdown = f"{updated_converted_markdown}\n\n{formatted_validated_tables}"
                    else:
                        updated_converted_markdown = formatted_validated_tables

            except Exception as e:
                logging.error(f"Error formatting validated tables: {e}")

        current_timestamp = row_dict.get('timestamp', '')
        extraction_time = datetime.now().isoformat()

        if current_timestamp:
            try:
                timestamp_dict = json.loads(current_timestamp)
                timestamp_dict["tables validated"] = extraction_time
                updated_timestamp = json.dumps(timestamp_dict)
            except json.JSONDecodeError:
                updated_timestamp = json.dumps({"tables validated": extraction_time})
        else:
            updated_timestamp = json.dumps({"tables validated": extraction_time})

        result_row = {
            "primaryKey": row_dict.get("primaryKey", ""),
            "originalMediaItemRid": row_dict.get("originalMediaItemRid", ""),
            "originalPath": row_dict.get("originalPath", ""),
            "originalMediaReference": row_dict.get("originalMediaReference", ""),
            "pageNumber": row_dict.get("pageNumber", ""),
            "totalPages": row_dict.get("totalPages", ""),
            "pageBase64": row_dict.get("pageBase64", ""),
            "pageImageBase64": row_dict.get("pageImageBase64", ""),
            "status": "tables validated",
            "timestamp": updated_timestamp,
            "converted_markdown": updated_converted_markdown,
            "markdown_layout_score": row_dict.get("markdown_layout_score", ""),
            "markdown_parse_score": row_dict.get("markdown_parse_score", ""),
            "markdown_ocr_score": row_dict.get("markdown_ocr_score", ""),
            "markdown_table_score": row_dict.get("markdown_table_score", ""),
            "has_table": row_dict.get("has_table", ""),
            "has_image": row_dict.get("has_image", ""),
            "word_count": row_dict.get("word_count", ""),

            "consensus_tables": json.dumps(consensus_result["consensus_tables"]),
            "consensus_metadata": json.dumps(consensus_result["metadata"]),
            "markdown_tables": json.dumps(markdown_tables),
            "validated_tables": json.dumps(final_validated_tables),
            "gpt4_vision_parsed": json.dumps(llm_responses.get("gpt4_vision", {})),
            "claude4_vision_parsed": json.dumps(llm_responses.get("claude4_vision", {})),
            "gemini_vision_parsed": json.dumps(llm_responses.get("gemini_vision", {})),
            "consensus_achieved": str(consensus_result["extraction_successful"]),
            "error": ""
        }

        return result_row

    except Exception as e:
        logging.error(f"Error processing row: {e}")
        return create_error_result(row_dict, str(e))


def create_error_result(row_dict: Dict, error_msg: str) -> Dict:
    """Create error result row maintaining schema"""
    current_timestamp = row_dict.get('timestamp', '')
    error_time = datetime.now().isoformat()

    if current_timestamp:
        try:
            timestamp_dict = json.loads(current_timestamp)
            timestamp_dict["tables validated"] = error_time
            updated_timestamp = json.dumps(timestamp_dict)
        except json.JSONDecodeError:
            updated_timestamp = json.dumps({"tables validated": error_time})
    else:
        updated_timestamp = json.dumps({"tables validated": error_time})

    return {
        "primaryKey": row_dict.get("primaryKey", ""),
        "originalMediaItemRid": row_dict.get("originalMediaItemRid", ""),
        "originalPath": row_dict.get("originalPath", ""),
        "originalMediaReference": row_dict.get("originalMediaReference", ""),
        "pageNumber": row_dict.get("pageNumber", ""),
        "totalPages": row_dict.get("totalPages", ""),
        "pageBase64": row_dict.get("pageBase64", ""),
        "pageImageBase64": row_dict.get("pageImageBase64", ""),
        "status": "tables validated",
        "timestamp": updated_timestamp,
        "converted_markdown": row_dict.get("converted_markdown", ""),
        "markdown_layout_score": row_dict.get("markdown_layout_score", ""),
        "markdown_parse_score": row_dict.get("markdown_parse_score", ""),
        "markdown_ocr_score": row_dict.get("markdown_ocr_score", ""),
        "markdown_table_score": row_dict.get("markdown_table_score", ""),
        "has_table": row_dict.get("has_table", ""),
        "has_image": row_dict.get("has_image", ""),
        "word_count": row_dict.get("word_count", ""),
        "consensus_tables": "[]",
        "consensus_metadata": "[]",
        "markdown_tables": "[]",
        "validated_tables": "[]",
        "gpt4_vision_parsed": "[]",
        "claude4_vision_parsed": "[]",
        "gemini_vision_parsed": "[]",
        "consensus_achieved": "False",
        "error": error_msg
    }


@configure(
    profile=[
        "NUM_EXECUTORS_16",
        "EXECUTOR_MEMORY_LARGE",
        "DRIVER_MEMORY_LARGE",
    ]
)
@transform(
    input_images=Input("ri.foundry.main.dataset.1a5b18ee-5340-4615-9612-db3cf1be4e36"),
    gpt4_vision=OpenAiGptChatWithVisionLanguageModelInput("ri.language-model-service..language-model.gpt-4-1"),
    claude4_vision=GenericVisionCompletionLanguageModelInput("ri.language-model-service..language-model.anthropic-claude-4-sonnet"),
    gemini_vision=GenericVisionCompletionLanguageModelInput("ri.language-model-service..language-model.gemini-2-5-pro"),
    output=Output("ri.foundry.main.dataset.9893f816-6b6e-4ddd-93f2-fb2ba114938b")
)
def compute(
    ctx: TransformContext,
    input_images: TransformInput,
    gpt4_vision: OpenAiGptChatWithVisionLanguageModel,
    claude4_vision: GenericVisionCompletionLanguageModel,
    gemini_vision: GenericVisionCompletionLanguageModel,
    output: TransformOutput
):
    """
    MEMORY-OPTIMIZED Committee of LLMs Transform for Table Extraction

    Key optimizations:
    1. Uses mapPartitions to process data on executors (not driver)
    2. Processes rows sequentially within each partition to minimize memory
    3. Avoids .collect() calls that bring data to driver
    4. Uses DataFrame operations for non-table rows
    5. Unions DataFrames instead of combining lists
    """

    spark = ctx.spark_session
    input_df = input_images.dataframe()

    table_rows_df = input_df.filter(col('has_table') == True)
    non_table_rows_df = input_df.filter(col('has_table') != True)

    total_rows = input_df.count()
    table_rows_count = table_rows_df.count()
    non_table_rows_count = non_table_rows_df.count()

    logging.info(f"Total input rows: {total_rows}")
    logging.info(f"Rows with tables (will process with LLMs): {table_rows_count}")
    logging.info(f"Rows without tables (will add empty schema): {non_table_rows_count}")

    if table_rows_count > 50:
        logging.warning(f"Processing {table_rows_count} images with LLMs. This may incur significant costs.")

    if total_rows == 0:
        logging.warning("No input rows found. Creating empty output dataset.")
        empty_df = spark.createDataFrame([], schema=input_df.schema)
        output.write_dataframe(empty_df)
        return

    llm_models = {
        "gpt4_vision": gpt4_vision,
        "claude4_vision": claude4_vision,
        "gemini_vision": gemini_vision
    }


    def process_partition_with_llms(partition_rows):
        """
        Process partition of table rows on executor node.

        IMPORTANT: Processes rows SEQUENTIALLY within partition to minimize
        memory usage. Each executor handles its own partition independently.
        """
        results = []
        row_count = 0

        for row in partition_rows:
            try:
                row_count += 1
                if row_count % 10 == 0:
                    logging.info(f"Partition processing row {row_count}")

                row_dict = row.asDict()

                result = process_single_row_with_llms(row_dict, llm_models)
                results.append(Row(**result))

                if row_count % 10 == 0:
                    logging.info(f"Completed {row_count} rows in this partition")

            except Exception as e:
                logging.error(f"Error processing row in partition: {e}")
                error_result = create_error_result(row.asDict(), str(e))
                results.append(Row(**error_result))

        logging.info(f"Partition completed processing {row_count} rows")
        return iter(results)

    logging.info("Starting distributed processing of table rows across executors")
    start_time = time.time()

    table_results_rdd = table_rows_df.rdd.mapPartitions(process_partition_with_llms)

    output_schema = StructType([
        StructField("primaryKey", StringType(), True),
        StructField("originalMediaItemRid", StringType(), True),
        StructField("originalPath", StringType(), True),
        StructField("originalMediaReference", StringType(), True),
        StructField("pageNumber", StringType(), True),
        StructField("totalPages", StringType(), True),
        StructField("pageBase64", StringType(), True),
        StructField("pageImageBase64", StringType(), True),
        StructField("status", StringType(), True),
        StructField("timestamp", StringType(), True),
        StructField("converted_markdown", StringType(), True),
        StructField("markdown_layout_score", StringType(), True),
        StructField("markdown_parse_score", StringType(), True),
        StructField("markdown_ocr_score", StringType(), True),
        StructField("markdown_table_score", StringType(), True),
        StructField("has_table", StringType(), True),
        StructField("has_image", StringType(), True),
        StructField("word_count", StringType(), True),
        StructField("consensus_tables", StringType(), True),
        StructField("consensus_metadata", StringType(), True),
        StructField("markdown_tables", StringType(), True),
        StructField("validated_tables", StringType(), True),
        StructField("gpt4_vision_parsed", StringType(), True),
        StructField("claude4_vision_parsed", StringType(), True),
        StructField("gemini_vision_parsed", StringType(), True),
        StructField("consensus_achieved", StringType(), True),
        StructField("error", StringType(), True),
    ])

    table_results_df = spark.createDataFrame(table_results_rdd, schema=output_schema)

    table_results_df.cache()
    table_result_count = table_results_df.count()

    processing_time = time.time() - start_time
    logging.info(f"Completed distributed processing of {table_result_count} table rows in {processing_time/60:.1f} minutes")
    logging.info(f"Average time per row: {processing_time/table_result_count:.1f}s")


    logging.info(f"Processing {non_table_rows_count} non-table rows with DataFrame operations")

    processing_time_str = datetime.now().isoformat()

    def update_timestamp_udf(current_timestamp):
        """UDF to update timestamp JSON"""
        if current_timestamp:
            try:
                timestamp_dict = json.loads(current_timestamp)
                timestamp_dict["tables validated"] = processing_time_str
                return json.dumps(timestamp_dict)
            except:
                return json.dumps({"tables validated": processing_time_str})
        else:
            return json.dumps({"tables validated": processing_time_str})

    timestamp_udf_func = udf(update_timestamp_udf, StringType())

    non_table_results_df = non_table_rows_df \
        .withColumn("status", lit("tables validated")) \
        .withColumn("timestamp", timestamp_udf_func(col("timestamp"))) \
        .withColumn("consensus_tables", lit("[]")) \
        .withColumn("consensus_metadata", lit(json.dumps({
            "llm_count": 0,
            "consensus_threshold": 0.7,
            "validated_tables": 0,
            "total_unique_tables": 0,
            "reason": "has_table=False, no LLM processing performed"
        }))) \
        .withColumn("markdown_tables", lit("[]")) \
        .withColumn("validated_tables", lit("[]")) \
        .withColumn("gpt4_vision_parsed", lit("{}")) \
        .withColumn("claude4_vision_parsed", lit("{}")) \
        .withColumn("gemini_vision_parsed", lit("{}")) \
        .withColumn("consensus_achieved", lit("False")) \
        .withColumn("error", lit(""))


    logging.info("Combining table and non-table results using DataFrame union")

    column_order = [
        "primaryKey", "originalMediaItemRid", "originalPath", "originalMediaReference",
        "pageNumber", "totalPages", "pageBase64", "pageImageBase64", "status", "timestamp",
        "converted_markdown", "markdown_layout_score", "markdown_parse_score",
        "markdown_ocr_score", "markdown_table_score", "has_table", "has_image", "word_count",
        "consensus_tables", "consensus_metadata", "markdown_tables", "validated_tables",
        "gpt4_vision_parsed", "claude4_vision_parsed", "gemini_vision_parsed",
        "consensus_achieved", "error"
    ]

    table_results_ordered = table_results_df.select(*column_order)
    non_table_results_ordered = non_table_results_df.select(*column_order)

    output_df = table_results_ordered.union(non_table_results_ordered)

    final_count = output_df.count()
    successful_extractions = output_df.filter(col("consensus_achieved") == "True").count()

    logging.info("Processing complete:")
    logging.info(f"- Total rows in output: {final_count}")
    logging.info(f"- Table rows processed with LLMs: {table_result_count}")
    logging.info(f"- Non-table rows (schema only): {non_table_rows_count}")
    logging.info(f"- Successful committee consensus: {successful_extractions}")
    if table_result_count > 0:
        logging.info(f"- Committee success rate: {successful_extractions/table_result_count*100:.1f}%")

    output.write_dataframe(output_df)

    logging.info("Output written successfully")
