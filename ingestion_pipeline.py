import os
import json
from pathlib import Path
from unstructured.partition.auto import partition
from typing import List, Dict

class IngestionPipeline:
    """
    Infrastructure Data Pipeline:
    Accepts a document directory and outputs structured JSONL for indexing.
    """
    def __init__(self, raw_dir: str, processed_file: str):
        self.raw_dir = Path(raw_dir)
        self.processed_file = processed_file

    def run(self):
        print(f"--- Starting Ingestion Pipeline: {self.raw_dir} ---")
        all_elements = []

        # Determine if source is a file or directory
        if self.raw_dir.is_file():
            target_files = [self.raw_dir]
        else:
            target_files = list(self.raw_dir.rglob("*"))

        # Iterate through documents
        for doc_path in target_files:
            if doc_path.suffix.lower() in [".pdf", ".docx", ".txt", ".html"]:
                print(f"Parsing: {doc_path.name}")
                try:
                    elements = partition(filename=str(doc_path))
                    
                    # Improved Chunking: Group elements by text length (min 500 chars)
                    current_chunk = ""
                    for element in elements:
                        text = str(element).strip()
                        if not text: continue
                        
                        current_chunk += text + " "
                        
                        if len(current_chunk) > 1000: # Threshold for a good chunk size
                            all_elements.append({
                                "text": current_chunk.strip(),
                                "metadata": {
                                    "filename": doc_path.name,
                                    "chunk_size": len(current_chunk),
                                    "source_type": doc_path.suffix[1:]
                                }
                            })
                            current_chunk = ""
                    
                    # Add remaining text as last chunk
                    if current_chunk.strip():
                        all_elements.append({
                            "text": current_chunk.strip(),
                            "metadata": {"filename": doc_path.name}
                        })
                except Exception as e:
                    print(f"Failed to parse {doc_path.name}: {e}")

        # Output to structured JSONL (Append mode)
        with open(self.processed_file, "a") as f:
            for el in all_elements:
                f.write(json.dumps(el) + "\n")
        
        print(f"Ingestion complete. {len(all_elements)} elements saved to {self.processed_file}")

if __name__ == "__main__":
    # Example usage
    pipeline = IngestionPipeline(
        raw_dir="e:\\DeepDoc\\deepdoc-intelligence\\data\\raw",
        processed_file="e:\\DeepDoc\\deepdoc-intelligence\\data\\processed\\ingested_docs.jsonl"
    )
    # Ensure directories exist
    os.makedirs("e:\\DeepDoc\\deepdoc-intelligence\\data\\raw", exist_ok=True)
    os.makedirs("e:\\DeepDoc\\deepdoc-intelligence\\data\\processed", exist_ok=True)
    
    pipeline.run()
