import json
import torch
import logging
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import gc

MODEL_NAME = "huggyllama/llama-7b"
INPUT_FILE = "data/vulnerability_fix_dataset.csv"
OUTPUT_FILE = "output/generated_prompt_pairs.jsonl"
MAX_NEW = 200
TEMPERATURE = 0.7
BATCH_SIZE = 1  # lower if your GPU is tight on VRAM
GOOD_COL = "fixed_code"
BAD_COL = "vulnerable_code"
# ID_COL is not in your dataset, so we'll use row index instead
# ───────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

def build_prompt(code_snippet):
    """Template that asks the model for a NL prompt that would yield *code_snippet*."""
    return (
        "You are a helpful coding assistant.\n\n"
        f"Given the following code:\n\n{code_snippet}\n\n"
        "Please generate a realistic natural-language prompt that would cause an AI model "
        "to generate this code."
    )

def prepare_batch(examples, start_idx):
    """Prepare a batch with both good and bad code samples."""
    prompts = []
    metadata = []
    
    for i in range(len(examples[GOOD_COL])):
        # Use the row index as ID since ID_COL doesn't exist
        id_val = start_idx + i
        
        # Create prompt for good code
        prompts.append(build_prompt(examples[GOOD_COL][i]))
        metadata.append(("good", id_val, examples[GOOD_COL][i], examples[BAD_COL][i]))
        
        # Create prompt for bad code
        prompts.append(build_prompt(examples[BAD_COL][i]))
        metadata.append(("bad", id_val, examples[GOOD_COL][i], examples[BAD_COL][i]))
    
    return {"prompts": prompts, "metadata": metadata}

def process_generated_text(prompt, generated_text):
    """Process and clean the generated text."""
    if generated_text.startswith(prompt):
        return generated_text[len(prompt):].lstrip()
    
    idx = generated_text.find(prompt)
    return generated_text[idx + len(prompt):].lstrip() if idx != -1 else generated_text

def main():
    # Create output directory
    path_out = Path(OUTPUT_FILE)
    path_out.parent.mkdir(parents=True, exist_ok=True)
    
    # Load the dataset using datasets for efficient processing
    logger.info(f"Loading dataset from {INPUT_FILE}")
    dataset = Dataset.from_csv(INPUT_FILE)
    
    # Validate required columns
    missing = {GOOD_COL, BAD_COL} - set(dataset.column_names)
    if missing:
        raise ValueError(f"Required columns missing in {INPUT_FILE}: {missing}")
    
    # Load model and tokenizer separately for more control
    logger.info(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Set padding token if needed
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    
    # Load model with optimizations
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    
    # Create generator pipeline
    generator = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="cuda"
    )
    
    # Process in batches and write results immediately
    logger.info("Starting batch processing")
    records_by_id = {}
    
    with path_out.open("w") as f:
        # Process dataset in batches
        for i in range(0, len(dataset), BATCH_SIZE):
            batch = dataset.select(range(i, min(i + BATCH_SIZE, len(dataset))))
            prepared = prepare_batch(batch, i)
            
            # Generate completions
            outputs = generator(
                prepared["prompts"],
                max_new_tokens=MAX_NEW,
                do_sample=True,
                temperature=TEMPERATURE,
                pad_token_id=tokenizer.pad_token_id,
                batch_size=BATCH_SIZE * 2,  # avoid internal micro-batching
            )
            
            # Process results
            for (kind, id_val, good_code, bad_code), prompt, output in zip(
                prepared["metadata"], prepared["prompts"], outputs
            ):
                gen_text = output[0]["generated_text"]
                processed_text = process_generated_text(prompt, gen_text)
                
                # Get or create record
                if id_val not in records_by_id:
                    records_by_id[id_val] = {
                        "id": id_val,
                        GOOD_COL: good_code,
                        BAD_COL: bad_code,
                    }
                
                # Add prompt to record
                records_by_id[id_val][f"{kind}_prompt"] = processed_text
                
                # Write complete records immediately and remove from memory
                if all(k in records_by_id[id_val] for k in ["good_prompt", "bad_prompt"]):
                    f.write(json.dumps(records_by_id[id_val], ensure_ascii=False) + "\n")
                    del records_by_id[id_val]
            
            # Clear memory after each batch
            torch.cuda.empty_cache()
            gc.collect()
            
            logger.info(f"Processed {min(i + BATCH_SIZE, len(dataset))}/{len(dataset)} examples")
    
    logger.info(f"✅ Finished – wrote output to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()