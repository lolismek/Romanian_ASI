"""
Script to explore the FULG dataset structure from HuggingFace.
https://huggingface.co/datasets/faur-ai/fulg
"""

from datasets import load_dataset
import json

def main():
    print("=" * 60)
    print("FULG Dataset Explorer")
    print("=" * 60)

    # First, try to get dataset info without streaming to see metadata
    print("\n1. Loading dataset info (metadata)...")
    try:
        # Load with streaming to avoid downloading everything
        ds_stream = load_dataset("faur-ai/fulg", split="train", streaming=True)

        # Get dataset info
        print(f"\nDataset info:")
        print(f"  - Features: {ds_stream.features}")
        print(f"  - Description: {getattr(ds_stream, 'description', 'N/A')}")

    except Exception as e:
        print(f"Error loading dataset info: {e}")
        return

    # Get a sample to see structure
    print("\n2. Sample record structure:")
    print("-" * 40)

    sample = next(iter(ds_stream))

    for key, value in sample.items():
        value_preview = str(value)[:200] + "..." if len(str(value)) > 200 else str(value)
        print(f"\n  Field: '{key}'")
        print(f"    Type: {type(value).__name__}")
        print(f"    Value preview: {value_preview}")

    # Show a few more samples to understand variety
    print("\n3. First 5 samples (text field preview):")
    print("-" * 40)

    ds_stream = load_dataset("faur-ai/fulg", split="train", streaming=True)
    for i, item in enumerate(ds_stream.take(5)):
        text_field = None
        # Try common field names
        for field in ['text', 'content', 'document', 'raw_content']:
            if field in item:
                text_field = item[field]
                break

        if text_field is None:
            text_field = str(list(item.values())[0])

        preview = text_field[:150].replace('\n', ' ') + "..." if len(text_field) > 150 else text_field
        print(f"\n  [{i+1}] {preview}")

    # Try to get row count from dataset info
    print("\n4. Dataset size information:")
    print("-" * 40)

    try:
        # Try loading dataset info directly
        from datasets import get_dataset_infos, get_dataset_config_names

        configs = get_dataset_config_names("faur-ai/fulg")
        print(f"  Available configs: {configs}")

        infos = get_dataset_infos("faur-ai/fulg")
        for config_name, info in infos.items():
            print(f"\n  Config '{config_name}':")
            if hasattr(info, 'splits') and info.splits:
                for split_name, split_info in info.splits.items():
                    print(f"    - {split_name}: {split_info.num_examples:,} rows")
            if hasattr(info, 'download_size') and info.download_size:
                print(f"    - Download size: {info.download_size / (1024**3):.2f} GB")
            if hasattr(info, 'dataset_size') and info.dataset_size:
                print(f"    - Dataset size: {info.dataset_size / (1024**3):.2f} GB")

    except Exception as e:
        print(f"  Could not get detailed info: {e}")
        print("  Attempting to count rows (this may take a while for large datasets)...")

        # If metadata not available, we could count but that's slow
        # Let's just estimate from a sample
        print("\n  Note: For very large datasets, row count may need to be")
        print("  fetched from the HuggingFace website directly.")

    print("\n" + "=" * 60)
    print("Exploration complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
