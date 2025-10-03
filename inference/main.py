import os
from pathlib import Path
from pipeline import SentimentInferencePipeline

MODEL_DIR = Path("output")
BASE_DIR = Path("")
TEST_DATA_DIR = BASE_DIR / "data"

MODEL_CONFIGS = {
    "deepseek": {
        "model_parent_dir": MODEL_DIR / "deepseek/sft-models",
        "output_template": str(MODEL_DIR / "deepseek/results/{dataset}")
    },
    "llama": {
        "model_parent_dir": MODEL_DIR / "llama3/sft-models",
        "output_template": str(MODEL_DIR / "llama3/results-test/{dataset}")
    },
    "qwen": {
        "model_parent_dir": MODEL_DIR / "qwen3/sft-models",
        "output_template": str(MODEL_DIR / "qwen3/results/{dataset}")
    }
}

DATASETS = {
    "fpb": {
        "name": "FinancialPhraseBank",
        "test_path": TEST_DATA_DIR / "FinancialPhraseBank-v1.0/test_data.json"
    },
    "fiqa": {
        "name": "FIQA",
        "test_path": TEST_DATA_DIR / "fiqa-sentiment-data-with-scores/test_data.json"
    },
    "gsd": {
        "name": "GoldSentiment",
        "test_path": TEST_DATA_DIR / "gold-sentiment/test_data.json"
    },
    "tsd": {
        "name": "TwitterSentiment",
        "test_path": TEST_DATA_DIR / "twitter-sentiment-data/test_data.json"
    },
    "csd": {
        "name": "ChineseSentiment",
        "test_path": TEST_DATA_DIR / "chinese-finance-data/test_data.json"
    }
}

def run_pipelines(models=None, datasets=None, mode = "shots", num_shots = 0):
    models_to_run = models or MODEL_CONFIGS.keys()
    datasets_to_run = datasets or DATASETS.keys()
    
    for model_type in models_to_run:
        if model_type not in MODEL_CONFIGS:
            print(f"Warning: Model type '{model_type}' not found in configurations")
            continue
            
        model_config = MODEL_CONFIGS[model_type]
        
        for dataset_key in datasets_to_run:
            if dataset_key not in DATASETS:
                print(f"Warning: Dataset '{dataset_key}' not found in configurations")
                continue
                
            dataset = DATASETS[dataset_key]
            
            output_dir = Path(model_config["output_template"].format(dataset=dataset_key))
            
            print(f"\nRunning {model_type.upper()} model on {dataset['name']} dataset...")
            print(f"Model directory: {model_config['model_parent_dir']}")
            print(f"Output directory: {output_dir}")
            print(f"Test data path: {dataset['test_path']}")
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            pipeline = SentimentInferencePipeline(
                model_parent_dir=model_config["model_parent_dir"],
                output_parent_dir=output_dir,
                test_data_path=dataset["test_path"],
                model_type=model_type,
                mode = mode,
                num_shots = num_shots
            )
            
            pipeline.run_all_splits()

if __name__ == "__main__":
    # Run all models on all datasets
    run_pipelines(mode = "shots", num_shots = 0) # Inference with shot learning
    # run_pipelines(mode = "sft") # Inference of the finetuned models
    
    # Or run specific models on specific datasets, some examples:
    # run_pipelines(models=["llama"], datasets=["fpb", "fiqa"], mode = "sft")
    # run_pipelines(models=["deepseek"], datasets=["fpb", "fiqa", "gsd", "tsd", "csd"], mode = "shots", num_shots = 0)
    # run_pipelines(models=["llama", "qwen"], datasets=["fpb", "fiqa", "gsd", "tsd", "csd"], mode = "shots", num_shots = 3)
    # run_pipelines(models=["qwen"], datasets=["fpb"], mode = "sft")