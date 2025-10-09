class Config:
    BASE_MODEL = "Qwen/Qwen3-8B"
    DATASET_PARENT_PATHS = [
        "data/FinancialPhraseBank-v1.0/scenarios_train",
        "data/fiqa-sentiment-data-with-scores/scenarios_train",
        "data/gold-sentiment/scenarios_train",
        "data/twitter-sentiment-data/scenarios_train",
        "data/chinese-finance-data/scenarios_train"
    ]
    TARGET_SIZES = {
        'FinancialPhraseBank-v1.0': 5000,  # FPB: original 4000
        'fiqa-sentiment-data-with-scores': 5000,  # FiQA: original 1200
        'gold-sentiment': 5000,  # GSD: original 10000
        'twitter-sentiment-data': 5000,  # TSD: original 12000
        'chinese-finance-data': 5000 # CSD: original 13700
    }
    
    PROPORTIONS = [5, 10, 20, 40, 75, 100]
    OUTPUT_ROOT = "output/qwen3/sft-models"
    
    LORA_R = 8
    LORA_ALPHA = 32
    LORA_DROPOUT = 0.05
    TARGET_MODULES = ["q_proj", "o_proj", "gate_proj", "down_proj", "k_proj", "v_proj", "up_proj"] 
    MAX_LENGTH = 256
    MIN_LENGTH = 10
    MAX_SAMPLES = 150000
    BATCH_SIZE = 16
    GRAD_ACCUM = 2
    EPOCHS = 5
    LEARNING_RATE = 2.0e-5
    LR_SCHEDULER = "cosine"
    WARMUP_RATIO = 0.03
    BF16 = True
    LOGGING_STEPS = 100
    SAVE_STEPS = 1000
