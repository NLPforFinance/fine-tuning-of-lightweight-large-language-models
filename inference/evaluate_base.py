import json
import jsonlines
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from tqdm import tqdm

class Evaluator:
    def __init__(self, config, predictor, model_loader, label_parser):
        self.config = config
        self.predictor = predictor
        self.model_loader = model_loader
        self.label_parser = label_parser

    def evaluate_base_model(self, batch_size=64):
        base_model_name = self.config.base_models[self.config.model_type]
        print(f"\nEvaluating base model: {base_model_name} with batch_size={batch_size}...")

        results_dir = self.config.output_parent_dir / f"results-{self.config.model_type}"
        results_dir.mkdir(exist_ok=True)

        with open(self.config.test_data_path) as f:
            test_data = json.load(f)

        model, tokenizer = self.model_loader.load_model(base_model_name)

        results, y_true, y_pred = [], [], []

        valid_samples = []
        for item in test_data:
            true_label = self.label_parser.extract_label(item['output'])
            if true_label:
                valid_samples.append({
                    'instruction': item['instruction'],
                    'input': item['input'],
                    'true_label': true_label
                })

        for i in tqdm(range(0, len(valid_samples), batch_size),
                      desc=f"Evaluating base {self.config.model_type}"):
            batch = valid_samples[i:i+batch_size]
            batch_instructions = [item['instruction'] for item in batch]
            batch_inputs = [item['input'] for item in batch]

            batch_preds = self.predictor.predict_batch(
                model,
                tokenizer,
                batch_instructions,
                batch_inputs
            )

            for item, pred_label in zip(batch, batch_preds):
                results.append({
                    "input": item['input'],
                    "instruction": item['instruction'],
                    "true_label": item['true_label'],
                    "predicted_label": pred_label or "unknown"
                })

                if pred_label:
                    y_true.append(item['true_label'])
                    y_pred.append(pred_label)

        self._save_results(results_dir, results, y_true, y_pred)

    def _save_results(self, results_dir, results, y_true, y_pred):
        with jsonlines.open(results_dir / "predictions.jsonl", 'w') as writer:
            writer.write_all(results)
        
        if y_true and y_pred:
            unique_classes = sorted(set(y_true + y_pred))
            
            class_mapping = {
                '0': 'negative',
                '1': 'positive', 
                '2': 'neutral',
                'positive': 'positive',
                'negative': 'negative',
                'neutral': 'neutral'
            }
            
            target_names = [class_mapping.get(str(cls), f"class_{cls}") for cls in unique_classes]
            
            metrics = {
                "accuracy": accuracy_score(y_true, y_pred),
                "precision_micro": precision_score(y_true, y_pred, average='micro'),
                "precision_macro": precision_score(y_true, y_pred, average='macro'),
                "precision_weighted": precision_score(y_true, y_pred, average='weighted'),
                "recall_micro": recall_score(y_true, y_pred, average='micro'),
                "recall_macro": recall_score(y_true, y_pred, average='macro'),
                "recall_weighted": recall_score(y_true, y_pred, average='weighted'),
                "f1_micro": f1_score(y_true, y_pred, average='micro'),
                "f1_macro": f1_score(y_true, y_pred, average='macro'),
                "f1_weighted": f1_score(y_true, y_pred, average='weighted'),
                "classification_report": classification_report(
                    y_true, y_pred,
                    target_names=target_names,
                    output_dict=True
                )
            }
            
            with open(results_dir / "metrics.json", 'w') as f:
                json.dump(metrics, f, indent=2)
            
            print(f"\nMetrics:")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"F1 Macro: {metrics['f1_macro']:.4f}")
    
    def run_all_splits(self):
        self.evaluate_base_model()
