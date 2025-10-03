from transformers import Trainer
import torch

class DomainBalancedTrainer(Trainer):
    def __init__(self, domain_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.domain_weights = domain_weights
        self.domain_cache = {} 
        
    def _cache_domains(self, dataset):
        for i, example in enumerate(dataset):
            self.domain_cache[i] = example['domain']
            
    def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
        input_ids = inputs["input_ids"]
        
        domains = []
        for ids in input_ids:
            key = tuple(ids.cpu().numpy())
            if key in self.domain_cache:
                domains.append(self.domain_cache[key])
            else:
                domains.append(None)

        
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"]
        )
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        shifted_labels = inputs["labels"][..., 1:].contiguous()
        shifted_logits = outputs.logits[..., :-1, :].contiguous()
        
        losses = loss_fct(shifted_logits.view(-1, shifted_logits.size(-1)), 
                        shifted_labels.view(-1))
        losses = losses.view(shifted_labels.size())
        
        weighted_losses = torch.zeros_like(losses)
        for i, domain in enumerate(domains):

            if domain is not None:
                weighted_losses[i] = losses[i] * self.domain_weights.get(domain, 1.0)
            else:
                weighted_losses[i] = losses[i]  

        loss = weighted_losses.mean()
        return (loss, outputs) if return_outputs else loss