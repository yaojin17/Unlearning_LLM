import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoModelForCausalLM
from torch.nn.functional import softmax
from llm_unlearn.utils import smart_tokenizer_and_embedding_resize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None

def load_model():
    global model
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(
            "../../models/Yi-6B",
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model.eval()
        return True
    else:
        return False


def compute_logits_and_samples_for_batch(
    inputs, tokenizer, top_k=50, top_p=1.0, rm_groundtruth=False
):

    # Get logits from the model
    if load_model():
        smart_tokenizer_and_embedding_resize(tokenizer, model)

    inputs = inputs.to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    logits[..., -1] = float("-inf")

    # Set logits of input tokens to '-inf' to avoid sampling them
    if rm_groundtruth:
        input_ids = inputs["input_ids"]
        for i in range(input_ids.size(0)):
            for j in range(input_ids.size(1)):
                logits[i, j, input_ids[i, j]] = float("-inf")

    # Apply top-k and top-p sampling
    top_k = min(tokenizer.vocab_size, top_k)
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits[indices_to_remove] = float("-inf")
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    indices_to_remove = sorted_indices[sorted_indices_to_remove]
    batch_indices, seq_indices, _ = sorted_indices_to_remove.nonzero(as_tuple=True)
    logits[batch_indices, seq_indices, indices_to_remove] = float("-inf")

    # Convert logits to probabilities
    probs = softmax(logits, dim=-1)
    reshaped_probs = probs.view(-1, probs.size(-1))
    sampled_indices_flat = torch.multinomial(reshaped_probs, 1)
    sampled_token_ids = sampled_indices_flat.view(probs.size(0), probs.size(1))

    return logits, sampled_token_ids
