from .llama import biLlamaForMaskedLM, biLlamaForMaskedNTP


MODELS_MAPPING = {
    'llama': {
        'mlm': biLlamaForMaskedLM,
        'mntp': biLlamaForMaskedNTP
    }
}