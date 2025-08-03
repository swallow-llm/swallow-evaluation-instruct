from argparse import ArgumentParser
from transformers import AutoConfig, AutoTokenizer


def get_max_model_length(config: AutoConfig, limit_model_length: int=32768) -> int:
    for key in (
        "max_position_embeddings",
        "model_max_length",
        "context_length",
        "seq_length"
    ):
        val = getattr(config, key, None)
        if isinstance(val, int) and val > 0:
            return min(limit_model_length, val)
    return limit_model_length


def check_think_tag(vocab: dict) -> str:
    think_tags = [
        ## Add think tags here
        "<think>", "</think>",
    ]
    return str(any(vocab.get(tag) is not None for tag in think_tags)).lower()


def check_metadata_shell(model_id: str):
    # This function is supposed to be used in shell script. 
    # So the output must be printed as standard output in the following order:
    # - MAX_MODEL_LENGTH: max model length
    # - HAS_THINK_TAG: whether the model has think tags

    # Get metadata
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    vocab = tokenizer.get_vocab()

    # Check metadata
    max_model_length = get_max_model_length(cfg)
    has_think_tag = check_think_tag(vocab)

    # Print metadata
    print(max_model_length)
    print(has_think_tag)
    

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    args = parser.parse_args()

    check_metadata_shell(args.model_id)
