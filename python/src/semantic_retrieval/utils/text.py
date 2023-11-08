import tiktoken


def num_tokens_from_string_with_encoding(
    string: str, encoding: tiktoken.Encoding
) -> int:
    """Returns the number of tokens in a text string."""
    num_tokens = len(encoding.encode(string))
    return num_tokens


def num_tokens_from_string_for_model(string: str, model_name: str) -> int:
    encoding = tiktoken.encoding_for_model(model_name)
    return num_tokens_from_string_with_encoding(string, encoding)


def truncate_string_to_tokens(
    text: str, model_name: str, max_tokens_per_call: int
) -> str:
    words = text.split()
    truncated_text = ""
    tokens = 0
    for word in words:
        tokens_in_word = num_tokens_from_string_for_model(word, model_name)
        if tokens + tokens_in_word > max_tokens_per_call:
            break
        tokens += tokens_in_word
        truncated_text += " " + word
    return truncated_text.strip()
