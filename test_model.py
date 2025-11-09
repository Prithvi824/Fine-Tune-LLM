"""
This is the test file for the fine-tuning of the LLM.
"""

# NOTE: Unsloth should be the first 3rd party module imported
import unsloth

# 3rd party imports
from colorama import Fore

# local imports
from settings import settings
from utils import _load_model, _test_model_generation

if __name__ == "__main__":

    # the testing parameters
    test_params = {
        "test_message": settings.user_test_message,
        "max_new_tokens": settings.max_new_tokens,
        "temperature": settings.temperature,
        "top_p": settings.top_p,
    }

    # load the model
    print(Fore.BLUE + "Loading the model..", end="\n")
    model, tokenizer = _load_model(
        settings.hf_repo_id,
        settings.max_seq_length,
        settings.dtype,
        settings.load_in_4bit,
    )
    print(Fore.GREEN + "Model loaded successfully..", end="\n")

    # test the model generation before training
    print(Fore.BLUE + "Testing the model.", end="\n")
    print(Fore.BLUE + "Test message: ", settings.user_test_message, end="\n")
    model_res = _test_model_generation(
        settings.user_test_message,
        model,
        tokenizer,
        settings.max_new_tokens,
        settings.temperature,
        settings.top_p,
        system_prompt=settings.system_prompt,
    )
    print(Fore.CYAN + "Model response: ", model_res, end="\n")
