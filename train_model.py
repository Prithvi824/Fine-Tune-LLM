"""
This is the main file for the fine-tuning of the LLM.
"""

# NOTE: Unsloth should be the first 3rd party module imported
import unsloth

# 3rd party imports
from colorama import init, Fore

# local imports
from settings import settings
from utils import _load_data, _load_model, _test_model_generation, _get_trainer

if __name__ == "__main__":

    # setup colorma
    init(autoreset=True)

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
        settings.model_name,
        settings.max_seq_length,
        settings.dtype,
        settings.load_in_4bit,
    )
    print(Fore.GREEN + "Model loaded successfully..", end="\n")

    # get the dataset
    print(Fore.BLUE + "Loading the dataset from ", settings.jsonl_file, end="\n")
    dataset = _load_data(tokenizer, settings.jsonl_file)
    print(Fore.GREEN + "Dataset loaded successfully..", end="\n")

    # test the model generation before training
    print(Fore.BLUE + "Testing the model generation before training..", end="\n")
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

    # switch the model evaluation mode to training mode
    print(Fore.BLUE + "Switching the model to training mode..", end="\n")
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    print(Fore.GREEN + "Model switched to training mode successfully..", end="\n")

    # get the trainer
    print(Fore.BLUE + "Creating the trainer..", end="\n")
    print(Fore.BLUE + "Check settings for training arguments: ", end="\n")
    trainer = _get_trainer(
        model,
        tokenizer,
        dataset,
        settings.lora_rank,
        settings.lora_target_modules,
        settings.lora_alpha,
        settings.lora_dropout,
        settings.training_args,
        settings.dataset_text_field,
        settings.dataset_num_proc,
        settings.assistant_only_loss,
    )
    print(Fore.GREEN + "Trainer created successfully..", end="\n")

    # train the model
    print(
        Fore.BLUE + "Training the model. You can stop training using",
        Fore.RED + "CTRL + C",
        end="\n",
    )

    trainer_stats = trainer.train()
    print(Fore.GREEN + "Model trained successfully..", end="\n")

    # test the model generation after training
    print(Fore.BLUE + "Testing the model generation after training..", end="\n")
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
    print(Fore.GREEN + "Model tested successfully..", end="\n")
    print(Fore.GREEN + "Model response: ", model_res, end="\n")

    # choose the upload mode
    print(Fore.BLUE + "Choose the upload mode: ", end="\n")
    print(Fore.BLUE + "1. Upload to Hugging Face", end="\n")
    print(Fore.BLUE + "2. Upload to Local at path: ./new_model", end="\n")
    upload_mode = input("Enter the upload mode (Invalid input will default to Local) :")

    # upload the model
    if upload_mode == "1":
        model.push_to_hub(settings.hf_repo_id)
        upload_mode = "Hugging Face"
    else:
        model.save_pretrained("./new_model")
        tokenizer.save_pretrained("./new_model")
        upload_mode = "Local"

    # print the upload mode
    print(Fore.GREEN + "Model uploaded successfully..", end="\n")
    print(Fore.GREEN + "Upload mode: ", upload_mode, end="\n")
