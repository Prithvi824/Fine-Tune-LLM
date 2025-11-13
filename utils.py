"""
This file contains the utils for the fine-tuning of the LLM.
"""

# 1st party imports
from typing import Tuple, Optional, Dict

# NOTE: this should be the first 3rd party module
from unsloth import FastLanguageModel

# 3rd party imports
from trl import SFTTrainer
from datasets import load_dataset, Dataset
from transformers import TrainingArguments


def _load_data(tokenizer: FastLanguageModel, file_path: str):
    """
    This function loads the data.
    ### NOTE: This is the second step in the training process.

    Args:
        tokenizer (FastLanguageModel): The tokenizer to use.
        file_path (str): The path to the file to load.

    Returns:
        data: The loaded data.
    """

    def formatting_prompts_func(example):
        """
        Formats the prompts for fine-tuning the model.
        """
        prompt = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": prompt}

    return load_dataset("json", data_files=file_path).map(formatting_prompts_func)


def _load_model(
    name: str,
    max_seq_length: int,
    dtype: str,
    load_in_4bit: bool,
    **kwargs: Optional[Dict],
) -> Tuple[FastLanguageModel, FastLanguageModel]:
    """
    This function loads the model and tokenizer.
    ### NOTE: This is the first step in the training process.

    Args:
        name (str): The name of the model.
        max_seq_length (int): The maximum sequence length.
        dtype (str): The data type.
        load_in_4bit (bool): Whether to load the model in 4bit.
        **kwargs (Optional[Dict]): Additional keyword arguments.

    Returns:
        model: The loaded model.
        tokenizer: The loaded tokenizer.
    """
    return FastLanguageModel.from_pretrained(
        model_name=name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        **kwargs,
    )


def _create_test_message(test_message: str, system_prompt: Optional[str] = None) -> str:
    """
    This function creates a test message.

    Returns:
        test_message (str): The test message.
        system_prompt (Optional[str]): The system prompt.
    """

    # create user prompt
    user_prompt = {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": test_message,
            }
        ],
    }

    # create system prompt if it exists
    if system_prompt:
        system_prompt = {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": system_prompt,
                }
            ],
        }
        return {"messages": [system_prompt, user_prompt]}

    # return user prompt if system prompt does not exist
    else:
        return {"messages": [user_prompt]}


def _test_model_generation(
    test_message: str,
    model: FastLanguageModel,
    tokenizer: FastLanguageModel,
    max_new_tokens: int = 512,
    temperature: float = 0.25,
    top_p: float = 0.1,
    system_prompt: Optional[str] = None,
) -> str:
    """
    This function tests the model generation.

    This is the second step in the training process.

    Args:
        test_message (str): The test message.
        model (FastLanguageModel): The model to test.
        tokenizer (FastLanguageModel): The tokenizer to use.
        max_new_tokens (int): The maximum number of new tokens to generate.

    Returns:
        response (str): The generated response.
    """

    # Step 1: switch the model for inference (eval mode)
    FastLanguageModel.for_inference(model)

    # create the test message in message template
    test_message = _create_test_message(test_message, system_prompt)

    # Step 2: apply chat template
    test_prompt = tokenizer.apply_chat_template(
        test_message["messages"],
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    # Step 1.3: Generate response
    test_outputs = model.generate(
        input_ids=test_prompt,
        attention_mask=(test_prompt != tokenizer.pad_token_id).long(),
        max_new_tokens=max_new_tokens,
        use_cache=True,
        temperature=temperature,
        do_sample=True,
        top_p=top_p,
    )

    # Step 1.4: Decode and return the response
    model_response = tokenizer.batch_decode(test_outputs, skip_special_tokens=True)[0]
    return model_response


def _get_trainer(
    model: FastLanguageModel,
    tokenizer: FastLanguageModel,
    training_dataset: Dataset,
    lora_rank: int,
    lora_target_modules: list,
    lora_alpha: int,
    lora_dropout: float,
    training_args: TrainingArguments,
    dataset_text_field: str,
    dataset_num_proc: int,
    assistant_only_loss: bool,
):
    """
    This function returns the trainer for the given model based on specific settings.

    This is the third step in the training process.

    Args:
        model (FastLanguageModel): The model to train.
        tokenizer (FastLanguageModel): The tokenizer to use.
        training_dataset (Dataset): The training dataset.
        lora_rank (int): The rank of the LoRA adapter.
        lora_target_modules (list): The target modules for the LoRA adapter.
        lora_alpha (int): The alpha value for the LoRA adapter.
        lora_dropout (float): The dropout value for the LoRA adapter.
        training_args (TrainingArguments): The training arguments.
        dataset_text_field (str): The text field of the dataset.
        dataset_num_proc (int): The number of processes to use for the dataset.
        assistant_only_loss (bool): Whether to use assistant only loss.

    Returns:
        trainer: The trainer.
    """

    # Step 1: Add LoRA adapters to the model
    model = FastLanguageModel.get_peft_model(
        model=model,
        r=lora_rank,
        target_modules=lora_target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_rslora=True,
        loftq_config=None,
    )

    # Step 2: create a SFTTrainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=training_dataset["train"],
        dataset_text_field=dataset_text_field,
        dataset_num_proc=dataset_num_proc,
        assistant_only_loss=assistant_only_loss,
        report_to="none"
    )

    # Step 3: Return the trainer
    return trainer
