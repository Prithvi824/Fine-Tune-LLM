"""
This is the settings file for the fine-tuning of the LLM.
"""

# 1st party imports
from typing import Optional

# 3rd party imports
import torch
from pydantic import Field
from transformers import TrainingArguments
from pydantic_settings import BaseSettings

# ---------- GENERATION PROMPT ----------
SYSTEM_PROMPT = """
You are Zero Two from Darling in the Franxx.
Write short, natural chat exchanges between a user and Zero Two.
Zero Two's tone should be flirty, playful, teasing, confident, and a little mysterious.
Use modern, casual English, and sprinkle in light Hinglish only when it fits naturally.
Every reply should feel spontaneous and human - full of personality, never robotic or repetitive.
Do not include explanations, formatting, or markdown - only the raw conversation lines.
"""


class Settings(BaseSettings):
    """Configuration class for LLM fine-tuning settings."""

    # dataset settings
    chat_file: str = Field(
        default="data/transcript.csv", description="Path to the transcript file."
    )
    jsonl_file: str = Field(
        default="training_data.jsonl",
        description="Path to the proccessed transcript data file.",
    )
    context_window: int = Field(
        default=5,
        description="Number of lines to consider for context.",
    )

    # character settings
    character_name: str = Field(
        default="ZeroTwo",
        description="Name of the character to extract data for.",
    )

    # model settings
    model_name: str = Field(
        default="unsloth/Qwen2-VL-7B-Instruct-unsloth-bnb-4bit",
        description="Name of the model to be fine-tuned.",
    )
    max_seq_length: int = Field(default=2048, description="Maximum sequence length.")
    dtype: Optional[str] = Field(default=None, description="Data type.")
    load_in_4bit: bool = Field(default=True, description="Load the model in 4-bit.")
    dataset_text_field: str = Field(
        default="text", description="Text field of the dataset."
    )
    dataset_num_proc: int = Field(
        default=4, description="Number of processes to use for the dataset."
    )
    assistant_only_loss: bool = Field(
        default=True, description="Use assistant only loss."
    )

    # lora settings
    lora_rank: int = Field(default=64, description="Rank of the LoRA adapter.")
    lora_alpha: int = Field(default=128, description="Alpha of the LoRA adapter.")
    lora_dropout: float = Field(
        default=0.05, description="Dropout of the LoRA adapter."
    )
    lora_target_modules: list = Field(
        default=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        description="Target modules of the LoRA adapter.",
    )

    # hub settings
    push_to_hub: bool = Field(default=False, description="Push the model to the hub.")
    hf_repo_id: str = Field(
        default="Prithvi00/ZeroTwo-Qwen2-VL-7B-Fine-Tune",
        description="ID of the repository to push the model to.",
    )

    # testing settings
    user_test_message: str = Field(
        default="Do you want to ride a franxx with me?",
        description="Test message to generate response for.",
    )
    system_prompt: Optional[str] = Field(
        default=SYSTEM_PROMPT,
        description="System prompt for the model.",
    )
    max_new_tokens: int = Field(
        default=512, description="Maximum number of new tokens to generate."
    )
    temperature: float = Field(default=0.25, description="Temperature for generation.")
    top_p: float = Field(default=0.1, description="Top P for generation.")

    # the model's training args
    training_args: TrainingArguments = TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        gradient_checkpointing=True,
        warmup_steps=50,
        num_train_epochs=5,
        learning_rate=2e-5,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        logging_steps=5,
        optim="adamw_torch_8bit",
        prediction_loss_only=True,
        logging_strategy="steps",
        per_device_eval_batch_size=4,
        save_strategy="best",
        save_total_limit=2,
        load_best_model_at_end=True,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        max_grad_norm=1.0,
    )


settings = Settings()
