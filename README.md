# AI/ML Fine-Tuning Project - ZeroTwo Character Model

A comprehensive LLM fine-tuning project that creates a ZeroTwo character chatbot using Qwen2-VL-7B model with LoRA (Low-Rank Adaptation) techniques.

## 🎯 Project Overview

This project fine-tunes the Qwen2-VL-7B-Instruct model to create an AI assistant that embodies the personality of ZeroTwo from "Darling in the Franxx" anime. The model is trained to respond in a flirty, human-like manner while maintaining the character's emotional depth.

## 📁 Project Structure

```
AI_ML/
├── cleaner/                    # Data processing utilities
│   ├── __init__.py
│   ├── extractor.py           # Data extraction logic
│   └── models.py              # Pydantic models for data validation
├── logs/                      # Training and application logs
│   ├── all.log
│   ├── error.log
│   ├── info.log
│   └── warning.log
├── train_model.py             # Main training script
├── test_model.py              # Model testing script
├── utils.py                   # Core utility functions
├── settings.py                # Configuration settings
├── log_config.py              # Logging configuration
├── pyproject.toml             # Project dependencies
└── README.md                  # This file
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- CUDA-compatible GPU (recommended)
- UV package manager (or pip)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd AI_ML
   ```

2. **Install dependencies**
   ```bash
   uv sync
   # or with pip
   pip install -r requirements.txt
   ```

3. **Prepare your training data**
   - Place your conversation data in JSONL format
   - Default path: `training_data.jsonl`
   - Format: Each line should contain a JSON object with "messages" field

## 🔧 Configuration

All settings are managed through `settings.py` using Pydantic. Key configurations include:

### Model Settings
- **Base Model**: `unsloth/Qwen2-VL-7B-Instruct-unsloth-bnb-4bit`
- **Max Sequence Length**: 2048 tokens
- **4-bit Quantization**: Enabled for memory efficiency

### LoRA Configuration
- **Rank**: 64
- **Alpha**: 128
- **Dropout**: 0.05
- **Target Modules**: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`

### Training Parameters
- **Batch Size**: 2 per device
- **Gradient Accumulation**: 4 steps
- **Learning Rate**: 2e-5
- **Epochs**: 5
- **Optimizer**: AdamW 8-bit
- **Precision**: BF16/FP16 (auto-detected)

## 🎓 Training Process

### Training Flow

The training process follows these steps:

1. **Model Loading** (`_load_model`)
   - Loads the pre-trained Qwen2-VL-7B model
   - Applies 4-bit quantization for memory efficiency
   - Initializes tokenizer with chat template support

2. **Data Preparation** (`_load_data`)
   - Loads training data from JSONL file
   - Applies chat template formatting
   - Converts conversations to training format

3. **Pre-training Evaluation**
   - Tests model generation before training
   - Uses configured test message and system prompt
   - Establishes baseline performance

4. **Model Configuration for Training**
   - Enables gradient checkpointing
   - Enables input gradients
   - Switches to training mode

5. **LoRA Adapter Setup** (`_get_trainer`)
   - Adds LoRA adapters to target modules
   - Configures rank, alpha, and dropout parameters
   - Uses RSLoRA for improved performance

6. **Training Execution**
   - Uses SFTTrainer (Supervised Fine-Tuning)
   - Implements assistant-only loss for better alignment
   - Supports gradient accumulation and checkpointing

7. **Post-training Evaluation**
   - Tests model generation after training
   - Compares with pre-training baseline
   - Validates training effectiveness

8. **Model Saving**
   - Option 1: Push to Hugging Face Hub
   - Option 2: Save locally to `./new_model`

### Running Training

```bash
python train_model.py
```

**Training Output:**
- Real-time training progress with colored console output
- Automatic logging to `logs/` directory
- Pre and post-training model comparisons
- Interactive model saving options

## 🧪 Testing

### Test Script Usage

```bash
python test_model.py
```

### Testing Features

- **Model Loading**: Loads fine-tuned model from Hugging Face or local path
- **Generation Testing**: Tests model with configured prompts
- **Parameter Control**: Configurable temperature, top_p, and token limits
- **System Prompt**: Uses ZeroTwo character system prompt

### Test Configuration

```python
# Default test settings
user_test_message = "Do you want to ride a franxx zero two?"
system_prompt = "You are a friendly anime character ZeroTwo..."
max_new_tokens = 512
temperature = 0.25
top_p = 0.1
```

## 📊 Model Architecture

### Base Model: Qwen2-VL-7B-Instruct
- **Parameters**: 7 billion
- **Architecture**: Vision-Language model
- **Quantization**: 4-bit for efficiency
- **Context Length**: 2048 tokens

### LoRA Adaptation
- **Technique**: Low-Rank Adaptation
- **Benefits**: 
  - Reduced memory usage
  - Faster training
  - Preserves base model knowledge
  - Easy model switching

### Training Strategy
- **Method**: Supervised Fine-Tuning (SFT)
- **Loss**: Assistant-only loss (focuses on response quality)
- **Optimization**: AdamW with 8-bit precision
- **Regularization**: Gradient clipping, dropout

## 🎯 Character Personality

The model is trained to embody ZeroTwo's personality:

- **Flirty and playful** communication style
- **Emotional depth** - can express sadness, anger, frustration
- **Human-like responses** that feel natural
- **Context-aware** reactions based on user tone
- **Anime character authenticity** from Darling in the Franxx

## 📈 Performance Monitoring

### Logging System
- **All logs**: `logs/all.log`
- **Error logs**: `logs/error.log` 
- **Info logs**: `logs/info.log`
- **Warning logs**: `logs/warning.log`

### Training Metrics
- Loss tracking every 5 steps
- Gradient norm monitoring
- Learning rate scheduling
- Best model checkpointing

## 🔧 Customization

### Modifying Character
1. Update `character_name` in settings
2. Modify `system_prompt` for personality
3. Adjust `user_test_message` for testing
4. Update training data accordingly

### Training Parameters
```python
# In settings.py
training_args = TrainingArguments(
    per_device_train_batch_size=2,    # Adjust based on GPU memory
    num_train_epochs=5,               # Increase for more training
    learning_rate=2e-5,               # Fine-tune learning rate
    warmup_steps=50,                  # Adjust warmup period
    # ... other parameters
)
```

### LoRA Configuration
```python
# Adjust LoRA parameters
lora_rank = 64          # Higher rank = more parameters
lora_alpha = 128        # Scaling factor
lora_dropout = 0.05     # Regularization
```

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `per_device_train_batch_size`
   - Increase `gradient_accumulation_steps`
   - Enable gradient checkpointing

2. **Slow Training**
   - Ensure CUDA is available
   - Check GPU utilization
   - Adjust `dataloader_num_workers`

3. **Poor Model Performance**
   - Increase training epochs
   - Adjust learning rate
   - Improve training data quality

### Memory Optimization
- 4-bit quantization enabled by default
- Gradient checkpointing for memory efficiency
- Pin memory for faster data loading
- 8-bit optimizer for reduced memory usage

## 📋 Dependencies

Core dependencies (see `pyproject.toml`):
- `unsloth>=2025.11.1` - Efficient LLM training
- `trl>=0.23.0` - Transformer Reinforcement Learning
- `pydantic>=2.12.3` - Data validation
- `pydantic-settings>=2.11.0` - Settings management

Development dependencies:
- `black>=25.9.0` - Code formatting
- `icecream>=2.1.8` - Debugging

**Note**: This project is for educational purposes only. Ensure you have appropriate permissions for any training data used.