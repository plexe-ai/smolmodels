<div align="center">

# smolmodels ✨

[![PyPI version](https://img.shields.io/pypi/v/smolmodels.svg)](https://pypi.org/project/smolmodels/)
[![Discord](https://img.shields.io/discord/1300920499886358529?logo=discord&logoColor=white)](https://discord.gg/3czW7BMj)
[![PyPI Downloads](https://static.pepy.tech/badge/smolmodels)](https://pepy.tech/projects/smolmodels)

Build machine learning models using natural language and minimal code

[Quickstart](#1-quickstart) |
[Features](#2-features) |
[Installation & Setup](#3-installation--setup) |
[Documentation](#4-documentation) |
[Benchmarks](#5-benchmarks)

<br>

Create machine learning models with minimal code by describing what you want them to do in
plain words. You explain the task, and the library builds a model for you, including data generation, feature 
engineering, training, and packaging.
</div>

> [!NOTE]
> This library is in early development, and we're actively working on new features and improvements! Please report any
> bugs or share your feature requests on [GitHub](https://github.com/plexe-ai/smolmodels/issues) 
> or [Discord](https://discord.gg/3czW7BMj) 💛


## 1. Quickstart
Installation:

```bash
pip install smolmodels
```

Define, train and save a `Model`:

```python
import smolmodels as sm

# Step 1: define the model
model = sm.Model(
    intent="Predict sentiment on a news article such that [...]",
    input_schema={"headline": str, "content": str},                     # [optional]
    output_schema={"sentiment": str}                                    # [optional]
)

# Step 2: build and train the model on data (existing or synthetic)
model.build(
   dataset=dataset,                                                     # [optional]
   generate_samples=1000,                                               # [optional]
   provider="openai/gpt-4o-mini",
   timeout=3600
)

# Step 3: use the model to get predictions on new data
sentiment = model.predict({
   "headline": "600B wiped off NVIDIA market cap",
   "content": "NVIDIA shares fell 38% after [...]",
})

# Step 4: save the model, can be loaded later for reuse
sm.save_model(model, "news-sentiment-predictor")

# Step 5: load a saved model and use it
loaded_model = sm.load_model("news-sentiment-predictor.tar.gz")
```

## 2. Features

`smolmodels` combines graph search, LLM code/data generation and code execution to produce a machine learning model
that meets the criteria of the task description. When you call `model.build()`, the library generates a graph of
possible model solutions, evaluates them, and selects the one that maximises the performance metric for this task.

### 2.1. 💬 Define Models using Natural Language
A model is defined as a transformation from an **input schema** to an **output schema**, which behaves according to an
**intent**.

```python
# This defines the model's identity
model = sm.Model(
    intent="Predict sentiment on a news article such that [...]",
    input_schema={"headline": str, "content": str},
    output_schema={"sentiment": str}
)
```

You describe the model's expected behaviour in plain English. The library will select a metric to optimise for, 
and produce logic for feature engineering, model training, evaluation, and so on.

### 2.2. 🎯 Model Building
The model is built by calling `model.build()`. This method takes a dataset (existing or synthetic) and 
generates a set of possible model solutions, training and evaluating them to select
the best one. The model with the highest performance metric becomes the "implementation" of the predictor.

You can specify the model building cutoff in terms of a timeout, a maximum number of solutions to explore, or both.

```python
model.build(
    dataset=dataset,
    provider="openai/gpt-4o-mini",
    timeout=3600,                       # [optional] max time in seconds
    max_iterations=10                   # [optional] max number of model solutions to explore
)
```

The model can now be used to make predictions, and can be saved or loaded using `sm.save_model()` or `sm.load_model()`.

```python
sentiment = model.predict({"headline": "600B wiped off NVIDIA market cap", ...})
```

### 2.3. 🎲 Data Generation and Schema Inference
The library can generate synthetic data for training and testing. This is useful if you have no data available, or 
want to augment existing data. When building a model, you specify either a dataset, a number of samples to be
generated, or both:

```python
model.build(
    dataset=dataset,                # [optional] -> at least one of these is required
    generate_samples=1000,          # [optional] -> at least one of these is required
    ...
)
```

> [!CAUTION]
> Data generation can consume a lot of tokens. Start with a conservative `generate_samples` value and
> increase it if needed.

The library can also infer the input and/or output schema of your predictor, if required. This is based either on the
dataset you provide, or on the model's intent. This can be useful when you don't know what the model should look like.

```python
# In this case, the library will infer a schema from the intent and generate data for you
model = sm.Model(intent="Predict sentiment on a news article such that [...]")
model.build(generate_samples=100, provider="openai/gpt-4o-mini")
```

> [!TIP]
> If you know how the model will be used, you will get better results by specifying the schema explicitly.
> Schema inference is primarily intended to be used if you don't know what the input/output schema at prediction time
> should be.

### 2.4. 🌐 Multi-Provider Support
You can use multiple LLM providers for model generation. Specify the provider and model in the format `provider/model`:

```python
model.build(provider="openai/gpt-4o-mini", ...)
```

See the section on installation and setup for more details on supported providers and how to configure API keys.

## 3. Installation & Setup
Install the library in the usual manner:

```bash
pip install smolmodels
```

Set your API key as an environment variable based on which provider you want to use. For example:

```bash
# For OpenAI
export OPENAI_API_KEY=<your-API-key>
# For Anthropic
export ANTHROPIC_API_KEY=<your-API-key>
# For Gemini
export GEMINI_API_KEY=<your-API-key>
```

> [!TIP]
> The library uses LiteLLM as its provider abstraction layer. For other supported providers and models,
> check the [LiteLLM](https://docs.litellm.ai/docs/providers) documentation.

## 4. Documentation
For full documentation, visit [docs.plexe.ai](https://docs.plexe.ai).

## 5. Benchmarks
Performance evaluated on 20 OpenML benchmark datasets and 12 Kaggle competitions. Higher performance observed on 12/20
OpenML datasets, with remaining datasets showing performance within 0.005 of baseline. Experiments conducted on standard
infrastructure (8 vCPUs, 30GB RAM) with 1-hour runtime limit per dataset.

Complete code and results are available at [plexe-ai/plexe-results](https://github.com/plexe-ai/plexe-results).

## 6. Contributing

We love contributions! You can get started with [issues](https://github.com/plexe-ai/smolmodels/issues),
submitting a PR with improvements, or joining the [Discord](https://discord.gg/3czW7BMj) to chat with the team. 
See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## 7. License

Apache-2.0 License - see [LICENSE](LICENSE) for details.
