<div align="center">

# smolmodels ✨

[![PyPI version](https://img.shields.io/pypi/v/smolmodels.svg)](https://pypi.org/project/smolmodels/)
[![Discord](https://img.shields.io/discord/1300920499886358529?logo=discord&logoColor=white)](https://discord.gg/SefZDepGMv)

<img src="resources/backed-by-yc.png" alt="backed-by-yc" width="20%">


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
    input_schema={"headline": str, "content": str},         # [optional - can be pydantic or dict]
    output_schema={"sentiment": str}                        # [optional - can be pydantic or dict]
)

# Step 2: build and train the model on data
model.build(
   datasets=[dataset, auxiliary_dataset],
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
**intent**. The schemas can be defined either using `pydantic` models, or plain dictionaries that are convertible to
`pydantic` models.

```python
# This defines the model's identity
model = sm.Model(
    intent="Predict sentiment on a news article such that [...]",
    input_schema={"headline": str, "content": str},                 # supported: pydantic or dict
    output_schema={"sentiment": str}                                # supported: pydantic or dict
)
```

You describe the model's expected behaviour in plain English. The library will select a metric to optimise for, 
and produce logic for feature engineering, model training, evaluation, and so on.

### 2.2. 🎯 Model Building
The model is built by calling `model.build()`. This method takes one or more datasets and 
generates a set of possible model solutions, training and evaluating them to select
the best one. The model with the highest performance metric becomes the "implementation" of the predictor.

You can specify the model building cutoff in terms of a timeout, a maximum number of solutions to explore, or both.

```python
model.build(
    datasets=[dataset_a, dataset_b],
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
want to augment existing data. You can do this with the `sm.DatasetGenerator` class:

```python
dataset = sm.DatasetGenerator(
    schema={"headline": str, "content": str, "sentiment": str},  # supported: pydantic or dict
    data=existing_data
)
dataset.generate(1000)

model.build(
    datasets=[dataset],
    ...
)
```

> [!CAUTION]
> Data generation can consume a lot of tokens. Start with a conservative `generate_samples` value and
> increase it if needed.

The library can also infer the input and/or output schema of your predictor, if required. This is based either on the
dataset you provide, or on the model's intent. This can be useful when you don't know what the model should look like.
As with the models, you can specify the schema using `pydantic` models or plain dictionaries.

```python
# In this case, the library will infer a schema from the intent and generate data for you
model = sm.Model(intent="Predict sentiment on a news article such that [...]")
model.build(provider="openai/gpt-4o-mini")
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
### 3.1. Installation Options

`smolmodels` offers different installation options to suit your needs:

```bash
pip install smolmodels                  # default installation without deep learning dependencies
pip install smolmodels[lightweight]     # explicitly lightweight, equivalent to 'smolmodels'

pip install smolmodels[all]             # full installation including deep learning dependencies
pip install smolmodels[deep-learning]   # specifically for deep-learning, equivalent to 'smolmodels[all]'
```

The lightweight installation is suitable for most machine learning tasks and reduces the installation size 
significantly. If you need deep learning capabilities, use the `[all]` or `[deep-learning]` option.

### 3.2. API Keys Setup
Set your API key as an environment variable based on which provider you want to use:

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

## 8. Docker Deployment

Run smolmodels as a platform with a RESTful API and web UI using Docker:

```bash
git clone https://github.com/plexe-ai/smolmodels.git
cd smolmodels/docker
cp .env.example .env  # Edit with your LLM provider API key
docker-compose up -d
```

Access your deployment:
- API: http://localhost:8000
- Web UI: http://localhost:8501 

The web interface provides an easy way to create models, view their status, and make predictions without writing code. See the [Docker README](docker/README.md) for more details.

## 9. Product Roadmap

- [X] Fine-tuning and transfer learning for small pre-trained models
- [X] Use Pydantic for schemas and split data generation into a separate module
- [X] Smolmodels self-hosted platform ⭐ (More details coming soon!)
- [X] Lightweight installation option without heavy deep learning dependencies
- [ ] Support for non-tabular data types in model generation
- [ ] File upload to docker containers