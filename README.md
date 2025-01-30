<div align="center">

# smolmodels ✨

[![PyPI version](https://img.shields.io/pypi/v/smolmodels.svg)](https://pypi.org/project/smolmodels/)
[![Discord](https://img.shields.io/discord/1300920499886358529?logo=discord&logoColor=white)](https://discord.gg/3czW7BMj)

Build specialized ML models using natural language.

</div>

`smolmodels` is a Python library that lets you create machine learning models by describing what you want them to do in
plain English. Instead of wrestling with model architectures and hyperparameters, you simply describe your intent,
define your inputs and outputs, and let `smolmodels` handle the rest.

```python
import pandas as pd
import smolmodels as sm

# Define a house price predictor in terms of inputs, outputs, and expected behaviour
model = sm.Model(
    intent="Predict house prices based on property features",
    input_schema={
        "square_feet": float,
        "bedrooms": int,
        "location": str,
        "year_built": int
    },
    output_schema={
        "predicted_price": float
    }
)

# Build the model, using the backend of your choice; optionally generate synthetic training data
model.build(
   dataset=pd.read_csv("house-prices.csv"),
   generate_samples=1000,
   provider="openai:gpt-4o-mini"
)

# Make predictions
price = model.predict({
    "square_feet": 2500,
    "bedrooms": 4,
    "location": "San Francisco",
    "year_built": 1985
})

# Save the model for later use
sm.save_model(model, "house-price-predictor")
```

## How Does It Work?

`smolmodels` combines graph search with LLMs to generate candidate models that meet the specified intent, and then
selects the best model based on performance and constraints. The process consists of four main phases:

1. **Intent Analysis**: problem description is analyzed to understand the type of model needed and what metric to
   optimise for.

2. **Data Generation**: synthetic data can be generated to enable model build when there is no training data
   available, or when the existing data has insufficient coverage of the feature space.

3. **Model Building**:
   1. Selects appropriate model architectures
   2. Handles feature engineering
   3. Manages training and validation

4. **Validation & Refinement**: the model is tested against constraints and refined using directives (like "optimize 
   for speed" or "prioritize model types with better explainability").

## Key Features

### 📝 Natural Language Intent

Models are defined using natural language descriptions and schema specifications, abstracting away machine learning
specifics.

### 🎲 Data Generation

Built-in synthetic data generation for training and validation.

### 🎯 Directives for fine-grained Control (Not Yet Implemented - Coming Soon)

Guide the model building process with high-level directives:

```python
from smolmodels import Directive

model.build(directives=[
    Directive("Optimize for inference speed"),
    Directive("Prioritize interpretability")
])
```

### ✅ Optional Constraints (Not Yet Implemented - Coming Soon)

Optional declarative constraints for model validation:

```python
from smolmodels import Constraint

# Ensure predictions are always positive
positive_constraint = Constraint(
    lambda inputs, outputs: outputs["predicted_price"] > 0,
    description="Predictions must be positive"
)

model = Model(
    intent="Predict house prices...",
    constraints=[positive_constraint],
    ...
)
```

### 🌐 Multi-Provider Support

You can use multiple LLM providers as a backend for model generation. You can specify the provider and model in the
format `provider:[model]` when calling `build()`:

```python
model.build(pd.read_csv("house-prices.csv"), provider="openai:gpt-4o-mini")
```

Currently supported providers are `openai`, `anthropic`, `google` and `deepseek`. You need to configure the
appropriate API keys for each provider as environment variables (see installation instructions).

## Installation & Setup

```bash
pip install smolmodels
```

## API Keys

Set required API keys as environment variables. Which API keys are required depends on which provider you are using.

```bash
export OPENAI_API_KEY=<your-API-key>
export ANTHROPIC_API_KEY=<your-API-key>
export GOOGLE_API_KEY=<your-API-key>
export DEEPSEEK_API_KEY=<your-API-key>
```

## Quick Start

1. **Define model**:

```python
import smolmodels as sm

model = sm.Model(
    intent="Classify customer feedback as positive, negative, or neutral",
    input_schema={"text": str},
    output_schema={"sentiment": str}
)
```

2. **Build and save**:

```python
# Build with existing data
model.build(dataset=pd.read_csv("feedback.csv"), provider="openai:gpt-4o-mini")

# Or generate synthetic data
model.build(generate_samples=1000)

# Save model for later use
sm.save_model(model, "sentiment_model")
```

3. **Load and use**:

```python
# Load existing model
loaded_model = sm.load_model("sentiment_model")

# Make predictions
result = loaded_model.predict({"text": "Great service, highly recommend!"})
print(result["sentiment"])  # "positive"
```

## Benchmarks

Performance evaluated on 20 OpenML benchmark datasets and 12 Kaggle competitions. Higher performance observed on 12/20
OpenML datasets, with remaining datasets showing performance within 0.005 of baseline. Experiments conducted on standard
infrastructure (8 vCPUs, 30GB RAM) with 1-hour runtime limit per dataset.

Complete code and results are available at [plexe-ai/plexe-results](https://github.com/plexe-ai/plexe-results).

## Documentation

For full documentation, visit [docs.plexe.ai](https://docs.plexe.ai).

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache-2.0 License - see [LICENSE](LICENSE) for details.
