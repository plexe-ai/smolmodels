# smolmodels 🤖✨

Build specialized ML models using natural language.

## What is smolmodels?

smolmodels is a Python library that lets you create machine learning models by describing what you want them to do in plain English. Instead of wrestling with model architectures and hyperparameters, you simply describe your intent, define your inputs and outputs, and let smolmodels handle the rest.

```python
from smolmodels import Model

# Create a house price predictor with just a description
model = Model(
    intent="Predict house prices based on property features, optimizing for accuracy within 10% of actual values",
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

# Build the model - optionally generate synthetic training data
model.build(generate_samples=1000)

# Make predictions
price = model.predict({
    "square_feet": 2500,
    "bedrooms": 4,
    "location": "San Francisco",
    "year_built": 1985
})
```

## How Does It Work?

smolmodels uses a multi-step process to turn your natural language description into a working model:

1. **Intent Analysis**: Your description is analyzed to understand the type of model needed, key requirements, and success criteria.

2. **Data Generation**: If you don't have training data, smolmodels can generate synthetic data that matches your problem description and schema.

3. **Model Building**: The library:
   - Selects appropriate model architectures
   - Handles feature engineering
   - Manages training and validation
   - Ensures outputs meet your specified constraints

4. **Validation & Refinement**: The model is tested against your constraints and refined using directives (like "optimize for speed" or "prioritize explainability").

## Key Features

### Natural Language Intent 📝
Specify what you want your model to do in plain English. No need to worry about model architecture or hyperparameters.

### Smart Data Generation 🎲
Need training data? smolmodels can generate synthetic data that matches your problem description.

### Constraints & Validation ✅
Define rules your model must follow:
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

### Directives for Fine-tuning 🎯
Guide the model building process with high-level instructions:
```python
from smolmodels import Directive

model.build(directives=[
    Directive("Optimize for inference speed"),
    Directive("Prioritize interpretability")
])
```

## Installation

```bash
pip install smolmodels
```

## Quick Start

1. **Define your model**:
```python
from smolmodels import Model

model = Model(
    intent="Classify customer feedback as positive, negative, or neutral",
    input_schema={"text": str},
    output_schema={"sentiment": str}
)
```

2. **Build it**:
```python
# With existing data
model.build(dataset="feedback.csv")

# Or generate synthetic data
model.build(generate_samples=1000)
```

3. **Use it**:
```python
result = model.predict({"text": "Great service, highly recommend!"})
print(result["sentiment"])  # "positive"
```

## Documentation

For full documentation, visit [docs.plexe.ai](https://docs.plexe.ai).

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache-2.0 License - see [LICENSE](LICENSE) for details.
