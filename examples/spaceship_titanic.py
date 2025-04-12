"""
This script demonstrates how to run the smolmodels ML engineering agent to build a predictive model. The example
uses the Kaggle 'Spaceship Titanic' competition's training dataset.

The dataset is owned and hosted by Kaggle, and is available at https://www.kaggle.com/c/spaceship-titanic/overview
under the Attribution 4.0 International (CC BY 4.0) license (https://creativecommons.org/licenses/by/4.0/). This
dataset is not part of the smolmodels package, and Plexe AI claims no ownership over it. The dataset is used here
for demonstration purposes only. Please refer to the Kaggle competition page for more details on the dataset and
its usage.

Citation:
Addison Howard, Ashley Chow, and Ryan Holbrook. Spaceship Titanic.
https://kaggle.com/competitions/spaceship-titanic, 2022. Kaggle.
"""

import pandas as pd
import smolmodels as sm

# Step 1: Define the model using the Spaceship Titanic problem statement as the model description
model = sm.Model(
    intent=(
        "From features describing a Spaceship Titanic passenger's information, determine whether they were "
        "transported or not."
    ),
    input_schema={
        "PassengerId": str,
        "HomePlanet": str,
        "CryoSleep": bool,
        "Cabin": str,
        "Destination": str,
        "Age": float,
        "VIP": bool,
        "RoomService": float,
        "FoodCourt": float,
        "ShoppingMall": float,
        "Spa": float,
        "VRDeck": float,
        "Name": str,
    },
    output_schema={
        "Transported": bool,
    },
)

# Step 2: Build the model using the Spaceship Titanic training dataset
model.build(
    datasets=[pd.read_csv("examples/datasets/spaceship-titanic-train.csv")],
    provider="openai/gpt-4o",
    max_iterations=5,
    timeout=300,  # 5 minute timeout
    run_timeout=150,
    verbose=True,
)

# Step 3: Run a prediction on the built model
test_df = pd.read_csv("examples/datasets/spaceship-titanic-test.csv")
predictions = pd.DataFrame.from_records([model.predict(x) for x in test_df.to_dict(orient="records")])

# Step 4: print a sample of predictions
print(predictions.sample(10))

# Step 5: Save the model
sm.save_model(model, "spaceship_titanic_model.tar.gz")

# Step 6: Print model description
description = model.describe()
print(description.as_text())
