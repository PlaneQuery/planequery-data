from flights_ai_model.train import train
from flights_ai_model.training.training_days import sequence_training_days

train_paths = sequence_training_days(bts_days=5, sfdps_days=5)
model_output_paths = train(train_paths, epochs=10)

for path in model_output_paths:
    pass

