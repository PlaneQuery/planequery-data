from flights_ai_model.train import train
from flights_ai_model.training.training_days import sequence_training_days

train_paths = sequence_training_days()
train(train_paths, epochs=10)