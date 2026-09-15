# pick how many days
# maybe have an ExpiermentConfig class, that has name and such. 

from datetime import date

from flights_airport_nn.inference import run_inference
from flights_airport_nn.train import run_train



def run_experiment():
    train_dates = [date(2026,2,1)]
    eval_dates = [date(2026,3,1)]

    model = run_train(train_dates)
    flights = run_inference(eval_dates[0], model)
    
if __name__ == "__main__":
    run_experiment()