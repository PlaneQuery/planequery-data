# pick how many days
# maybe have an ExpiermentConfig class, that has name and such. 

from datetime import date

from flights_airport_nn.train import run_train



def run_experiment():
    train_dates = [date(2026,2,1)]
    eval_dates = [date(2026,3,1)]
    
    run_train(train_dates)
    pass