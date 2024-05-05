import numpy as np

def get_rolling_forecast(
    forecast, 
    data: 'array_like', 
    train_size: int, 
    horizon: int, 
    window: int
) -> list:
    total_size = train_size + horizon
    predictions = []

    for i in range(train_size, total_size, window):
        out_of_sample_predictions = forecast(data[:i], window)
        predictions.extend(out_of_sample_predictions)
    
    return predictions


def forecast_by_mean(data, window):
    mean = np.mean(data)
    return [mean for _ in range(window)]


def forecast_by_last_value(data, window):
    last_value = data[-1]
    return [last_value for _ in range(window)]