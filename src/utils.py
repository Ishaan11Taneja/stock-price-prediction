import matplotlib.pyplot as plt

def plot_actual_vs_predicted(dates, actual, predicted, title='Actual vs Predicted Prices'):
    plt.figure(figsize=(12,6))
    plt.plot(dates, actual, label='Actual')
    plt.plot(dates, predicted, label='Predicted')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title(title)
    plt.legend()
    plt.show() 