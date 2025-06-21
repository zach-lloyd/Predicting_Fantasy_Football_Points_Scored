import matplotlib.pyplot as plt

positions = ['QB', 'RB', 'WR', 'TE']

rf_model_maes = [39.4686, 43.0762, 41.8577, 25.4159]
xgb_model_maes = [37.6895, 40.4442, 40.2976, 23.9453]
rf_baseline_maes = [64.7629, 58.0342, 57.0397, 35.7293]
xgb_baseline_maes = [58.2566, 51.9242, 53.5469, 31.8777]

plt.figure(figsize=(6, 4))
plt.plot(positions, rf_model_maes, marker='o', label='RF Model', color='green')
plt.plot(positions, xgb_model_maes, marker='o', label='XGB Model', color='pink')
plt.plot(positions, rf_baseline_maes, marker='o', label='RF Baseline', color='black')
plt.plot(positions, xgb_baseline_maes, marker='o', label='XGB Baseline', color='blue')

plt.xlabel('Position')
plt.ylabel('Test MAE')
plt.legend()

plt.title('Test Mean Absolute Errors')
plt.show()