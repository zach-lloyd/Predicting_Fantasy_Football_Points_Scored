import matplotlib.pyplot as plt

positions = ['QB', 'RB', 'WR', 'TE']

rf_model_r2s = [0.6545, 0.4837, 0.5140, 0.5623]
xgb_model_r2s = [0.6319, 0.4462, 0.4845, 0.5376]
rf_baseline_r2s = [0.2627, 0.1674, 0.2402, 0.2322]
xgb_baseline_r2s = [0.1777, 0.0921, 0.1774, 0.1739]

plt.figure(figsize=(6, 4))
plt.plot(positions, rf_model_r2s, marker='o', label='RF Model', color='green')
plt.plot(positions, xgb_model_r2s, marker='o', label='XGB Model', color='pink')
plt.plot(positions, rf_baseline_r2s, marker='o', label='RF Baseline', color='black')
plt.plot(positions, xgb_baseline_r2s, marker='o', label='XGB Baseline', color='blue')

plt.xlabel('Position')
plt.ylabel('R^2 Score')
plt.legend()

plt.title('R^2 Scores')
plt.show()