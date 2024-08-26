import numpy as np 
import matplotlib.pyplot as plt

ns_5 = {
    'nc': [5, 11, 16, 21, 31, 41],
    'CL': [0.71275432, 0.95294739, 1.01170338, 1.03246467, 1.03757968, 1.0302055],
}

ns_11 = {
    'nc': [5, 11, 16, 21, 31, 41],
    'CL': [0.67898853, 0.91053153, 0.96688689, 0.98665196, 0.99123974, 0.98393449],
}

ns_21 = {
    'nc': [5, 11, 16, 21, 31, 41],
    'CL': [0.66257284, 0.89022545, 0.94559196, 0.96496624, 0.96937396, 0.96212263],
}

ns_31 = {
    'nc': [5, 11, 16, 21, 26, 31, 41],
    'CL': [0.65688681, 0.88308449, 0.93810694, 0.95735138, 0.96250055, 0.96170423, 0.95447426],
}

ns_41 = {
    'nc': [5, 11, 16, 21, 26, 31, 41],
    'CL': [0.65422164, 0.87969244, 0.93453838, 0.95371656, 0.95884205, 0.95804032, 0.950819],
}

data_dict = {
    5: ns_5,
    11: ns_11,
    21: ns_21,
    31: ns_31,
    41: ns_41
}

true_val = 1.074902
margin_5 = 0.95*true_val
margin_10 = 0.9*true_val
margin_15 = 0.85*true_val
 
plt.plot([5,41], [true_val, true_val], 'k', label='experimental data')
plt.plot([5,41], [margin_5, margin_5], 'k--', label='5% error margin')
plt.plot([5,41], [margin_10, margin_10], 'k-.', label='10% error margin')
plt.plot([5,41], [margin_15, margin_15], 'k:', label='15% error margin')

for ns in data_dict.keys():
    plt.plot(np.array(data_dict[ns]['nc']) - 1, data_dict[ns]['CL'], '-*', label=f'ns = {ns-1} panels')

# plt.plot([3,41], [1.074902, 1.074902], 'k-*', label='experimental data')
# plt.ylim([1., 1.2])
plt.xlabel('chordwise panels (one-way)')
plt.ylabel('CL')
plt.grid()
plt.legend()

plt.show()