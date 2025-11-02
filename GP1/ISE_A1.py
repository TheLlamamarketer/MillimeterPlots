import numpy as np
import matplotlib.pyplot as plt

data = {
    'h_0': [60, 60, 60, 59, 60, 60, 60, 59, 60, 59],
    'h_1': [99, 105, 96, 87, 79, 91, 94, 94, 83, 108],
    'h_3': [70, 71, 67, 66, 65, 70, 68, 68, 66, 77]
}

k = np.array([(h_1 - h_0) / (h_1 - h_3) for h_0, h_1, h_3 in zip(data['h_0'], data['h_1'], data['h_3'])])
k_avg = np.average(k)

end = [2, 5, 9]
ind = [i for i in range(len(k)) if i not in end]
k_avg_2 = np.average(k[ind])

dh = 0.5

dk = np.array([
    k_i * dh * np.sqrt(
        (1 / (h_1 - h_3)) ** 2 +
        (1 / (h_1 - h_0)) ** 2 +
        ((h_0 - h_3) / ((h_1 - h_3) * (h_1 - h_0))) ** 2
    )
    for k_i, h_0, h_1, h_3 in zip(k, data['h_0'], data['h_1'], data['h_3'])
])

dk_max = np.array([
    k_i * dh * (
        abs(1 / (h_1 - h_3)) +
        abs(1 / (h_1 - h_0)) +
        abs((h_0 - h_3) / ((h_1 - h_3) * (h_1 - h_0)))
    )
    for k_i, h_0, h_1, h_3 in zip(k, data['h_0'], data['h_1'], data['h_3'])
])

dk_avg = np.sqrt(sum((dk)**2))/len(dk)
dk_max_avg = np.sqrt(sum((dk_max)**2))/len(dk_max)
dk_avg_max = sum(dk) / len(dk)
dk_max_avg_max = sum(dk_max) / len(dk_max)

k_std = np.sqrt(sum((k - k_avg)**2) / ( len(k)*(len(k) - 1) ) )


dk_2 = dk[ind]
dk_avg_2 = np.sqrt(sum(np.square(dk_2))) / len(dk_2)

k2_std = np.sqrt(sum((k[ind] - k_avg_2)**2) / ( len(k[ind])*(len(k[ind]) - 1) ) )

print(f"$\\kappa_1 = {k_avg:.3f} \\pm avg dk:{dk_avg:.3f} or avg max dk:{dk_max_avg:.3f} or max avg dk:{dk_avg_max:.3f} or max avg max dk:{dk_max_avg_max:.3f} or std k:{k_std:.3f}$")
print(f"$\\kappa_2 = {k_avg_2:.4f} \\pm {dk_avg_2:.4f}$ or std k: ${k2_std:.4f}$")

plt.errorbar(np.linspace(1, 10, num=10), k, color='#51b2e2', linestyle='none', marker='x', yerr=dk, capsize=5)
plt.axhline(y=k_avg, color='g', linestyle='--')
plt.axhline(y=k_avg_2, color='r', linestyle='--')
plt.errorbar([3, 6, 10], k[[2, 5, 9]], color='r', linestyle = 'none', marker='x', yerr=dk[[2, 5, 9]], capsize=5)

plt.show()


def print_table(data, k, dk):
    num_rows = len(data['h_0'])

    print("\\begin{table}[h!]")
    print("    \\centering")
    print("    \\caption{{Messdaten Clément-Desormes-Apparatur ($h_i$ sind die Manometerwerte der Luft im Inneren: mit Außendruck und Außentemperatur (i=0), mit händisch erhöhtem Druck und Außentemperatur (i=1), nach Ablassen des händisch erhöhten Drucks und Ausdehnung beim Aufwärmen zur Außentemperatur (i=3); $\kappa$ ist der daraus berechnete Isotropenexponent von Luft und $\Delta\kappa$ der zugehörige Ablesefehler)}")
    print("    \\sisetup{table-format=2.2}")
    print("    \\begin{tabular}{| *{3}{S[table-format=2.2]} | *{2}{S[table-format=2.4]} |}")
    print("    \\toprule")
    header = "{$h_0$(mm)} & {$h_1$(mm)} & {$h_3$(mm)} & {$\\kappa$} & {$\\Delta\\kappa$}"
    print(f"    {header} \\\\")
    print("    \\midrule")
    
    for row in range(num_rows):
        h0 = data['h_0'][row]
        h1 = data['h_1'][row]
        h3 = data['h_3'][row]
        k_val = k[row]
        dk_val = dk[row]
        row_data = f"{h0:.0f} & {h1:.0f} & {h3:.0f} & {k_val:.4f} & {dk_val:.4f}"
        print("    " + row_data + " \\\\")
    
    print("    \\bottomrule")
    print("    \\end{tabular}")
    print("    \\label{tab:A1}")
    print("\\end{table}")




#print_table(data, k, dk)

