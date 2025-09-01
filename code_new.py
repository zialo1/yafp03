#%% Load data
from os import stat_result

import numpy as np
import matplotlib.dates as mdates
from datetime import datetime, date
import matplotlib.pyplot as plt

# Import the data from the May21YA csv file
file_path = r"C:\Users\yanni\OneDrive\Desktop\Praktikum_Ph\May21YA.csv"
file_path = r"yannik.csv"

show_plots = "00000" # 5 plots not shown but saved

def load_csv2np(fname)->tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    # Load the CSV file, skipping the header and using comma as delimiter
    data = np.genfromtxt(fname=fname, delimiter=',', skip_header=1, dtype=str)
    # %% generate a time axis from the first column, the time is given in the format: HH:MM:SS
    time_str = data[:, 0]
    time = [datetime.strptime(t.strip('"'), "%H:%M:%S").time() for t in time_str]
    time_dt = [datetime.combine(date(1900, 1, 1), t) for t in time]

    # generate a data axis from the second column
    def safe_float(x):
        x = x.strip('"')
        return float(x) if x else np.nan

    bb_temp = np.array([safe_float(x) for x in data[:, 1]])
    tleft_in = np.array([safe_float(x) for x in data[:, 2]])
    tright_in = np.array([safe_float(x) for x in data[:, 3]])
    tleft_out = np.array([safe_float(x) for x in data[:, 4]])
    # Temperaturen in Kelvin
    offset=np.float64(273.15)
    return time_dt,bb_temp+offset, tleft_in+offset, tright_in+offset, tleft_out+offset

#show_plots = "11111" # shows all plots
time_dt, bb_temp_K, tleft_in_K, tright_in_K, tleft_out_K = load_csv2np(file_path)

# PLOT 1 - Alle Temperaturen über Zeit
# %% Generate a plot where time is plotted against all of the other data axes
# Create the figure and the first y-axis
fig, ax1 = plt.subplots(figsize=(10, 6))
# Plot Black Body Temperature on the left y-axis
ax1.plot(time_dt, bb_temp_K, label='Schwarzkörper Temperatur', color='blue')
ax1.set_xlabel('Zeit [h]')
ax1.set_ylabel('Schwarzkörper Temperatur [K]', color='black')
ax1.tick_params(axis='y', labelcolor='black')
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%#H'))
# Create a second y-axis for the wall temperatures
ax2 = ax1.twinx()
ax2.plot(time_dt, tleft_in_K, label='Linke Innenwand Temperatur', color='orange')
ax2.plot(time_dt, tright_in_K, label='Rechte Innenwand Temperatur', color='green')
ax2.plot(time_dt, tleft_out_K, label='Linke Aussenwand Temperatur', color='red')
ax2.set_ylabel('Umgebungstemperaturen [K]')
ax2.tick_params(axis='y')
# Combine legends from both axes
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax2.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
# plt.title('Temperature Measurements Over Time')
plt.grid()
plt.tight_layout()
# Save the plot to a file
plt.savefig('temperature_measurements_over_time.png')
if eval(show_plots[0]):
    plt.show()

# PLOT 2 - Konstanten
t_error = 0.1 # K
L_error = 0.0002 # m
curr_error = 0.005 # A
epsilon = 1
A_hp = 1899.4e-6 # m^2
true_sigma = 5.670374419e-8
voltages = np.array([4, 5, 6, 7, 8, 9])
currents = np.array([1.27, 1.58, 1.89, 2.20, 2.48, 2.81])
# Zeitstempel
time_stamps = [datetime.strptime(t.strip('"'), "%H:%M:%S") for t in [
"01:10:10", "01:59:20", "02:55:40", "03:38:10", "04:27:40", "05:14:30"
]]
time_stamps[-1] = time_dt[-1]
# As a verification, we plot the time stamps onto a plot where we also plot the black body temperature
plt.figure(figsize=(10, 6))
plt.plot(time_dt, bb_temp_K, label='Black Body Temperature (K)', color='blue')
for ts in time_stamps:
    plt.axvline(ts, color='red', linestyle='--', label=f'Time Stamp: {ts.strftime("%H:%M:%S")}')
plt.xlabel('Zeit (HH:MM:SS)')
plt.ylabel('Schwarzkörper-Temperatur (K)')
#plt.title('Black Body Temperature with Time Stamps')
plt.legend()
if eval(show_plots[1]):
    plt.show()

# PLOT 3- Mittelwerte
class stat_datapoint: # generates the statistical descriptions on the fly
    class innerlist(list): # hybrid of list and np.array; (-,*,**) forces np.array
        def __sub__(self, other):
            return np.array([xi - yi for xi, yi in zip(self, other)])

        def __add__(self, other):
            return np.array([xi - yi for xi, yi in zip(self, other)])

        def __mul__(self, other):
            return np.array([xi * other for xi in self])

        def __pow__(self, other):
            return np.array([xi ** other for xi in self])

    def __init__(self):
        self.avg = self.innerlist()
        self.std = self.innerlist()
        self.stderr = self.innerlist()

    def append(self,aslice:np.ndarray): # add 3 np.float64 to the innerlist
        self.avg.append(aslice.mean())
        self.std.append(aslice.std(ddof=1))
        self.stderr.append(self.std[-1] / np.sqrt(len(aslice)))

class sdatasets: # data class - append interface to np.ndarray
    def __init__(self):
        self.bb = stat_datapoint() # a list that calls np.array for the *,**,- operators
        self.iwalls = stat_datapoint()
        self.iwall_left = stat_datapoint()
        self.iwall_right = stat_datapoint()
        self.owall = stat_datapoint()

    def append(self,bb:np.ndarray,ileft:np.ndarray, iright:np.ndarray, outer:np.ndarray):
        self.bb.append(bb)
        self.owall.append(outer)
        self.iwall_left.append(ileft)
        self.iwall_right.append(iright)
        self.iwalls.append((ileft+iright)/2)

stdata=sdatasets() # a class for data statistical descriptions

# Statistische Fehler: Standardabweichung / sqrt(n)
n_points = 20
for ts in time_stamps:
    idx = np.argmin(np.abs(np.array(time_dt) - ts))
    start = max(0, idx - (n_points - 1))
    end = idx + 1

    # results are stored in data.bb data.iwalls data.iwall_left data.iwall_right data.owall
    stdata.append(bb_temp_K[start:end],
                  tleft_in_K[start:end], tright_in_K[start:end], tleft_out_K[start:end])

print("T_bb")
print(stdata.bb.avg)
print("T_Umgebung")
print(stdata.iwalls.avg)

# Verlustleistung
delta_T = stdata.bb.avg - stdata.iwalls.avg # np.array

pv_factor = np.float64(
            6 * 390 * (0.14e-3) ** 2 * np.pi / 16e-3 +
            2 * 69.9 * (0.13e-3) ** 2 * np.pi / 20e-3 +
            1 * 29.7 * (0.10e-3) ** 2 * np.pi / 30e-3 +
            1 * 19.2 * (0.10e-3) ** 2 * np.pi / 30e-3 )

P_loss = pv_factor * delta_T
P_eff = voltages * currents - P_loss
radiated_power = P_eff / (epsilon * A_hp)
# Temperatur-Differenz
temp4_diff = stdata.bb.avg ** 4 - stdata.iwalls.avg ** 4 # np.array
# Fehlerrechnung
PV_error = pv_factor * delta_T + 0.5874 * L_error * delta_T
P_eff_error = np.abs(voltages) * curr_error + PV_error
error_radiated_power = P_eff_error / (epsilon * A_hp)
# Fehler von T^4 Differenz (systematisch)
errors_temp4diff = np.sqrt(
4 * (np.abs(stdata.bb.avg ** 3) * t_error + np.abs(stdata.iwalls.avg ** 3) * t_error))
# Fehler von T^4 Differenz (statistisch)
error_temp4_diff_stat = 4 * np.array(stdata.bb.avg ** 3) * stdata.bb.stderr + 4 * stdata.iwalls.avg ** 3 * stdata.iwalls.stderr
# Stefan-Boltzmann-Konstante berechnen
sigma_calc = radiated_power / temp4_diff
# Systematischer Fehler von
sigma_error = (1 / (epsilon * A_hp)) * (
np.abs(1 / temp4_diff) * P_eff_error +
np.abs(P_eff / temp4_diff**2) * errors_temp4diff
)
# Statistischer Fehler von
sigma_stat_error = np.abs(sigma_calc / temp4_diff) * error_temp4_diff_stat
print("sigma")
print(sigma_calc)
print("sigma_sys")
print(sigma_error)
print("sigma_stat")
print(sigma_stat_error)
# Emissivität
epsilon = P_eff / (A_hp * true_sigma * temp4_diff)
# Systematischer Fehler von Emissivität
epsilon_error = (
(1 / (true_sigma * A_hp)) *
(np.abs(1 / temp4_diff) * P_eff_error +
np.abs(P_eff / temp4_diff**2) * errors_temp4diff)
)
# Statistischer Fehler von Emissivität
epsilon_stat_error = np.abs(epsilon / temp4_diff) * error_temp4_diff_stat
print("epsilon")
print(epsilon)
print("epsilon_sys")
print(epsilon_error)
print("epsilon_stat")
print(epsilon_stat_error)
# Plot: Strahlungsleistung vs. Temperatur-Differenz
plt.figure(figsize=(10, 6))
#plt.errorbar(temp4_diff, radiated_power, xerr=errors_temp4diff, yerr=error_radiated_power, fmt='o', color='purple', label="Messwerte")
plt.scatter(temp4_diff, radiated_power, color='black', label="Messwerte")
# Trendlinie einfügen
coeffs = np.polyfit(temp4_diff, radiated_power, 1)
fit_line = np.poly1d(coeffs)
x_fit = np.linspace(min(temp4_diff), max(temp4_diff), 100)
y_fit = fit_line(x_fit)
plt.plot(x_fit, y_fit, '-', color='black', label=f"Fit: y = {coeffs[0]:.3e} x + {coeffs[1]:.3e}")
plt.xlabel(r"$T_{\mathrm{Schwarzkörper}}^4 - T_{\mathrm{Umgebung}}^4\ [\mathrm{K}^4]$")
plt.ylabel("Normalisierte Strahlungsleistung [W/m²]")
plt.grid()
plt.legend()
plt.tight_layout()
if eval(show_plots[2]):
    plt.show()

# Plot 4: Berechnete über T_BB
plt.figure(figsize=(10, 6))
plt.errorbar(stdata.bb.avg, sigma_calc, xerr=t_error, yerr=sigma_stat_error, fmt='o', label='Stat. Fehler', color='red', capsize=3)
plt.errorbar(stdata.bb.avg, sigma_calc, xerr=t_error, yerr=sigma_error, fmt='o', label='System. Fehler', color='black', capsize=3)
plt.xlabel("Schwarzkörpertemperatur [K]")
plt.ylabel(" [W/m²·K]")
plt.grid()
plt.tight_layout()
if eval(show_plots[3]):
    plt.show()

# Plot: Emissivität über T_BB
plt.figure(figsize=(10, 6))
plt.errorbar(stdata.bb.avg, epsilon, xerr=t_error, yerr=epsilon_stat_error, fmt='o', label='Stat. Fehler', color='red', capsize=3)
plt.errorbar(stdata.bb.avg, epsilon, xerr=t_error, yerr=epsilon_error, fmt='o', label='System. Fehler', color='black', capsize=3)
plt.xlabel("Schwarzkörpertemperatur (K)")
plt.ylabel("Emissivität ")
plt.grid()
plt.tight_layout()
if eval(show_plots[4]):
    plt.show()

