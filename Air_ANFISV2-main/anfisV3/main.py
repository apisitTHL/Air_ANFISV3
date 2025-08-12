import tkinter as tk
import numpy as np
import dask.dataframe as dd
from sklearn.preprocessing import MinMaxScaler
from tkinter import filedialog, ttk
from tkinter.scrolledtext import ScrolledText
import matplotlib.pyplot as plt
import threading
import testV2
import myANFIS_V2 as anfis
import skfuzzy as fuzz
from sklearn.metrics import mean_squared_error, r2_score
import datetime
import os

global checkTrained

# Create folder if it doesn't exist
folder_name = "logs"
os.makedirs(folder_name, exist_ok=True)

# Get the current date and time for the log filename
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Use the current timestamp to create a unique log file name
log_filename = os.path.join(folder_name, f"ANFIS_results_{current_time}.log")

# Open the log file in append mode with the timestamped filename
log_file = open(log_filename, 'a', encoding='utf-8')





def plot_results_in_main_thread(actual_output, anfis_predictions, bestnet, data):
    # Plots that were originally called from the background thread
    plot_Nodes(bestnet)
    plot_mf(bestnet, data)
    plot_predictions(actual_output, anfis_predictions)
    plot_r2(actual_output, anfis_predictions)
    
    log_message("Plots generated successfully.")
    log_message("##################################################")

def gbellmf(x, params):
    a, b, c = params
    if a == 0:
        return 0
    tmp = (x - c) / a
    return 1 / (1 + np.exp(2 * b * np.log(np.abs(tmp))))

def plot_Nodes(mynet):
    # Plot the Node Connections
    plt.figure()
    plt.imshow(mynet['config'], aspect='auto', cmap='cool')
    plt.title('Node Connections')
    plt.savefig('Plot_Node.png')
    plt.show()
    


def plot_mf(mynet, data):
    plt.figure(figsize=(12, 4))  # Create a single figure for all plots
    # print(mynet['mparams'])
    k = 0
    for i in range(0, mynet['ni'] * mynet['mf'], mynet['mf']):
        # print(k)
        plt.subplot(2, 2, k + 1)  # Create subplots for each input variable
        plt.title(f'Input {k + 1}')
        plt.xlabel('X')
        plt.ylabel('Degree of Membership')
        min_val = np.min(data[:, k])
        max_val = np.max(data[:, k])
        step = 0.1
        x = np.arange(min_val, max_val, step)
        # x = np.arange(-1, 1, step)

        for j in range(mynet['mf']):
            mf_x = fuzz.gbellmf(x, mynet['mparams'][i + j][0], mynet['mparams'][i + j][1], mynet['mparams'][i + j][2])
            plt.plot(x, mf_x, label=f'Membership Function {j + 1}')

        k = k + 1
        plt.tight_layout()
        plt.legend()
    plt.savefig('Plot_MF.png')
    plt.show()
    


def plot_predictions(actual_output, anfis_predictions):
    plt.figure()
    plt.plot(actual_output, 'b*', label='Actual Output')
    plt.plot(anfis_predictions, 'r-', linewidth=0.5, label='ANFIS Prediction')
    plt.xlabel('Data Point')
    plt.ylabel('Output Value')
    plt.legend()
    plt.savefig('Plot_Predict.png')
    plt.show()

def plot_r2(x, y):
    r_squared = r2_score(x, y)
    # Create a scatter plot for the data points
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, label=f'R-squared = {r_squared:.4f}')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)  # Add the identity line
    plt.xlabel('Actual Output')
    plt.ylabel('Predicted Output')
    plt.title("Correlation_Coefficient (R2)")
    plt.legend(loc='upper left')
    plt.tight_layout()
    # Show the plot
    plt.savefig('Plot_r2.png')
    plt.show
    

# Color scheme
BG_COLOR = "#1e1e2e"
FG_COLOR = "#ffffff"
ACCENT_COLOR = "#00e5ff"
BUTTON_COLOR = "#00ff85"
ENTRY_BG = "#32324e"

root = tk.Tk()
root.title("AI ANFIS Model Trainer")
root.configure(bg=BG_COLOR)
root.geometry("750x650")

# Font
FONT = ("Segoe UI", 11)
FONT_BOLD = ("Segoe UI", 11, "bold")

# Functions
def load_file():
    filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if filepath:
        entry_path.delete(0, tk.END)
        entry_path.insert(0, filepath)
        log_message(f"File loaded: {filepath}")

def start_train_thread():

    threading.Thread(target=start_train, daemon=True).start()

def start_test_thread():

    threading.Thread(target=start_test, daemon=True).start()
    
def load_test_file():
    load_testfile_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if load_testfile_path:
        entry_test_path.delete(0, tk.END)
        entry_test_path.insert(0, load_testfile_path)  # Corrected here
        log_message(f"Test file loaded: {load_testfile_path}")
   

def start_train():
    try:
        progress.start(10)
        log_message("Starting training...")

        epoch_n = int(entry_epoch.get())
        mf = int(entry_mf.get())
        step_size = float(entry_step_size.get())
        decrease_rate = float(entry_decrease.get())
        increase_rate = float(entry_increase.get())
        filepath = entry_path.get()

        log_message(f"Epoch: {epoch_n} Membership: {mf} Step_size: {step_size} Decrease: {decrease_rate} Increase: {increase_rate}")

        if not filepath:
            log_message("Error: No CSV file selected.")
            progress.stop()
            return

        bestnet, data, output, anfis_predictions = testV2.run_test(filepath, epoch_n, mf, step_size, decrease_rate, increase_rate, log,tk,np,dd,MinMaxScaler,anfis,log_message)
        
        # Call the plot results in main thread using root.after
        root.after(0, plot_results_in_main_thread, output, anfis_predictions, bestnet, data)


        log_message("Training completed successfully.")
        log_message("By AP_Lab")
        
        start_test()
        
        

    except ValueError as e:
        log_message(f"Value Error: {str(e)}")
    except Exception as e:
        log_message(f"Unexpected error: {str(e)}")
    finally:
        progress.stop()

def start_test():
    try:
        # if checkTrained == True:
        filepath = entry_test_path.get()
        anfis.test_gbell(dd,log_message,filepath)

        if not filepath:
            log_message("Error: No CSV file selected.")
            progress.stop()
            return
            
        # else:
        #     log_message("Please Trained Data Before Test Data")

        
    except ValueError as e:
        log_message(f"Value Error: {str(e)}")
    except Exception as e:
        log_message("Please Trained Data Before Test Data")
    finally:
        progress.stop()


def log_message(message):
    log.config(state=tk.NORMAL)
    log.insert(tk.END, message + "\n")
    log.see(tk.END)
    log.config(state=tk.DISABLED)

    log_file.write(message + "\n")
    log_file.flush()


# Add a method to close the log file when the application is closed


# GUI Components
tk.Label(root, text="AI ANFIS Model Trainer", bg=BG_COLOR, fg=ACCENT_COLOR, font=("Segoe UI", 16, "bold")).pack(pady=10)

frame = tk.Frame(root, bg=BG_COLOR)
frame.pack(pady=10)

# Modify grid placement for the test CSV input to appear below the training CSV
tk.Label(frame, text="CSV Path:", bg=BG_COLOR, fg=FG_COLOR, font=FONT_BOLD).grid(row=0, column=0, sticky='e', padx=5, pady=5)
entry_path = tk.Entry(frame, width=40, bg=ENTRY_BG, fg=FG_COLOR, font=FONT)
entry_path.grid(row=0, column=1, padx=5)
tk.Button(frame, text="Load CSV", command=load_file, bg=BUTTON_COLOR, fg=BG_COLOR, font=FONT_BOLD).grid(row=0, column=2, padx=5)

# Correct grid placement for Test CSV input to a different row
tk.Label(frame, text="Test CSV Path:", bg=BG_COLOR, fg=FG_COLOR, font=FONT_BOLD).grid(row=1, column=0, sticky='e', padx=5, pady=5)
entry_test_path = tk.Entry(frame, width=40, bg=ENTRY_BG, fg=FG_COLOR, font=FONT)
entry_test_path.grid(row=1, column=1, padx=5)
tk.Button(frame, text="Load Test CSV", command=load_test_file, bg=BUTTON_COLOR, fg=BG_COLOR, font=FONT_BOLD).grid(row=1, column=2, padx=5)

# Modify the rest of the layout
labels = ["Epochs:", "MF:", "Step Size:", "Decrease Rate:", "Increase Rate:"]
default_values = ["20", "3", "0.1", "0.9", "0.1"]
entries = []

for i, label in enumerate(labels):
    tk.Label(frame, text=label, bg=BG_COLOR, fg=FG_COLOR, font=FONT_BOLD).grid(row=i+2, column=0, sticky='e', padx=5, pady=5)  # Adjusted row index
    entry = tk.Entry(frame, bg=ENTRY_BG, fg=FG_COLOR, font=FONT)
    entry.grid(row=i+2, column=1, padx=5, pady=5)  # Adjusted row index
    entry.insert(0, default_values[i])
    entries.append(entry)
entry_epoch, entry_mf, entry_step_size, entry_decrease, entry_increase = entries



tk.Button(root, text="Start Training", command=start_train_thread, bg=BUTTON_COLOR, fg=BG_COLOR, font=FONT_BOLD, width=20).pack(pady=10)
tk.Button(root, text="Start Testing CSV", command=start_test_thread, bg=BUTTON_COLOR, fg=BG_COLOR, font=FONT_BOLD, width=20).pack(pady=10)


progress = ttk.Progressbar(root, mode="indeterminate")
progress.pack(pady=5, fill='x', padx=20)

log = ScrolledText(root, width=95, height=30, state=tk.DISABLED, bg=ENTRY_BG, fg=FG_COLOR, font=("Consolas", 10))
log.pack(pady=10)

root.mainloop()
