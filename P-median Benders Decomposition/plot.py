import pandas as pd
import matplotlib.pyplot as plt

def plot_time_comparison(files):
    data_frames = [pd.read_csv(file) for file in files]
    
    plt.figure(figsize=(10, 6))
    
    for i, df in enumerate(data_frames):
        plt.plot(df['Time'], label=f'File {i + 1}')
    
    plt.xlabel('Index')
    plt.ylabel('Time')
    plt.title('Time Comparison')
    plt.legend()
    plt.show()

# Substitua 'file1.csv', 'file2.csv', etc., pelos nomes reais dos seus arquivos CSV
files_to_compare = ['resultados_p-median_f1.csv', 'resultados_p-median_f3.csv', 'resultados_p-median_bd.csv']

plot_time_comparison(files_to_compare)