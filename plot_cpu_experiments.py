#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from datetime import datetime

# Set up plot style
plt.style.use('ggplot')
sns.set_context("talk")
colors = sns.color_palette("muted")

# Load data
data_dir = "results/cpu_experiments_20250603_155220"
csv_file = os.path.join(data_dir, "all_experiments.csv")
df = pd.read_csv(csv_file)

# Create output directory for plots
output_dir = os.path.join(data_dir, "plots")
os.makedirs(output_dir, exist_ok=True)

# Add timestamp to plot filenames
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# =============== 4. Effect of Space-Filling Curves (CURVES INSTEAD OF BARS) ===============
# Definir colores para cada tipo de SFC
sfc_colors = {
    'No order': 'blue',
    'Order: Morton': 'green',
    'Order: Hilbert': 'red'
}

# Definir marcadores para cada tamaño de problema
markers = {
    1000: 'o',
    5000: 's',
    10000: '^',
    20000: 'D',
    50000: '*'
}

# BarnesHut SFC impact - all problem sizes in one larger figure
plt.figure(figsize=(16, 10))  # Aumentado el tamaño para acomodar mejor la leyenda

# Primero organizamos los datos por tipo de SFC
for use_sfc in [False, True]:
    for curve_type in [0, 1]:
        if not use_sfc and curve_type > 0:
            continue  # Skip extra combinations for no SFC
            
        # Determinar el tipo de SFC y su color
        if not use_sfc:
            sfc_type = 'No order'
        else:
            if curve_type == 0:
                sfc_type = 'Order: Morton'
            else:
                sfc_type = 'Order: Hilbert'
        
        color = sfc_colors[sfc_type]
        
        # Luego para cada tamaño de problema
        for n in sorted(df['n_bodies'].unique()):
            data_points = []
            for t in sorted(df['threads'].unique()):
                subset = df[(df['n_bodies'] == n) & 
                          (df['algorithm'] == 'barneshut') & 
                          (df['threads'] == t) & 
                          (df['actual_threads'] == t) &
                          (df['use_sfc'] == use_sfc)]
                
                if use_sfc:
                    subset = subset[subset['curve_type'] == curve_type]
                
                if len(subset) > 0:
                    avg_time = subset['total_time'].mean()
                    data_points.append((t, avg_time))
            
            if data_points:
                threads, times = zip(*data_points)
                marker = markers[n]
                plt.semilogy(threads, times, color=color, marker=marker, 
                           linestyle='-', label=f'{sfc_type}, n={n}')

plt.xlabel('Number of Threads')
plt.ylabel('Average Total Time (s)')
plt.title('Impact of Space-Filling Curves on Barnes-Hut Performance')
plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1))  # Coloca la leyenda más alejada
plt.grid(True)
# Set x-ticks to exactly match thread values
plt.xticks(sorted(df['threads'].unique()))
plt.tight_layout()  # Ajusta automáticamente los márgenes
plt.savefig(os.path.join(output_dir, f'barneshut_sfc_impact_{timestamp}.png'), dpi=300, bbox_inches='tight')

# DirectSum SFC impact
plt.figure(figsize=(16, 10))  # Aumentado el tamaño para acomodar mejor la leyenda

# Primero organizamos los datos por tipo de SFC
for use_sfc in [False, True]:
    for curve_type in [0, 1]:
        if not use_sfc and curve_type > 0:
            continue  # Skip extra combinations for no SFC
            
        # Determinar el tipo de SFC y su color
        if not use_sfc:
            sfc_type = 'No order'
        else:
            if curve_type == 0:
                sfc_type = 'Order: Morton'
            else:
                sfc_type = 'Order: Hilbert'
        
        color = sfc_colors[sfc_type]
        
        # Luego para cada tamaño de problema
        for n in sorted(df['n_bodies'].unique()):
            data_points = []
            for t in sorted(df['threads'].unique()):
                subset = df[(df['n_bodies'] == n) & 
                          (df['algorithm'] == 'directsum') & 
                          (df['threads'] == t) & 
                          (df['actual_threads'] == t) &
                          (df['use_sfc'] == use_sfc)]
                
                if use_sfc:
                    subset = subset[subset['curve_type'] == curve_type]
                
                if len(subset) > 0:
                    avg_time = subset['total_time'].mean()
                    data_points.append((t, avg_time))
            
            if data_points:
                threads, times = zip(*data_points)
                marker = markers[n]
                plt.semilogy(threads, times, color=color, marker=marker, 
                           linestyle='-', label=f'{sfc_type}, n={n}')

plt.xlabel('Number of Threads')
plt.ylabel('Average Total Time (s)')
plt.title('Impact of Space-Filling Curves on DirectSum Performance')
plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1))  # Coloca la leyenda más alejada
plt.grid(True)
# Set x-ticks to exactly match thread values
plt.xticks(sorted(df['threads'].unique()))
plt.tight_layout()  # Ajusta automáticamente los márgenes
plt.savefig(os.path.join(output_dir, f'directsum_sfc_impact_{timestamp}.png'), dpi=300, bbox_inches='tight')

# =============== 5. Speedup for each SFC configuration (relative to same version with 1 thread) ===============
# BarnesHut SFC speedup
plt.figure(figsize=(16, 10))

# Primero organizamos los datos por tipo de SFC
for use_sfc in [False, True]:
    for curve_type in [0, 1]:
        if not use_sfc and curve_type > 0:
            continue
            
        # Determinar el tipo de SFC y su color
        if not use_sfc:
            sfc_type = 'No order'
        else:
            if curve_type == 0:
                sfc_type = 'Order: Morton'
            else:
                sfc_type = 'Order: Hilbert'
        
        color = sfc_colors[sfc_type]
        
        # Luego para cada tamaño de problema
        for n in sorted(df['n_bodies'].unique()):
            # Recopilamos los datos para todos los threads
            thread_times = {}
            for t in sorted(df['threads'].unique()):
                subset = df[(df['n_bodies'] == n) & 
                          (df['algorithm'] == 'barneshut') & 
                          (df['threads'] == t) & 
                          (df['actual_threads'] == t) &
                          (df['use_sfc'] == use_sfc)]
                
                if use_sfc:
                    subset = subset[subset['curve_type'] == curve_type]
                
                if len(subset) > 0:
                    avg_time = subset['total_time'].mean()
                    thread_times[t] = avg_time
            
            # Si tenemos datos para 1 thread, calculamos el speedup
            if 1 in thread_times and len(thread_times) > 1:
                baseline = thread_times[1]
                threads = []
                speedups = []
                
                for t, time in sorted(thread_times.items()):
                    threads.append(t)
                    # Calcular speedup: tiempo_1_thread / tiempo_n_threads
                    speedups.append(baseline / time)
                
                marker = markers[n]
                plt.plot(threads, speedups, color=color, marker=marker, 
                       linestyle='-', label=f'{sfc_type}, n={n}')

# Add the ideal speedup line (y = x)
thread_values = sorted(df['threads'].unique())
plt.plot(thread_values, thread_values, 'k--', alpha=0.5, label='Ideal Speedup')

plt.xlabel('Number of Threads')
plt.ylabel('Speedup (T1/Tn)')
plt.title('Barnes-Hut Speedup for Different SFC Configurations')
plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1))
plt.grid(True)
plt.xticks(sorted(df['threads'].unique()))
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'barneshut_sfc_speedup_{timestamp}.png'), dpi=300, bbox_inches='tight')

# DirectSum SFC speedup
plt.figure(figsize=(16, 10))

# Primero organizamos los datos por tipo de SFC
for use_sfc in [False, True]:
    for curve_type in [0, 1]:
        if not use_sfc and curve_type > 0:
            continue
            
        # Determinar el tipo de SFC y su color
        if not use_sfc:
            sfc_type = 'No order'
        else:
            if curve_type == 0:
                sfc_type = 'Order: Morton'
            else:
                sfc_type = 'Order: Hilbert'
        
        color = sfc_colors[sfc_type]
        
        # Luego para cada tamaño de problema
        for n in sorted(df['n_bodies'].unique()):
            # Recopilamos los datos para todos los threads
            thread_times = {}
            for t in sorted(df['threads'].unique()):
                subset = df[(df['n_bodies'] == n) & 
                          (df['algorithm'] == 'directsum') & 
                          (df['threads'] == t) & 
                          (df['actual_threads'] == t) &
                          (df['use_sfc'] == use_sfc)]
                
                if use_sfc:
                    subset = subset[subset['curve_type'] == curve_type]
                
                if len(subset) > 0:
                    avg_time = subset['total_time'].mean()
                    thread_times[t] = avg_time
            
            # Si tenemos datos para 1 thread, calculamos el speedup
            if 1 in thread_times and len(thread_times) > 1:
                baseline = thread_times[1]
                threads = []
                speedups = []
                
                for t, time in sorted(thread_times.items()):
                    threads.append(t)
                    # Calcular speedup: tiempo_1_thread / tiempo_n_threads
                    speedups.append(baseline / time)
                
                marker = markers[n]
                plt.plot(threads, speedups, color=color, marker=marker, 
                       linestyle='-', label=f'{sfc_type}, n={n}')

# Add the ideal speedup line (y = x)
thread_values = sorted(df['threads'].unique())
plt.plot(thread_values, thread_values, 'k--', alpha=0.5, label='Ideal Speedup')

plt.xlabel('Number of Threads')
plt.ylabel('Speedup (T1/Tn)')
plt.title('DirectSum Speedup for Different SFC Configurations')
plt.legend(loc='upper right', bbox_to_anchor=(1.4, 1))
plt.grid(True)
plt.xticks(sorted(df['threads'].unique()))
plt.tight_layout()
plt.savefig(os.path.join(output_dir, f'directsum_sfc_speedup_{timestamp}.png'), dpi=300, bbox_inches='tight')

print(f"Plots saved to {output_dir}") 