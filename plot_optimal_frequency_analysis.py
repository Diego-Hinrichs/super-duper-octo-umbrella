#!/usr/bin/env python3
"""
Script para generar gráficos de análisis de frecuencia óptima vs N
Genera dos gráficos principales:
1. Curva de speedup (SFC-Opt vs Normal)
2. Frecuencia óptima vs N
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

def plot_speedup_curves(analysis_data, output_dir):
    """Genera grilla de gráficos de curvas de speedup (2x2)"""
    print("📊 Generando grilla de gráficos de speedup...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Curvas de Speedup: SFC Óptimo vs Normal', fontsize=16, fontweight='bold')
    
    # Configuración de subplots
    subplot_config = [
        ('directsum', 'Morton', 0, 0),
        ('directsum', 'Hilbert', 0, 1),
        ('barneshut', 'Morton', 1, 0),
        ('barneshut', 'Hilbert', 1, 1)
    ]
    
    for algorithm, curve, row, col in subplot_config:
        ax = axes[row, col]
        key = f"{algorithm}_{curve}"
        
        if key in analysis_data and analysis_data[key]['normal'] and analysis_data[key]['sfc_opt']:
            data = analysis_data[key]
            
            # Obtener datos ordenados por N
            n_values = sorted(data['normal'].keys())
            speedups = []
            
            for n in n_values:
                if n in data['normal'] and n in data['sfc_opt']:
                    normal_time = data['normal'][n]
                    opt_time = data['sfc_opt'][n]
                    speedup = normal_time / opt_time if opt_time > 0 else 0
                    speedups.append(speedup)
                else:
                    speedups.append(0)
            
            # Filtrar valores válidos
            valid_data = [(n, s) for n, s in zip(n_values, speedups) if s > 0]
            
            if valid_data:
                n_vals, speedup_vals = zip(*valid_data)
                
                ax.plot(n_vals, speedup_vals, 
                       marker='o', color='#2E86AB',
                       linewidth=2, markersize=4)
                
                # Línea de referencia en y=1
                ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
                
                ax.set_ylim(bottom=0.8)
            else:
                ax.text(0.5, 0.5, 'No hay datos\ndisponibles', 
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        else:
            ax.text(0.5, 0.5, 'No hay datos\ndisponibles', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        ax.set_title(f'{algorithm.title()} - {curve}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Número de Cuerpos (N)', fontsize=10)
        ax.set_ylabel('Speedup', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar gráfico
    speedup_file = os.path.join(output_dir, 'speedup_curves_grid.png')
    plt.savefig(speedup_file, dpi=300, bbox_inches='tight')
    print(f"✅ Grilla de speedup guardada: {speedup_file}")
    
    plt.close()

def plot_optimal_frequency_vs_n(analysis_data, output_dir):
    """Genera grilla de gráficos de frecuencia óptima vs N (2x2)"""
    print("📊 Generando grilla de gráficos de frecuencia óptima...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Frecuencia Óptima de Reordenamiento vs N', fontsize=16, fontweight='bold')
    
    # Configuración de subplots
    subplot_config = [
        ('directsum', 'Morton', 0, 0),
        ('directsum', 'Hilbert', 0, 1),
        ('barneshut', 'Morton', 1, 0),
        ('barneshut', 'Hilbert', 1, 1)
    ]
    
    for algorithm, curve, row, col in subplot_config:
        ax = axes[row, col]
        key = f"{algorithm}_{curve}"
        
        if key in analysis_data and analysis_data[key]['optimal_frequencies']:
            data = analysis_data[key]
            
            # Obtener datos ordenados por N
            n_values = sorted(data['optimal_frequencies'].keys())
            frequencies = [data['optimal_frequencies'][n] for n in n_values]
            
            # Filtrar valores válidos
            valid_data = [(n, f) for n, f in zip(n_values, frequencies) if f > 0]
            
            if valid_data:
                n_vals, freq_vals = zip(*valid_data)
                
                ax.plot(n_vals, freq_vals, 
                       marker='s', color='#A23B72',
                       linewidth=2, markersize=4)
                
                ax.set_ylim(bottom=0)
            else:
                ax.text(0.5, 0.5, 'No hay datos\ndisponibles', 
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        else:
            ax.text(0.5, 0.5, 'No hay datos\ndisponibles', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        
        ax.set_title(f'{algorithm.title()} - {curve}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Número de Cuerpos (N)', fontsize=10)
        ax.set_ylabel('Frecuencia Óptima', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar gráfico
    freq_file = os.path.join(output_dir, 'optimal_frequency_vs_n_grid.png')
    plt.savefig(freq_file, dpi=300, bbox_inches='tight')
    print(f"✅ Grilla de frecuencia óptima guardada: {freq_file}")
    
    plt.close()

def plot_performance_comparison(analysis_data, output_dir):
    """Genera grilla de gráficos de rendimiento vs frecuencia (2x2)"""
    print("📊 Generando grilla de gráficos de rendimiento vs frecuencia...")
    
    # Seleccionar un N representativo para mostrar el efecto de las frecuencias
    target_n = 10000  # N medio para el análisis
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Rendimiento vs Frecuencia de Reordenamiento', fontsize=16, fontweight='bold')
    
    # Configuración de subplots
    subplot_config = [
        ('directsum', 'Morton', 0, 0),
        ('directsum', 'Hilbert', 0, 1),
        ('barneshut', 'Morton', 1, 0),
        ('barneshut', 'Hilbert', 1, 1)
    ]
    
    for algorithm, curve, row, col in subplot_config:
        ax = axes[row, col]
        key = f"{algorithm}_{curve}"
        
        if key in analysis_data and analysis_data[key]['all_frequencies']:
            data = analysis_data[key]
            
            # Buscar el N más cercano al target
            available_n = set()
            for freq_data in data['all_frequencies'].values():
                available_n.update(freq_data.keys())
            
            if available_n:
                closest_n = min(available_n, key=lambda x: abs(x - target_n))
                
                # Recopilar datos para este N
                frequencies = []
                times = []
                
                for freq in sorted(data['all_frequencies'].keys()):
                    if closest_n in data['all_frequencies'][freq]:
                        frequencies.append(freq)
                        times.append(data['all_frequencies'][freq][closest_n])
                
                if len(frequencies) >= 2:
                    ax.plot(frequencies, times, 
                           marker='o', color='#F18F01',
                           linewidth=2, markersize=4)
                    
                    # Marcar la frecuencia óptima
                    if closest_n in data['optimal_frequencies']:
                        opt_freq = data['optimal_frequencies'][closest_n]
                        opt_time = data['sfc_opt'].get(closest_n, 0)
                        if opt_time > 0:
                            ax.scatter([opt_freq], [opt_time], 
                                     color='red', s=80, marker='*', 
                                     edgecolors='black', linewidth=1,
                                     zorder=5, label='Óptimo')
                    
                    ax.set_title(f'{algorithm.title()} - {curve} (N={closest_n})', fontsize=12, fontweight='bold')
                else:
                    ax.text(0.5, 0.5, 'Datos insuficientes\npara análisis', 
                           transform=ax.transAxes, ha='center', va='center',
                           fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                    ax.set_title(f'{algorithm.title()} - {curve}', fontsize=12, fontweight='bold')
            else:
                ax.text(0.5, 0.5, 'No hay datos\ndisponibles', 
                       transform=ax.transAxes, ha='center', va='center',
                       fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
                ax.set_title(f'{algorithm.title()} - {curve}', fontsize=12, fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No hay datos\ndisponibles', 
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            ax.set_title(f'{algorithm.title()} - {curve}', fontsize=12, fontweight='bold')
        
        ax.set_xlabel('Frecuencia de Reordenamiento', fontsize=10)
        ax.set_ylabel('Tiempo de Ejecución (s)', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar gráfico
    perf_file = os.path.join(output_dir, 'performance_vs_frequency_grid.png')
    plt.savefig(perf_file, dpi=300, bbox_inches='tight')
    print(f"✅ Grilla de rendimiento vs frecuencia guardada: {perf_file}")
    
    plt.close()

if __name__ == "__main__":
    # Para testing independiente
    print("Script de plotting para análisis de frecuencia óptima") 