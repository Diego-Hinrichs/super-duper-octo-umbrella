#!/usr/bin/env python3
"""
Script para automatizar la experimentación de algoritmos CPU (BarnesHut y DirectSum)
Genera datasets, ejecuta experimentos con diferentes configuraciones y analiza resultados.
"""

import os
import subprocess
import time
import csv
import json
import argparse
import itertools
from datetime import datetime
from pathlib import Path
import importlib
import importlib.util

class CPUExperimentRunner:
    def __init__(self, base_dir="./", no_energy=False, algorithms=None):
        self.base_dir = Path(base_dir)
        self.results_dir = self.base_dir / "results" / f"cpu_experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.no_energy = no_energy
        
        # Algoritmos disponibles
        all_algorithms = {
            "barneshut": {
                "path": self.base_dir / "BarnesHut" / "BarnesHutCPU",
                "executable": "main"
            },
            "directsum": {
                "path": self.base_dir / "DirectSum" / "DirectSumCPU", 
                "executable": "main"
            }
        }
        
        # Filtrar algoritmos según la selección del usuario
        if algorithms:
            self.algorithms = {k: v for k, v in all_algorithms.items() if k in algorithms}
        else:
            self.algorithms = all_algorithms
            
        print(f"🎯 Algoritmos seleccionados: {list(self.algorithms.keys())}")
        
    def compile_algorithms(self):
        """Compila todos los algoritmos seleccionados"""
        print("🔨 Compilando algoritmos...")
        for name, config in self.algorithms.items():
            print(f"  Compilando {name}...")
            try:
                result = subprocess.run(
                    ["make", "clean"], 
                    cwd=config["path"], 
                    check=True, 
                    capture_output=True, 
                    text=True
                )
                result = subprocess.run(
                    ["make"], 
                    cwd=config["path"], 
                    check=True, 
                    capture_output=True, 
                    text=True
                )
                print(f"  ✅ {name} compilado exitosamente")
            except subprocess.CalledProcessError as e:
                print(f"  ❌ Error compilando {name}: {e}")
                print(f"  Error output: {e.stderr}")
                return False
        return True
    
    def generate_experiment_configurations(self):
        """Genera todas las configuraciones de experimentos"""
        # Configuraciones base para explorar
        configs = {
            # Número de cuerpos - escalabilidad
            "n_bodies": [1000, 5000, 10000, 20000, 50000],
            
            # Número de threads - paralelización (extendido hasta 32)
            "threads": [1, 2, 4, 8, 16, 32],
            
            # Configuraciones de SFC
            "sfc_configs": [
                {"use_sfc": False, "curve_type": 0, "fixed_reordering": False, "reorder_frequency": 0},  # Sin SFC
                {"use_sfc": True, "curve_type": 0, "fixed_reordering": False, "reorder_frequency": 0},   # Morton dinámico
                {"use_sfc": True, "curve_type": 1, "fixed_reordering": False, "reorder_frequency": 0},   # Hilbert dinámico
                {"use_sfc": True, "curve_type": 0, "fixed_reordering": True, "reorder_frequency": 10},   # Morton fijo freq=10
                {"use_sfc": True, "curve_type": 0, "fixed_reordering": True, "reorder_frequency": 20},   # Morton fijo freq=20
                {"use_sfc": True, "curve_type": 0, "fixed_reordering": True, "reorder_frequency": 50},   # Morton fijo freq=50
                {"use_sfc": True, "curve_type": 1, "fixed_reordering": True, "reorder_frequency": 10},   # Hilbert fijo freq=10
                {"use_sfc": True, "curve_type": 1, "fixed_reordering": True, "reorder_frequency": 20},   # Hilbert fijo freq=20
                {"use_sfc": True, "curve_type": 1, "fixed_reordering": True, "reorder_frequency": 50},   # Hilbert fijo freq=50
            ],
            
            # Distribuciones de masa
            "mass_distributions": [0, 1],  # UNIFORM, NORMAL
            
            # Número de iteraciones para cada test
            "iterations": [100],
            
            # Seeds para reproducibilidad
            "seeds": [42]
        }
        
        return configs
    
    def generate_test_configurations(self):
        """Genera configuraciones de test pequeñas para pruebas rápidas"""
        configs = {
            # Número de cuerpos - escalabilidad (reducido para pruebas)
            "n_bodies": [1000, 2000],
            
            # Número de threads - paralelización  
            "threads": [1, 2],
            
            # Configuraciones de SFC
            "sfc_configs": [
                {"use_sfc": False, "curve_type": 0, "fixed_reordering": False, "reorder_frequency": 0},  # Sin SFC
                {"use_sfc": True, "curve_type": 0, "fixed_reordering": False, "reorder_frequency": 0},   # Morton dinámico
                {"use_sfc": True, "curve_type": 0, "fixed_reordering": True, "reorder_frequency": 10},   # Morton fijo freq=10
                {"use_sfc": True, "curve_type": 0, "fixed_reordering": True, "reorder_frequency": 20},   # Morton fijo freq=20
                {"use_sfc": True, "curve_type": 0, "fixed_reordering": True, "reorder_frequency": 50},   # Morton fijo freq=50
            ],
            
            # Distribuciones de masa
            "mass_distributions": [0],  # Solo UNIFORM para tests rápidos
            
            # Número de iteraciones para cada test
            "iterations": [10],  # Menos iteraciones para tests rápidos
            
            # Seeds para reproducibilidad
            "seeds": [42]  # Solo una seed para tests rápidos
        }
        
        return configs
    
    def run_single_experiment(self, algorithm, config):
        """Ejecuta un experimento individual"""
        algo_config = self.algorithms[algorithm]
        
        # Construir comando usando los parámetros reales
        cmd = [
            f"./{algo_config['executable']}",
            "-n", str(config["n_bodies"]),
            "-t", str(config["threads"]),
            "-s", str(config["iterations"]),
            "-seed", str(config["seed"]),
            "-mass", "uniform" if config["mass_dist"] == 0 else "normal"
        ]
        
        # Agregar parámetros SFC
        if not config["use_sfc"]:
            cmd.append("-nosfc")
        else:
            cmd.extend(["-curve", "morton" if config["curve_type"] == 0 else "hilbert"])
            # Añadir parámetros de reordenamiento fijo si es aplicable
            cmd.extend(["-fixreorder", "1" if config["fixed_reordering"] else "0"])
            cmd.extend(["-freq", str(config["reorder_frequency"]) if config["reorder_frequency"] > 0 else "10"])
            
        # Desactivar cálculo de energía si se especifica
        if config.get("no_energy", False):
            cmd.extend(["-energy", "0"])
            
        print(f"    Ejecutando: {' '.join(cmd)}")
        
        try:
            start_time = time.time()
            result = subprocess.run(
                cmd,
                cwd=algo_config["path"],
                check=True,
                capture_output=True,
                text=True
                # Sin timeout - permitir que los experimentos tomen el tiempo necesario
            )
            end_time = time.time()
            
            # Parsear output para extraer métricas
            metrics = self.parse_output(result.stdout)
            metrics["wall_time"] = end_time - start_time
            metrics["algorithm"] = algorithm
            metrics.update(config)
            
            return metrics
            
        except subprocess.CalledProcessError as e:
            print(f"    ❌ Error en experimento: {e}")
            print(f"    Stderr: {e.stderr}")
            return None
    
    def parse_output(self, output):
        """Parsea la salida del programa para extraer métricas"""
        metrics = {}
        lines = output.strip().split('\n')
        steps = 100  # valor por defecto
        
        for line in lines:
            line = line.strip()
            
            # Extraer número de steps primero
            if "Steps:" in line:
                try:
                    steps = int(line.split(':')[1].strip())
                except:
                    steps = 100
            
            # Buscar métricas específicas en la salida
            elif "Average time per step:" in line:
                # Extraer valor de "Average time per step: 12.34 ms"
                value_str = line.split(':')[1].strip().replace('ms', '')
                avg_time_ms = float(value_str)
                metrics["avg_time_per_step"] = avg_time_ms
                # Calcular tiempo total en segundos
                metrics["total_time"] = (avg_time_ms * steps) / 1000.0
                
            elif "Compute forces:" in line:
                value_str = line.split(':')[1].strip().replace('ms', '')
                metrics["force_time"] = float(value_str)
                
            elif "Build tree:" in line:
                value_str = line.split(':')[1].strip().replace('ms', '')
                metrics["tree_time"] = float(value_str)
                
            elif "Reordering:" in line:
                value_str = line.split(':')[1].strip().replace('ms', '')
                metrics["sfc_time"] = float(value_str)
                
            elif "Bounding box:" in line:
                value_str = line.split(':')[1].strip().replace('ms', '')
                metrics["bbox_time"] = float(value_str)
                
            elif "Total Energy:" in line:
                value_str = line.split(':')[1].strip()
                metrics["total_energy"] = float(value_str)
                
            elif "Potential Energy:" in line:
                value_str = line.split(':')[1].strip()
                metrics["potential_energy"] = float(value_str)
                
            elif "Kinetic Energy:" in line:
                value_str = line.split(':')[1].strip()
                metrics["kinetic_energy"] = float(value_str)
                
            elif "Threads:" in line and "CPU" not in line:
                # Extraer número real de threads usados
                value_str = line.split(':')[1].strip()
                metrics["actual_threads"] = int(value_str)
                
            elif "Fixed reordering:" in line:
                value_str = line.split(':')[1].strip()
                metrics["is_fixed_reordering"] = value_str.lower() == "yes"
                
            elif "Optimal reorder freq:" in line:
                value_str = line.split(':')[1].strip()
                metrics["reorder_frequency"] = int(value_str)
        
        return metrics
    
    def run_scalability_experiments(self):
        """Ejecuta experimentos de escalabilidad"""
        print("🚀 Ejecutando experimentos de escalabilidad...")
        
        configs = self.generate_experiment_configurations()
        results = []
        
        # Test de escalabilidad por número de cuerpos
        for algorithm in self.algorithms.keys():
            print(f"\n📊 Algoritmo: {algorithm.upper()}")
            
            for n_bodies in configs["n_bodies"]:
                for threads in configs["threads"]:
                    for sfc_config in configs["sfc_configs"]:
                        for mass_dist in configs["mass_distributions"]:
                            for seed in configs["seeds"]:
                                config = {
                                    "n_bodies": n_bodies,
                                    "threads": threads,
                                    "iterations": configs["iterations"][0],
                                    "seed": seed,
                                    "mass_dist": mass_dist,
                                    "no_energy": self.no_energy,
                                    **sfc_config
                                }
                                
                                sfc_info = f"SFC={sfc_config['use_sfc']}, Curve={sfc_config['curve_type']}"
                                if sfc_config['use_sfc']:
                                    sfc_info += f", Fixed={sfc_config['fixed_reordering']}"
                                    if sfc_config['fixed_reordering']:
                                        sfc_info += f", Freq={sfc_config['reorder_frequency']}"
                                
                                print(f"  🔬 N={n_bodies}, T={threads}, {sfc_info}, Mass={mass_dist}")
                                
                                result = self.run_single_experiment(algorithm, config)
                                if result:
                                    results.append(result)
                                    print(f"    ✅ Tiempo total: {result.get('total_time', 'N/A')}s")
        
        return results
    
    def run_thread_efficiency_experiments(self):
        """Ejecuta experimentos específicos de eficiencia de threads"""
        print("🧵 Ejecutando experimentos de eficiencia de threads...")
        
        results = []
        # Usar configuración más pequeña en modo test
        fixed_bodies = 2000 if hasattr(self, 'generate_test_configurations') else 20000
        iterations = 10 if hasattr(self, 'generate_test_configurations') else 50
        thread_list = [1, 2, 4] if hasattr(self, 'generate_test_configurations') else [1, 2, 4, 8, 16, 32]
        
        # Configuraciones de SFC a probar
        sfc_configs = [
            {"use_sfc": True, "curve_type": 0, "fixed_reordering": False, "reorder_frequency": 0},  # Morton dinámico
            {"use_sfc": True, "curve_type": 0, "fixed_reordering": True, "reorder_frequency": 10},  # Morton fijo freq=10
            {"use_sfc": True, "curve_type": 0, "fixed_reordering": True, "reorder_frequency": 20},  # Morton fijo freq=20
            {"use_sfc": True, "curve_type": 0, "fixed_reordering": True, "reorder_frequency": 50},  # Morton fijo freq=50
        ]
        
        for algorithm in self.algorithms.keys():
            print(f"\n📊 Algoritmo: {algorithm.upper()}")
            
            for threads in thread_list:
                for sfc_config in sfc_configs:
                    config = {
                        "n_bodies": fixed_bodies,
                        "threads": threads,
                        "iterations": iterations,
                        "seed": 42,
                        "mass_dist": 0,  # UNIFORM
                        "no_energy": self.no_energy,
                        **sfc_config
                    }
                    
                    sfc_info = f"SFC={sfc_config['use_sfc']}, Curve={sfc_config['curve_type']}"
                    if sfc_config['use_sfc']:
                        sfc_info += f", Fixed={sfc_config['fixed_reordering']}"
                        if sfc_config['fixed_reordering']:
                            sfc_info += f", Freq={sfc_config['reorder_frequency']}"
                    
                    print(f"  🔬 Threads: {threads}, {sfc_info}")
                    
                    result = self.run_single_experiment(algorithm, config)
                    if result:
                        results.append(result)
                        print(f"    ✅ Tiempo total: {result.get('total_time', 'N/A')}s")
                    
        return results
    
    def run_sfc_frequency_experiments(self):
        """Ejecuta experimentos específicos para analizar el impacto de diferentes frecuencias de reordenamiento"""
        print("🔄 Ejecutando experimentos de frecuencia de reordenamiento SFC...")
        
        results = []
        # Usar configuración adaptada para análisis de frecuencia
        n_bodies = 10000  # Tamaño fijo para este análisis
        iterations = 100  # Más iteraciones para ver mejor el impacto
        threads = 4       # Número fijo de threads
        
        # Probar diferentes frecuencias de reordenamiento
        freq_configs = [
            {"use_sfc": True, "curve_type": 0, "fixed_reordering": False, "reorder_frequency": 0},   # Morton dinámico (línea base)
            {"use_sfc": True, "curve_type": 0, "fixed_reordering": True, "reorder_frequency": 5},    # Morton freq=5
            {"use_sfc": True, "curve_type": 0, "fixed_reordering": True, "reorder_frequency": 10},   # Morton freq=10
            {"use_sfc": True, "curve_type": 0, "fixed_reordering": True, "reorder_frequency": 15},   # Morton freq=15
            {"use_sfc": True, "curve_type": 0, "fixed_reordering": True, "reorder_frequency": 20},   # Morton freq=20
            {"use_sfc": True, "curve_type": 0, "fixed_reordering": True, "reorder_frequency": 30},   # Morton freq=30
            {"use_sfc": True, "curve_type": 0, "fixed_reordering": True, "reorder_frequency": 50},   # Morton freq=50
            {"use_sfc": True, "curve_type": 0, "fixed_reordering": True, "reorder_frequency": 100},  # Morton freq=100
            {"use_sfc": True, "curve_type": 1, "fixed_reordering": False, "reorder_frequency": 0},   # Hilbert dinámico (línea base)
            {"use_sfc": True, "curve_type": 1, "fixed_reordering": True, "reorder_frequency": 5},    # Hilbert freq=5
            {"use_sfc": True, "curve_type": 1, "fixed_reordering": True, "reorder_frequency": 10},   # Hilbert freq=10
            {"use_sfc": True, "curve_type": 1, "fixed_reordering": True, "reorder_frequency": 15},   # Hilbert freq=15
            {"use_sfc": True, "curve_type": 1, "fixed_reordering": True, "reorder_frequency": 20},   # Hilbert freq=20
            {"use_sfc": True, "curve_type": 1, "fixed_reordering": True, "reorder_frequency": 30},   # Hilbert freq=30
            {"use_sfc": True, "curve_type": 1, "fixed_reordering": True, "reorder_frequency": 50},   # Hilbert freq=50
            {"use_sfc": True, "curve_type": 1, "fixed_reordering": True, "reorder_frequency": 100},  # Hilbert freq=100
        ]
        
        for algorithm in self.algorithms.keys():
            print(f"\n📊 Algoritmo: {algorithm.upper()}")
            
            for sfc_config in freq_configs:
                config = {
                    "n_bodies": n_bodies,
                    "threads": threads,
                    "iterations": iterations,
                    "seed": 42,
                    "mass_dist": 0,  # UNIFORM
                    "no_energy": self.no_energy,
                    **sfc_config
                }
                
                curve_name = "Morton" if sfc_config["curve_type"] == 0 else "Hilbert"
                reorder_info = "Dinámico" if not sfc_config["fixed_reordering"] else f"Fijo-{sfc_config['reorder_frequency']}"
                
                print(f"  🔬 {curve_name}, Reordenamiento: {reorder_info}")
                
                result = self.run_single_experiment(algorithm, config)
                if result:
                    results.append(result)
                    print(f"    ✅ Tiempo total: {result.get('total_time', 'N/A')}s, Freq: {result.get('reorder_frequency', 'N/A')}")
        
        return results
    
    def run_optimal_frequency_vs_n_experiments(self):
        """Ejecuta experimentos para analizar la frecuencia óptima vs número de cuerpos (N)"""
        print("📈 Ejecutando experimentos de frecuencia óptima vs N...")
        
        results = []
        threads = 32  # Número fijo de threads para este análisis
        iterations = 100  # Iteraciones fijas según especificación
        
        # 20+ PUNTOS DE N para una curva muy suave
        n_bodies_list = [
            1000, 1200, 1500, 1800, 2000, 2500, 3000, 3500, 4000, 4500,
            5000, 6000, 7000, 8000, 9000, 10000, 12000, 15000, 18000, 20000,
            25000, 30000, 35000, 40000, 45000, 50000
        ]
        
        # Frecuencias específicas solicitadas
        frequencies_to_test = [5, 10, 15, 20, 50]
        
        for algorithm in self.algorithms.keys():
            print(f"\n📊 Algoritmo: {algorithm.upper()}")
            
            for n_bodies in n_bodies_list:
                print(f"\n  🔬 Analizando N={n_bodies}...")
                
                # 1. BASELINE: Sin SFC (ejecutar solo una vez por configuración N/algorithm)
                baseline_config = {
                    "n_bodies": n_bodies,
                    "threads": threads,
                    "iterations": iterations,
                    "seed": 42,
                    "mass_dist": 0,  # UNIFORM
                    "no_energy": self.no_energy,
                    "use_sfc": False,
                    "curve_type": 0,  # No importa el tipo de curva para baseline
                    "fixed_reordering": False,
                    "reorder_frequency": 0
                }
                
                print(f"    🚫 Baseline (Sin SFC)")
                baseline_result = self.run_single_experiment(algorithm, baseline_config)
                baseline_time = 0
                if baseline_result:
                    baseline_result["experiment_type"] = "optimal_frequency_vs_n"
                    baseline_result["sfc_configuration"] = "baseline"
                    results.append(baseline_result)
                    baseline_time = baseline_result.get('total_time', 0)
                    print(f"      ✅ Tiempo baseline: {baseline_time:.3f}s")
                
                # 2. Probar cada curva con todas las frecuencias
                for curve_type in [0, 1]:  # Morton y Hilbert
                    curve_name = "Morton" if curve_type == 0 else "Hilbert"
                    print(f"\n    📍 Curva: {curve_name}")
                    
                    # Probar diferentes frecuencias
                    best_fixed_freq = None
                    best_fixed_time = float('inf')
                    
                    print(f"      🔍 Probando frecuencias: {frequencies_to_test}")
                    for freq in frequencies_to_test:
                        fixed_config = {
                            "n_bodies": n_bodies,
                            "threads": threads,
                            "iterations": iterations,
                            "seed": 42,
                            "mass_dist": 0,  # UNIFORM
                            "no_energy": self.no_energy,
                            "use_sfc": True,
                            "curve_type": curve_type,
                            "fixed_reordering": True,
                            "reorder_frequency": freq
                        }
                        
                        fixed_result = self.run_single_experiment(algorithm, fixed_config)
                        if fixed_result:
                            fixed_result["experiment_type"] = "optimal_frequency_vs_n"
                            fixed_result["sfc_configuration"] = f"sfc_freq_{freq}"
                            results.append(fixed_result)
                            fixed_time = fixed_result.get('total_time', 0)
                            
                            # Verificar si esta es la mejor frecuencia
                            if fixed_time < best_fixed_time:
                                best_fixed_time = fixed_time
                                best_fixed_freq = freq
                            
                            print(f"        ⚡ Freq={freq}: {fixed_time:.3f}s {'← MEJOR' if fixed_time == best_fixed_time else ''}")
                    
                    # 3. SFC-OPT: Crear resultado explícito con la mejor frecuencia
                    if best_fixed_freq is not None:
                        sfc_opt_result = {
                            "algorithm": algorithm,
                            "curve_type": curve_type,
                            "n_bodies": n_bodies,
                            "threads": threads,
                            "iterations": iterations,
                            "seed": 42,
                            "mass_dist": 0,
                            "no_energy": self.no_energy,
                            "use_sfc": True,
                            "fixed_reordering": True,
                            "reorder_frequency": best_fixed_freq,
                            "total_time": best_fixed_time,
                            "experiment_type": "optimal_frequency_vs_n",
                            "sfc_configuration": "sfc_opt"
                        }
                        results.append(sfc_opt_result)
                        print(f"      🎯 SFC-Opt (freq={best_fixed_freq}): {best_fixed_time:.3f}s")
                        
                        # Speedup calculation
                        if baseline_time > 0:
                            speedup = baseline_time / best_fixed_time if best_fixed_time > 0 else 0
                            print(f"        📊 Speedup vs Baseline: {speedup:.2f}x")
        
        return results
    
    def analyze_optimal_frequency_vs_n(self, results):
        """Analiza los resultados de frecuencia óptima vs N y genera gráficos"""
        print("\n📊 Analizando resultados de frecuencia óptima vs N...")
        
        # Filtrar solo los resultados del experimento correcto
        filtered_results = [r for r in results if r.get('experiment_type') == 'optimal_frequency_vs_n']
        print(f"📋 Total de resultados del experimento: {len(filtered_results)}")
        
        # Organizar datos por algoritmo y curva
        analysis_data = {}
        
        for result in filtered_results:
            algorithm = result.get('algorithm', 'unknown')
            curve_type = result.get('curve_type', 0)
            curve_name = "Morton" if curve_type == 0 else "Hilbert"
            sfc_config = result.get('sfc_configuration', 'unknown')
            n_bodies = result.get('n_bodies', 0)
            
            key = f"{algorithm}_{curve_name}"
            
            if key not in analysis_data:
                analysis_data[key] = {
                    'normal': {},           # Sin SFC
                    'sfc_opt': {},         # SFC con frecuencia óptima
                    'optimal_frequencies': {},  # Las frecuencias óptimas encontradas
                    'all_frequencies': {}  # Todos los resultados por frecuencia
                }
            
            if sfc_config == 'normal':
                analysis_data[key]['normal'][n_bodies] = result.get('total_time', 0)
                
            elif sfc_config == 'sfc_opt':
                analysis_data[key]['sfc_opt'][n_bodies] = result.get('total_time', 0)
                analysis_data[key]['optimal_frequencies'][n_bodies] = result.get('reorder_frequency', 0)
                print(f"  ✅ SFC-Opt guardado: {key}, N={n_bodies}, freq={result.get('reorder_frequency', 0)}, tiempo={result.get('total_time', 0):.3f}s")
                
            elif sfc_config.startswith('sfc_freq_'):
                # Guardar todos los resultados por frecuencia para análisis detallado
                freq = result.get('reorder_frequency', 0)
                if freq not in analysis_data[key]['all_frequencies']:
                    analysis_data[key]['all_frequencies'][freq] = {}
                analysis_data[key]['all_frequencies'][freq][n_bodies] = result.get('total_time', 0)
        
        # Verificar que tenemos datos
        total_data_points = 0
        for key, data in analysis_data.items():
            normal_count = len(data['normal'])
            sfc_opt_count = len(data['sfc_opt'])
            freq_count = len(data['optimal_frequencies'])
            all_freq_count = sum(len(freq_data) for freq_data in data['all_frequencies'].values())
            
            total_data_points += normal_count + sfc_opt_count + all_freq_count
            
            print(f"📈 {key}:")
            print(f"  - Normal: {normal_count} puntos")
            print(f"  - SFC-Opt: {sfc_opt_count} puntos")
            print(f"  - Frecuencias óptimas: {freq_count} puntos")
            print(f"  - Datos por frecuencia: {all_freq_count} puntos")
            
            # Mostrar algunas frecuencias óptimas para verificar
            if data['optimal_frequencies']:
                sample_freqs = list(data['optimal_frequencies'].items())[:5]
                print(f"  - Muestra de frecuencias: {sample_freqs}")
        
        print(f"\n📊 Total de puntos de datos: {total_data_points}")
        
        if total_data_points == 0:
            print("❌ No se encontraron datos válidos para analizar!")
            return
        
        print(f"\n📊 Datos organizados y listos para graficar")
        
        # Generar gráficos
        try:
            # Importar el módulo de plotting
            plotting_module_path = os.path.join(os.path.dirname(__file__), 'plot_optimal_frequency_analysis.py')
            if os.path.exists(plotting_module_path):
                spec = importlib.util.spec_from_file_location("plot_optimal_frequency_analysis", plotting_module_path)
                plotting_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(plotting_module)
                
                print("📊 Generando gráficos...")
                
                # Gráfico de frecuencia óptima vs N
                plotting_module.plot_optimal_frequency_vs_n(analysis_data, self.results_dir)
                
                # Gráfico de comparación de rendimiento
                plotting_module.plot_performance_comparison(analysis_data, self.results_dir)
                
                # Gráfico de curvas de speedup (si hay datos)
                plotting_module.plot_speedup_curves(analysis_data, self.results_dir)
                
                print("✅ Gráficos generados exitosamente!")
                
            else:
                print(f"❌ No se encontró el módulo de plotting en: {plotting_module_path}")
                
        except Exception as e:
            print(f"❌ Error generando gráficos: {e}")
            import traceback
            traceback.print_exc()
        
        # Resumen estadístico
        print("\n📈 RESUMEN ESTADÍSTICO:")
        for key, data in analysis_data.items():
            if data['optimal_frequencies']:
                frequencies = list(data['optimal_frequencies'].values())
                n_values = list(data['optimal_frequencies'].keys())
                
                print(f"\n🔍 {key}:")
                print(f"  - Rango de N: {min(n_values)} - {max(n_values)}")
                print(f"  - Frecuencias óptimas: {min(frequencies)} - {max(frequencies)}")
                print(f"  - Frecuencia promedio: {sum(frequencies)/len(frequencies):.1f}")
                print(f"  - Total de puntos: {len(frequencies)}")
        
        return analysis_data
    
    def save_results(self, results, filename):
        """Guarda resultados solo en JSON"""
        if not results:
            print("⚠️ No hay resultados para guardar")
            return

        # Guardar como JSON para análisis posterior
        json_file = self.results_dir / f"{filename}.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"💾 Resultados guardados en: {json_file}")
    
    def generate_summary_report(self, results):
        """Genera un reporte resumen de los experimentos"""
        if not results:
            return
            
        report_file = self.results_dir / "experiment_summary.txt"
        
        with open(report_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("REPORTE DE EXPERIMENTACIÓN CPU\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total de experimentos: {len(results)}\n\n")
            
            # Análisis por algoritmo
            algorithms = set(r["algorithm"] for r in results)
            for algo in algorithms:
                algo_results = [r for r in results if r["algorithm"] == algo]
                f.write(f"Algoritmo: {algo.upper()}\n")
                f.write("-" * 30 + "\n")
                
                if algo_results:
                    avg_time = sum(r.get("total_time", 0) for r in algo_results) / len(algo_results)
                    f.write(f"Tiempo promedio: {avg_time:.3f}s\n")
                    
                    min_time = min(r.get("total_time", float('inf')) for r in algo_results)
                    max_time = max(r.get("total_time", 0) for r in algo_results)
                    f.write(f"Tiempo mínimo: {min_time:.3f}s\n")
                    f.write(f"Tiempo máximo: {max_time:.3f}s\n")
                    
                f.write("\n")
        
        print(f"📋 Reporte generado en: {report_file}")

    def generate_sfc_analysis_report(self, results):
        """Genera un reporte específico analizando el desempeño de diferentes configuraciones SFC"""
        if not results:
            return
            
        # Filtrar solo resultados que usan SFC
        sfc_results = [r for r in results if r.get("use_sfc", False)]
        if not sfc_results:
            print("⚠️ No hay resultados con SFC para analizar")
            return
            
        report_file = self.results_dir / "sfc_analysis_report.txt"
        
        with open(report_file, 'w') as f:
            f.write("=" * 70 + "\n")
            f.write("ANÁLISIS DE RENDIMIENTO DE CONFIGURACIONES SFC\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total de experimentos SFC: {len(sfc_results)}\n\n")
            
            # Analizar por algoritmo
            algorithms = sorted(set(r["algorithm"] for r in sfc_results))
            for algo in algorithms:
                algo_results = [r for r in sfc_results if r["algorithm"] == algo]
                f.write(f"\n{'=' * 50}\n")
                f.write(f"ALGORITMO: {algo.upper()}\n")
                f.write(f"{'=' * 50}\n\n")
                
                # Análisis por curva
                for curve_type in [0, 1]:  # Morton, Hilbert
                    curve_name = "MORTON" if curve_type == 0 else "HILBERT"
                    curve_results = [r for r in algo_results if r["curve_type"] == curve_type]
                    
                    if not curve_results:
                        continue
                        
                    f.write(f"\n{'-' * 40}\n")
                    f.write(f"CURVA: {curve_name}\n")
                    f.write(f"{'-' * 40}\n\n")
                    
                    # Resultados dinámicos vs. fijos
                    dynamic_results = [r for r in curve_results if not r.get("fixed_reordering", False)]
                    fixed_results = [r for r in curve_results if r.get("fixed_reordering", False)]
                    
                    # Análisis de reordenamiento dinámico
                    if dynamic_results:
                        f.write("REORDENAMIENTO DINÁMICO:\n")
                        avg_time = sum(r.get("total_time", 0) for r in dynamic_results) / len(dynamic_results)
                        avg_freq = sum(r.get("reorder_frequency", 0) for r in dynamic_results) / len(dynamic_results)
                        
                        f.write(f"  Tiempo promedio: {avg_time:.3f}s\n")
                        f.write(f"  Frecuencia de reordenamiento promedio: {avg_freq:.1f}\n")
                        
                        # Agrupar por número de cuerpos para ver cómo varía la frecuencia
                        body_counts = sorted(set(r.get("n_bodies", 0) for r in dynamic_results))
                        f.write("\n  Frecuencia por número de cuerpos:\n")
                        for n_bodies in body_counts:
                            n_body_results = [r for r in dynamic_results if r.get("n_bodies", 0) == n_bodies]
                            if n_body_results:
                                avg_freq = sum(r.get("reorder_frequency", 0) for r in n_body_results) / len(n_body_results)
                                f.write(f"    N={n_bodies}: {avg_freq:.1f}\n")
                        
                        f.write("\n")
                    
                    # Análisis de reordenamiento fijo
                    if fixed_results:
                        f.write("REORDENAMIENTO FIJO:\n")
                        # Agrupar por frecuencia de reordenamiento
                        freq_values = sorted(set(r.get("reorder_frequency", 0) for r in fixed_results))
                        
                        f.write("\n  Rendimiento por frecuencia de reordenamiento:\n")
                        for freq in freq_values:
                            freq_results = [r for r in fixed_results if r.get("reorder_frequency", 0) == freq]
                            if freq_results:
                                avg_time = sum(r.get("total_time", 0) for r in freq_results) / len(freq_results)
                                f.write(f"    Freq={freq}: {avg_time:.3f}s\n")
                        
                        # Encontrar la mejor frecuencia
                        best_freq = 0
                        best_time = float('inf')
                        for freq in freq_values:
                            freq_results = [r for r in fixed_results if r.get("reorder_frequency", 0) == freq]
                            if freq_results:
                                avg_time = sum(r.get("total_time", 0) for r in freq_results) / len(freq_results)
                                if avg_time < best_time:
                                    best_time = avg_time
                                    best_freq = freq
                        
                        f.write(f"\n  Mejor frecuencia de reordenamiento: {best_freq} (tiempo: {best_time:.3f}s)\n")
                        
                        # Comparar con dinámico
                        if dynamic_results:
                            dynamic_avg_time = sum(r.get("total_time", 0) for r in dynamic_results) / len(dynamic_results)
                            diff_percent = (dynamic_avg_time - best_time) / dynamic_avg_time * 100
                            f.write(f"  Comparación con dinámico: {diff_percent:.2f}% {'mejor' if diff_percent > 0 else 'peor'}\n")
                        
                        f.write("\n")
                    
                    # Comparativa global entre frecuencias
                    if len(freq_values) > 1:
                        f.write("\nCOMPARATIVA DE RENDIMIENTO POR FRECUENCIA Y TAMAÑO:\n")
                        
                        # Para cada tamaño de cuerpos, ver cómo varía el rendimiento con la frecuencia
                        body_counts = sorted(set(r.get("n_bodies", 0) for r in fixed_results))
                        for n_bodies in body_counts:
                            f.write(f"\n  N={n_bodies}:\n")
                            n_body_results = [r for r in fixed_results if r.get("n_bodies", 0) == n_bodies]
                            
                            for freq in freq_values:
                                freq_body_results = [r for r in n_body_results if r.get("reorder_frequency", 0) == freq]
                                if freq_body_results:
                                    avg_time = sum(r.get("total_time", 0) for r in freq_body_results) / len(freq_body_results)
                                    f.write(f"    Freq={freq}: {avg_time:.3f}s\n")
            
        print(f"📊 Reporte de análisis SFC generado en: {report_file}")
        
        # Generar también un archivo JSON para futuros análisis o visualizaciones
        analysis_data = {
            "algorithms": {},
            "summary": {
                "total_experiments": len(sfc_results),
                "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        for algo in algorithms:
            algo_results = [r for r in sfc_results if r["algorithm"] == algo]
            analysis_data["algorithms"][algo] = {
                "curves": {}
            }
            
            for curve_type in [0, 1]:  # Morton, Hilbert
                curve_name = "morton" if curve_type == 0 else "hilbert"
                curve_results = [r for r in algo_results if r["curve_type"] == curve_type]
                
                if not curve_results:
                    continue
                    
                analysis_data["algorithms"][algo]["curves"][curve_name] = {
                    "dynamic": {},
                    "fixed": {}
                }
                
                # Datos dinámicos
                dynamic_results = [r for r in curve_results if not r.get("fixed_reordering", False)]
                if dynamic_results:
                    analysis_data["algorithms"][algo]["curves"][curve_name]["dynamic"] = {
                        "avg_time": sum(r.get("total_time", 0) for r in dynamic_results) / len(dynamic_results),
                        "avg_frequency": sum(r.get("reorder_frequency", 0) for r in dynamic_results) / len(dynamic_results),
                        "by_body_count": {}
                    }
                    
                    body_counts = sorted(set(r.get("n_bodies", 0) for r in dynamic_results))
                    for n_bodies in body_counts:
                        n_body_results = [r for r in dynamic_results if r.get("n_bodies", 0) == n_bodies]
                        if n_body_results:
                            analysis_data["algorithms"][algo]["curves"][curve_name]["dynamic"]["by_body_count"][n_bodies] = {
                                "avg_frequency": sum(r.get("reorder_frequency", 0) for r in n_body_results) / len(n_body_results),
                                "avg_time": sum(r.get("total_time", 0) for r in n_body_results) / len(n_body_results)
                            }
                
                # Datos fijos
                fixed_results = [r for r in curve_results if r.get("fixed_reordering", False)]
                if fixed_results:
                    freq_values = sorted(set(r.get("reorder_frequency", 0) for r in fixed_results))
                    analysis_data["algorithms"][algo]["curves"][curve_name]["fixed"] = {
                        "by_frequency": {},
                        "by_body_count": {}
                    }
                    
                    for freq in freq_values:
                        freq_results = [r for r in fixed_results if r.get("reorder_frequency", 0) == freq]
                        if freq_results:
                            analysis_data["algorithms"][algo]["curves"][curve_name]["fixed"]["by_frequency"][freq] = {
                                "avg_time": sum(r.get("total_time", 0) for r in freq_results) / len(freq_results)
                            }
                    
                    body_counts = sorted(set(r.get("n_bodies", 0) for r in fixed_results))
                    for n_bodies in body_counts:
                        analysis_data["algorithms"][algo]["curves"][curve_name]["fixed"]["by_body_count"][n_bodies] = {}
                        
                        for freq in freq_values:
                            freq_body_results = [r for r in fixed_results if r.get("n_bodies", 0) == n_bodies and r.get("reorder_frequency", 0) == freq]
                            if freq_body_results:
                                analysis_data["algorithms"][algo]["curves"][curve_name]["fixed"]["by_body_count"][n_bodies][freq] = {
                                    "avg_time": sum(r.get("total_time", 0) for r in freq_body_results) / len(freq_body_results)
                                }
        
        # Guardar los datos JSON
        analysis_file = self.results_dir / "sfc_analysis_data.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis_data, f, indent=2)
            
        print(f"💾 Datos de análisis SFC guardados en: {analysis_file}")

def main():
    parser = argparse.ArgumentParser(description="Runner de experimentos CPU")
    parser.add_argument("--experiment-type", choices=["scalability", "threads", "sfc_frequency", "optimal_freq_vs_n", "all"], 
                       default="all", help="Tipo de experimento a ejecutar")
    parser.add_argument("--algorithm", choices=["barneshut", "directsum", "both"], 
                       default="both", help="Algoritmo a ejecutar (barneshut, directsum, o both)")
    parser.add_argument("--base-dir", default="./", help="Directorio base del proyecto")
    parser.add_argument("--test-mode", action="store_true", 
                       help="Usar configuración de test pequeña para pruebas rápidas")
    parser.add_argument("--no-energy", action="store_true",
                       help="Desactivar el cálculo de energía para acelerar los experimentos")
    
    args = parser.parse_args()
    
    # Determinar qué algoritmos ejecutar
    if args.algorithm == "both":
        selected_algorithms = ["barneshut", "directsum"]
    else:
        selected_algorithms = [args.algorithm]
    
    runner = CPUExperimentRunner(args.base_dir, args.no_energy, selected_algorithms)
    
    # Compilar algoritmos
    if not runner.compile_algorithms():
        print("❌ Falló la compilación, abortando experimentos")
        return
    
    # Usar configuraciones de test si se especifica
    if args.test_mode:
        print("🧪 Modo de prueba: usando configuración reducida")
        runner.generate_experiment_configurations = runner.generate_test_configurations
    
    # Mostrar estado del cálculo de energía
    if args.no_energy:
        print("⚡ Cálculo de energía desactivado para acelerar experimentos")
    
    all_results = []
    
    if args.experiment_type in ["scalability", "all"]:
        scalability_results = runner.run_scalability_experiments()
        all_results.extend(scalability_results)
    
    if args.experiment_type in ["threads", "all"]:
        thread_results = runner.run_thread_efficiency_experiments()
        all_results.extend(thread_results)
    
    if args.experiment_type in ["sfc_frequency", "all"]:
        sfc_frequency_results = runner.run_sfc_frequency_experiments()
        all_results.extend(sfc_frequency_results)
    
    if args.experiment_type in ["optimal_freq_vs_n", "all"]:
        optimal_freq_results = runner.run_optimal_frequency_vs_n_experiments()
        all_results.extend(optimal_freq_results)
        # Analizar los resultados inmediatamente
        runner.analyze_optimal_frequency_vs_n(optimal_freq_results)
    
    # Guardar solo un archivo JSON con todos los resultados
    if all_results:
        runner.save_results(all_results, "experiment_results")
        
        # Si tenemos resultados de frecuencia óptima, generar el análisis
        if any(r.get("experiment_type") == "optimal_frequency_vs_n" for r in all_results):
            runner.analyze_optimal_frequency_vs_n(all_results)
    
    print(f"\n✅ Experimentación completada! Resultados en: {runner.results_dir}")

if __name__ == "__main__":
    main() 