#!/usr/bin/env python3
"""
Script para automatizar la experimentación de algoritmos GPU (BarnesHut y DirectSum)
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

class GPUExperimentRunner:
    def __init__(self, base_dir="./", no_energy=False, algorithm=None):
        self.base_dir = Path(base_dir)
        self.results_dir = self.base_dir / "results" / f"gpu_experiments_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.no_energy = no_energy
        
        # Algoritmos disponibles
        self.algorithms = {
            "barneshut": {
                "path": self.base_dir / "BarnesHut" / "BarnesHutGPU",
                "executable": "main"
            },
            "directsum": {
                "path": self.base_dir / "DirectSum" / "DirectSumGPU", 
                "executable": "main"
            }
        }
        
        # Filtrar algoritmos si se especifica uno específico
        if algorithm and algorithm in self.algorithms:
            self.algorithms = {algorithm: self.algorithms[algorithm]}
        
    def compile_algorithms(self):
        """Compila todos los algoritmos"""
        print("🔨 Compilando algoritmos GPU...")
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
        """Genera todas las configuraciones de experimentos para GPU"""
        # Configuraciones base para explorar (ampliadas para GPU)
        configs = {
            # Número de cuerpos - escalabilidad (EXACTAMENTE la misma escala que CPU)
            "n_bodies": [1000, 5000, 10000, 20000, 50000],
            
            # Configuraciones de bloque CUDA
            "block_sizes": [256],
            
            # Configuraciones de SFC (EXACTAMENTE las mismas que CPU)
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
            
            # Distribuciones de masa (EXACTAMENTE las mismas que CPU)
            "mass_distributions": [0, 1],  # UNIFORM, NORMAL
            
            # Número de iteraciones para cada test (EXACTAMENTE las mismas que CPU)
            "iterations": [100],
            
            # Seeds para reproducibilidad (EXACTAMENTE las mismas que CPU)
            "seeds": [42],
            
            # Parámetro theta para Barnes-Hut (solo aplica a Barnes-Hut)
            "theta_values": [0.5]
        }
        
        return configs
    
    def run_single_experiment(self, algorithm, config):
        """Ejecuta un experimento individual"""
        algo_config = self.algorithms[algorithm]
        
        # Construir comando usando los parámetros reales
        cmd = [
            f"./{algo_config['executable']}",
            "-n", str(config["n_bodies"]),
            "-block", str(config["block_size"]),
            "-s", str(config["iterations"]),
            "-seed", str(config["seed"]),
            "-mass", "uniform" if config["mass_dist"] == 0 else "normal"
        ]
        
        # Agregar theta para Barnes-Hut
        if algorithm == "barneshut" and "theta" in config:
            cmd.extend(["-theta", str(config["theta"])])
        
        # Agregar parámetros SFC
        if not config["use_sfc"]:
            cmd.append("-nosfc")
        else:
            cmd.extend(["-curve", "morton" if config["curve_type"] == 0 else "hilbert"])
            # Añadir parámetros de reordenamiento fijo si es aplicable
            cmd.extend(["-fixreorder", "1" if config["fixed_reordering"] else "0"])
            cmd.extend(["-freq", str(config["reorder_frequency"]) if config["reorder_frequency"] > 0 else "10"])
            
        # Desactivar cálculo de energía si se especifica (usando -energy 0 en lugar de -noenergy)
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
                text=True,
                # Establecer un timeout razonable para experimentos GPU (2 horas)
                timeout=7200
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
        except subprocess.TimeoutExpired:
            print(f"    ⏱️ Timeout en experimento después de 2 horas")
            return None

    def parse_output(self, output):
        """Parsea la salida del programa para extraer métricas"""
        metrics = {}
        lines = output.strip().split('\n')
        steps = 100  # valor por defecto
        
        for line in lines:
            line = line.strip()
            
            # Extraer número de steps primero
            if "Iterations:" in line:
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
                
            elif "Force calculation:" in line:
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
                
            elif "Potential Energy:" in line and "Average" not in line:
                value_str = line.split(':')[1].strip()
                metrics["potential_energy"] = float(value_str)
                
            elif "Kinetic Energy:" in line and "Average" not in line:
                value_str = line.split(':')[1].strip()
                metrics["kinetic_energy"] = float(value_str)
                
            elif "Block size:" in line:
                # Extraer tamaño de bloque real
                value_str = line.split(':')[1].strip()
                metrics["actual_block_size"] = int(value_str)
                
            elif "Fixed reordering:" in line:
                value_str = line.split(':')[1].strip()
                metrics["is_fixed_reordering"] = value_str.lower() == "yes"
                
            elif "Optimal reorder freq:" in line:
                value_str = line.split(':')[1].strip()
                metrics["reorder_frequency"] = int(value_str)
                
            elif "Min time:" in line:
                value_str = line.split(':')[1].strip().replace('ms', '')
                metrics["min_time"] = float(value_str)
                
            elif "Max time:" in line:
                value_str = line.split(':')[1].strip().replace('ms', '')
                metrics["max_time"] = float(value_str)
        
        return metrics

    def run_scalability_experiments(self):
        """Ejecuta experimentos de escalabilidad en GPU"""
        print("🚀 Ejecutando experimentos de escalabilidad en GPU...")
        
        configs = self.generate_experiment_configurations()
        results = []
        
        # Test de escalabilidad por número de cuerpos
        for algorithm in self.algorithms.keys():
            print(f"\n📊 Algoritmo: {algorithm.upper()}")
            
            for n_bodies in configs["n_bodies"]:
                for block_size in configs["block_sizes"]:
                    # Theta solo aplica para Barnes-Hut
                    theta_values = configs["theta_values"] if algorithm == "barneshut" else [None]
                    
                    for theta in theta_values:
                        for sfc_config in configs["sfc_configs"]:
                            for mass_dist in configs["mass_distributions"]:
                                for seed in configs["seeds"]:
                                    # Configurar experimento
                                    config = {
                                        "n_bodies": n_bodies,
                                        "block_size": block_size,
                                        "iterations": configs["iterations"][0],
                                        "seed": seed,
                                        "mass_dist": mass_dist,
                                        "no_energy": hasattr(self, 'no_energy') and self.no_energy,
                                        **sfc_config
                                    }
                                    
                                    # Agregar theta solo para Barnes-Hut
                                    if algorithm == "barneshut" and theta is not None:
                                        config["theta"] = theta
                                    
                                    sfc_info = f"SFC={sfc_config['use_sfc']}, Curve={sfc_config['curve_type']}"
                                    if sfc_config['use_sfc']:
                                        sfc_info += f", Fixed={sfc_config['fixed_reordering']}"
                                        if sfc_config['fixed_reordering']:
                                            sfc_info += f", Freq={sfc_config['reorder_frequency']}"
                                    
                                    theta_info = f", Theta={theta}" if algorithm == "barneshut" and theta is not None else ""
                                    
                                    print(f"  🔬 N={n_bodies}, Block={block_size}{theta_info}, {sfc_info}, Mass={mass_dist}")
                                    
                                    result = self.run_single_experiment(algorithm, config)
                                    if result:
                                        results.append(result)
                                        print(f"    ✅ Tiempo total: {result.get('total_time', 'N/A')}s")
        
        return results

    def run_block_size_experiments(self):
        """Ejecuta experimentos específicos para analizar el impacto de diferentes tamaños de bloque CUDA"""
        print("🧩 Ejecutando experimentos de tamaño de bloque CUDA...")
        
        results = []
        # Usar EXACTAMENTE la misma configuración que CPU (equivalente a thread efficiency)
        n_bodies = 20000   # EXACTAMENTE el mismo que CPU usa para thread efficiency
        iterations = 50    # EXACTAMENTE el mismo que CPU
        
        # Probar diferentes tamaños de bloque con más detalle
        block_sizes = [32, 64, 128, 192, 256, 384, 512, 768, 1024]
        
        # Configuraciones de SFC a probar (EXACTAMENTE las mismas que CPU thread efficiency)
        sfc_configs = [
            {"use_sfc": True, "curve_type": 0, "fixed_reordering": False, "reorder_frequency": 0},  # Morton dinámico
            {"use_sfc": True, "curve_type": 0, "fixed_reordering": True, "reorder_frequency": 10},  # Morton fijo freq=10
            {"use_sfc": True, "curve_type": 0, "fixed_reordering": True, "reorder_frequency": 20},  # Morton fijo freq=20
            {"use_sfc": True, "curve_type": 0, "fixed_reordering": True, "reorder_frequency": 50},  # Morton fijo freq=50
        ]
        
        for algorithm in self.algorithms.keys():
            print(f"\n📊 Algoritmo: {algorithm.upper()}")
            
            # Theta solo para Barnes-Hut (EXACTAMENTE el mismo valor que optimal_frequency_vs_n)
            theta_value = 0.5 if algorithm == "barneshut" else None
            
            for block_size in block_sizes:
                for sfc_config in sfc_configs:
                    # Configurar experimento
                    config = {
                        "n_bodies": n_bodies,
                        "block_size": block_size,
                        "iterations": iterations,
                        "seed": 42,
                        "mass_dist": 0,  # UNIFORM
                        "no_energy": hasattr(self, 'no_energy') and self.no_energy,
                        **sfc_config
                    }
                    
                    # Agregar theta solo para Barnes-Hut
                    if algorithm == "barneshut" and theta_value is not None:
                        config["theta"] = theta_value
                    
                    sfc_info = f"SFC={sfc_config['use_sfc']}, Curve={sfc_config['curve_type']}"
                    if sfc_config['use_sfc']:
                        sfc_info += f", Fixed={sfc_config['fixed_reordering']}"
                        if sfc_config['fixed_reordering']:
                            sfc_info += f", Freq={sfc_config['reorder_frequency']}"
                    
                    print(f"  🔬 Block Size: {block_size}, {sfc_info}")
                    
                    result = self.run_single_experiment(algorithm, config)
                    if result:
                        results.append(result)
                        print(f"    ✅ Tiempo total: {result.get('total_time', 'N/A')}s")
        
        return results

    def run_sfc_frequency_experiments(self):
        """Ejecuta experimentos específicos para analizar el impacto de diferentes frecuencias de reordenamiento"""
        print("🔄 Ejecutando experimentos de frecuencia de reordenamiento SFC en GPU...")
        
        results = []
        # Usar EXACTAMENTE la misma configuración que CPU
        n_bodies = 10000   # EXACTAMENTE el mismo que CPU
        iterations = 100   # EXACTAMENTE el mismo que CPU
        block_size = 256   # Tamaño de bloque fijo (equivalente a threads=4 en CPU)
        
        # Probar diferentes frecuencias de reordenamiento (EXACTAMENTE las mismas que CPU)
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
            
            # Theta solo para Barnes-Hut (EXACTAMENTE el mismo valor que optimal_frequency_vs_n)
            theta_value = 0.5 if algorithm == "barneshut" else None
            
            for sfc_config in freq_configs:
                # Configurar experimento
                config = {
                    "n_bodies": n_bodies,
                    "block_size": block_size,
                    "iterations": iterations,
                    "seed": 42,
                    "mass_dist": 0,  # UNIFORM
                    "no_energy": hasattr(self, 'no_energy') and self.no_energy,
                    **sfc_config
                }
                
                # Agregar theta solo para Barnes-Hut
                if algorithm == "barneshut" and theta_value is not None:
                    config["theta"] = theta_value
                
                curve_name = "Morton" if sfc_config["curve_type"] == 0 else "Hilbert"
                reorder_info = "Dinámico" if not sfc_config["fixed_reordering"] else f"Fijo-{sfc_config['reorder_frequency']}"
                
                print(f"  🔬 {curve_name}, Reordenamiento: {reorder_info}")
                
                result = self.run_single_experiment(algorithm, config)
                if result:
                    results.append(result)
                    print(f"    ✅ Tiempo total: {result.get('total_time', 'N/A')}s, Freq: {result.get('reorder_frequency', 'N/A')}")
        
        return results

    def run_optimal_frequency_vs_n_experiments(self):
        """Ejecuta experimentos para analizar la frecuencia óptima vs número de cuerpos (N) en GPU"""
        print("📈 Ejecutando experimentos de frecuencia óptima vs N en GPU...")
        
        results = []
        block_size = 256  # Tamaño de bloque fijo para este análisis
        iterations = 100  # Iteraciones fijas según especificación
        
        # 20+ PUNTOS DE N para una curva muy suave (EXACTAMENTE la misma escala que CPU)
        n_bodies_list = [
            1000, 1200, 1500, 1800, 2000, 2500, 3000, 3500, 4000, 4500,
            5000, 6000, 7000, 8000, 9000, 10000, 12000, 15000, 18000, 20000,
            25000, 30000, 35000, 40000, 45000, 50000
        ]
        
        # Frecuencias específicas solicitadas (mismas que CPU)
        frequencies_to_test = [5, 10, 15, 20, 50]
        
        for algorithm in self.algorithms.keys():
            print(f"\n📊 Algoritmo: {algorithm.upper()}")
            
            for n_bodies in n_bodies_list:
                print(f"\n  🔬 Analizando N={n_bodies}...")
                
                for curve_type in [0, 1]:  # Morton y Hilbert
                    curve_name = "Morton" if curve_type == 0 else "Hilbert"
                    print(f"\n    📍 Curva: {curve_name}")
                    
                    # 1. NORMAL: Sin SFC (baseline)
                    normal_config = {
                        "n_bodies": n_bodies,
                        "block_size": block_size,
                        "iterations": iterations,
                        "seed": 42,
                        "mass_dist": 0,  # UNIFORM
                        "no_energy": hasattr(self, 'no_energy') and self.no_energy,
                        "use_sfc": False,
                        "curve_type": curve_type,
                        "fixed_reordering": False,
                        "reorder_frequency": 0
                    }
                    
                    # Agregar theta solo para Barnes-Hut
                    if algorithm == "barneshut":
                        normal_config["theta"] = 0.5
                    
                    print(f"      🚫 Normal (Sin SFC)")
                    normal_result = self.run_single_experiment(algorithm, normal_config)
                    if normal_result:
                        normal_result["experiment_type"] = "optimal_frequency_vs_n"
                        normal_result["sfc_configuration"] = "normal"
                        results.append(normal_result)
                        normal_time = normal_result.get('total_time', 0)
                        print(f"        ✅ Tiempo: {normal_time:.3f}s")
                    
                    # 2. Probar frecuencias específicas
                    best_fixed_freq = None
                    best_fixed_time = float('inf')
                    
                    print(f"      🔍 Probando frecuencias: {frequencies_to_test}")
                    for freq in frequencies_to_test:
                        fixed_config = {
                            "n_bodies": n_bodies,
                            "block_size": block_size,
                            "iterations": iterations,
                            "seed": 42,
                            "mass_dist": 0,  # UNIFORM
                            "no_energy": hasattr(self, 'no_energy') and self.no_energy,
                            "use_sfc": True,
                            "curve_type": curve_type,
                            "fixed_reordering": True,
                            "reorder_frequency": freq
                        }
                        
                        # Agregar theta solo para Barnes-Hut
                        if algorithm == "barneshut":
                            fixed_config["theta"] = 0.5
                        
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
                            "block_size": block_size,
                            "iterations": iterations,
                            "seed": 42,
                            "mass_dist": 0,
                            "no_energy": hasattr(self, 'no_energy') and self.no_energy,
                            "use_sfc": True,
                            "fixed_reordering": True,
                            "reorder_frequency": best_fixed_freq,
                            "total_time": best_fixed_time,
                            "experiment_type": "optimal_frequency_vs_n",
                            "sfc_configuration": "sfc_opt"
                        }
                        
                        # Agregar theta solo para Barnes-Hut
                        if algorithm == "barneshut":
                            sfc_opt_result["theta"] = 0.5
                        
                        results.append(sfc_opt_result)
                        print(f"      🎯 SFC-Opt (freq={best_fixed_freq}): {best_fixed_time:.3f}s")
                        
                        # Speedup calculation
                        if normal_result:
                            normal_time = normal_result.get('total_time', 0)
                            speedup = normal_time / best_fixed_time if best_fixed_time > 0 else 0
                            print(f"        📊 Speedup vs Normal: {speedup:.2f}x")
        
        return results

    def analyze_optimal_frequency_vs_n(self, results):
        """Analiza los resultados de frecuencia óptima vs N y genera gráficos para GPU"""
        print("\n📊 Analizando resultados de frecuencia óptima vs N en GPU...")
        
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
        """Guarda resultados en CSV y JSON"""
        if not results:
            print("⚠️ No hay resultados para guardar")
            return
        
        # Recopilar todos los campos únicos de todos los resultados
        all_fieldnames = set()
        for result in results:
            all_fieldnames.update(result.keys())
        all_fieldnames = sorted(list(all_fieldnames))
        
        # Guardar como CSV
        csv_file = self.results_dir / f"{filename}.csv"
        with open(csv_file, 'w', newline='') as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=all_fieldnames)
                writer.writeheader()
                writer.writerows(results)
        
        # Guardar como JSON para análisis posterior
        json_file = self.results_dir / f"{filename}.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"💾 Resultados guardados en: {csv_file} y {json_file}")
    
    def generate_summary_report(self, results):
        """Genera un reporte resumen de los experimentos"""
        if not results:
            return
            
        report_file = self.results_dir / "experiment_summary.txt"
        
        with open(report_file, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("REPORTE DE EXPERIMENTACIÓN GPU\n")
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
                
                # Análisis por tamaño de bloque
                block_sizes = sorted(set(r.get("block_size", 0) for r in algo_results))
                if block_sizes:
                    f.write("\nRendimiento por tamaño de bloque:\n")
                    for block in block_sizes:
                        block_results = [r for r in algo_results if r.get("block_size", 0) == block]
                        if block_results:
                            avg_block_time = sum(r.get("total_time", 0) for r in block_results) / len(block_results)
                            f.write(f"  Block={block}: {avg_block_time:.3f}s\n")
                
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
            f.write("ANÁLISIS DE RENDIMIENTO DE CONFIGURACIONES SFC EN GPU\n")
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
        
        print(f"📊 Reporte de análisis SFC generado en: {report_file}")
        
        # Generar también un archivo JSON para futuros análisis o visualizaciones
        analysis_data = {
            "algorithms": {},
            "summary": {
                "total_experiments": len(sfc_results),
                "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
        }
        
        # Guardar los datos JSON
        analysis_file = self.results_dir / "sfc_analysis_data.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis_data, f, indent=2)
            
        print(f"💾 Datos de análisis SFC guardados en: {analysis_file}")


def main():
    parser = argparse.ArgumentParser(description="Runner de experimentos GPU")
    parser.add_argument("--experiment-type", choices=["scalability", "block_size", "sfc_frequency", "optimal_frequency_vs_n", "all"], 
                       default="all", help="Tipo de experimento a ejecutar")
    parser.add_argument("--base-dir", default="./", help="Directorio base del proyecto")
    parser.add_argument("--test-mode", action="store_true", 
                       help="Usar configuración de test pequeña para pruebas rápidas")
    parser.add_argument("--no-energy", action="store_true",
                       help="Desactivar el cálculo de energía para acelerar los experimentos")
    parser.add_argument("--algorithm", choices=["directsum", "barneshut", "all"], default="all",
                       help="Algoritmo específico a ejecutar (directsum, barneshut o ambos)")
    
    args = parser.parse_args()
    
    # Determinar qué algoritmo ejecutar
    algorithm = None if args.algorithm == "all" else args.algorithm
    
    runner = GPUExperimentRunner(args.base_dir, args.no_energy, algorithm)
    
    # Compilar algoritmos
    if not runner.compile_algorithms():
        print("❌ Falló la compilación, abortando experimentos")
        return
    
    
    # Mostrar estado del cálculo de energía
    if args.no_energy:
        print("⚡ Cálculo de energía desactivado para acelerar experimentos")
        
    # Mostrar algoritmos a ejecutar
    if args.algorithm != "all":
        print(f"🔬 Ejecutando sólo el algoritmo: {args.algorithm.upper()}")
    
    all_results = []
    
    if args.experiment_type in ["scalability", "all"]:
        scalability_results = runner.run_scalability_experiments()
        all_results.extend(scalability_results)
        runner.save_results(scalability_results, "scalability_experiments")
    
    if args.experiment_type in ["block_size", "all"]:
        block_size_results = runner.run_block_size_experiments()
        all_results.extend(block_size_results)
        runner.save_results(block_size_results, "block_size_experiments")
    
    if args.experiment_type in ["sfc_frequency", "all"]:
        sfc_frequency_results = runner.run_sfc_frequency_experiments()
        all_results.extend(sfc_frequency_results)
        runner.save_results(sfc_frequency_results, "sfc_frequency_experiments")
    
    if args.experiment_type in ["optimal_frequency_vs_n", "all"]:
        optimal_frequency_vs_n_results = runner.run_optimal_frequency_vs_n_experiments()
        all_results.extend(optimal_frequency_vs_n_results)
        runner.save_results(optimal_frequency_vs_n_results, "optimal_frequency_vs_n_experiments")
    
    if all_results:
        runner.save_results(all_results, "all_experiments")
        runner.generate_summary_report(all_results)
        runner.generate_sfc_analysis_report(all_results)
        
        # Si tenemos resultados de frecuencia óptima, generar el análisis
        if any(r.get("experiment_type") == "optimal_frequency_vs_n" for r in all_results):
            runner.analyze_optimal_frequency_vs_n(all_results)
    
    print(f"\n✅ Experimentación completada! Resultados en: {runner.results_dir}")

if __name__ == "__main__":
    main()
