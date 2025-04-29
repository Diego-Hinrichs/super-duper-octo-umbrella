# Makefile para compilar SFCDirectSumGPU en un solo archivo

# Compilador CUDA
NVCC = nvcc

# Flags de compilación
NVCC_FLAGS = -O3

# Buscar un compilador compatible
# CUDA 12.x soporta oficialmente hasta GCC 12
HOST_COMPILER := $(shell which g++-11 2>/dev/null || which g++-10 2>/dev/null || which g++-9 2>/dev/null || which g++-8 2>/dev/null || echo "g++")

# Objetivo principal
all: 
	@echo "Usando compilador host: $(HOST_COMPILER)"
	$(MAKE) main

# Compilar el ejecutable
main: main.cu
	$(NVCC) $(NVCC_FLAGS) -ccbin=$(HOST_COMPILER) -o $@ $<

# Limpiar archivos generados
clean:
	rm -f main

.PHONY: all clean 