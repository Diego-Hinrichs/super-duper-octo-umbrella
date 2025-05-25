CXX = g++
CXXFLAGS = -std=c++17 -O3 -fopenmp
INCLUDES = -I../../

SOURCES = main.cpp
OUTPUT = main

.PHONY: all clean

all: $(OUTPUT)

$(OUTPUT): $(SOURCES)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(OUTPUT) $(SOURCES)

clean:
	rm -f $(OUTPUT) 