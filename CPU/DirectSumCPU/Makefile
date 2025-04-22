CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -fopenmp -Wall -Wextra
TARGET = main
SOURCE = main.cpp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(TARGET) 