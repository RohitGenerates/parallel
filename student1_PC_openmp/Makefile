CXX = g++
CXXFLAGS = -O0 -fopenmp -std=c++11
SRC = src/main.cpp
OBJ = main.o
TARGET = packet_simulator

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) *.o results.csv
