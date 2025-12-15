# Compiler settings
# CXX = g++
CXX = /opt/homebrew/bin/g++-15
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra
LDFLAGS = -L/opt/homebrew/lib/ -ltbb
BIN = bin

# Target executable
TARGET = monte_carlo_sim

# Source files
SOURCES = $(wildcard src/*.cpp)
OBJECTS = $(SOURCES:.cpp=.o)
HEADERS = montecarlo_gbm.h

# Default target
all: bin $(BIN)/$(TARGET)

# Link object files to create executable
$(BIN)/$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "Build complete! Run with: ./$(BIN)/$(TARGET)"

# Compile source files to object files
%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $<

# Clean build artifacts
clean:
	rm -rf $(OBJECTS) $(BIN)
	@echo "Cleaned build files"

# Run the program
run: $(BIN)/$(TARGET)
	./$(BIN)/$(TARGET)

bin:
	@mkdir -p $(BIN)

# Debug build
debug: CXXFLAGS += -g -DDEBUG
debug: clean $(BIN)/$(TARGET)

.PHONY: all clean run debug
