# Compiler settings
CXX = /opt/homebrew/bin/g++-15
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra
LDFLAGS = -L/opt/homebrew/lib/ -ltbb

# output dictionary
BIN = bin
DATA = data

# Target executable
TARGET = monte_carlo_sim

# Source files (auto-detect from src/)
SOURCES = $(wildcard src/*.cpp)
OBJECTS = $(SOURCES:.cpp=.o)
HEADERS = montecarlo_gbm.h csv_reader.h european_option.h regime_switch.h portfolio.h

# Default target
all: bin data app

# Link object files to create executable
app: $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $(BIN)/$(TARGET) $^ $(LDFLAGS)
	@echo "Build complete! Run with: ./$(TARGET)"

# Compile source files to object files
%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $<

# Clean build artifacts
clean:
	rm -rf $(OBJECTS) $(BIN)
	@echo "Cleaned build files"

# Clean everything including generated data
clean-all: clean
	rm -f $(DATA)/mc_*.csv
	@echo "Cleaned all files including generated results"

# Run the program
run:
	./$(BIN)/$(TARGET)

bin:
	@mkdir -p $(BIN)

data:
	@mkdir -p $(DATA)

# Debug build
debug: CXXFLAGS += -g -DDEBUG
debug: clean $(BIN)/$(TARGET)

.PHONY: all clean clean-all run debug
