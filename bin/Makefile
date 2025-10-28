# Compiler
CXX := g++

# Compiler and include flags
CXXFLAGS := -std=c++17 -Wall -Wextra \
    `pkg-config --cflags opencv4` \
    -I/opt/homebrew/opt/nlohmann-json/include

# Linker flags
LDFLAGS := `pkg-config --libs opencv4`

# Source and object files
SRC_DIR := ..
SRCS := $(wildcard $(SRC_DIR)/*.cpp)
OBJS := $(patsubst $(SRC_DIR)/%.cpp, %.o, $(SRCS))

# Output executable name
TARGET := flight_processor

# Default build rule
all: $(TARGET)

# Link object files into the final binary
$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS)

# Compile .cpp files into .o files
%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean rule
clean:
	rm -f $(OBJS) $(TARGET)


