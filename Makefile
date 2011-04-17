CC = nvcc 
CXX = nvcc 
NVCC = nvcc
FLAGS = -I./src/headers -I./src -I./common 
LDFLAGS =
CCFLAGS = -g
CXXFLAGS = -g
NVCCFLAGS = -g 

OBJECT_DIRECTORY := common/obj
SOURCE_DIRECTORY := src
DEMO_DIRECTORY := $(SOURCE_DIRECTORY)/demos
TEST_DIRECTORY := $(SOURCE_DIRECTORY)/tests
VPATH := $(OBJECT_DIRECTORY) $(SOURCE_DIRECTORY) $(DEMO_DIRECTORY) $(TEST_DIRECTORY)

OBJECTS := $(patsubst $(SOURCE_DIRECTORY)/%.cu, $(OBJECT_DIRECTORY)/%.o, $(wildcard $(SOURCE_DIRECTORY)/*.cu))
DEMO_OBJECTS := $(OBJECTS) $(patsubst $(DEMO_DIRECTORY)/%.cu, $(OBJECT_DIRECTORY)/%.o, $(wildcard $(DEMO_DIRECTORY)/*.cu))
TEST_OBJECTS := $(OBJECTS) $(patsubst $(TEST_DIRECTORY)/%.cu, $(OBJECT_DIRECTORY)/%.o, $(wildcard $(TEST_DIRECTORY)/*.cu))

DEMO := demo
TEST := test

# Similar for OPENCC_FLAGS and PTXAS_FLAGS. 
# These are simply passed via the environment: 
 #
export OPENCC_FLAGS := 
export PTXAS_FLAGS  := -fastimul 
#
# cuda and C/C++ compilation rules, with 
# dependency generation: 
#
$(OBJECT_DIRECTORY)/%.o : %.cpp
	$(CXX) -c $^ $(CXXFLAGS) $(FLAGS) -o $@ 
	$(CXX) -M $^ $(CXXFLAGS) $(FLAGS) > $@.dep 
$(OBJECT_DIRECTORY)/%.o : %.c 
	$(CC) -c $^ $(CCFLAGS) $(FLAGS) -o $@ 
	$(CC) -M $^ $(CCFLAGS) $(FLAGS) > $@.dep 
$(OBJECT_DIRECTORY)/%.o : %.cu 
	$(NVCC) -c $^ $(NVCCFLAGS) $(FLAGS) -o $@ 
	$(NVCC) -M $^ $(NVCCFLAGS) $(FLAGS)  > $@.dep
#
# Pick up generated dependency files, and
# add /dev/null because gmake does not consider 
# an empty list to be a list: 
#
include  $(wildcard *.dep) /dev/null 
#
# Define the application;  
# for each object file, there must be a 
# corresponding .c or .cpp or .cu file: 
#

$(DEMO) : $(DEMO_OBJECTS) 
	$(NVCC) $(DEMO_OBJECTS) $(LDFLAGS) -o $@ 

$(TEST) : $(TEST_OBJECTS)
	$(NVCC) $(TEST_OBJECTS) $(LDFLAGS) -o $@

clean :  
	$(RM) $(DEMO_OBJECTS) $(TEST_OBJECTS) $(OBJECT_DIRECTORY)/*.dep $(DEMO) $(TEST) 



