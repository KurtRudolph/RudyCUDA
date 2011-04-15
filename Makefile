CC = gcc
CXX = g++
NVCC = nvcc
FLAGS = -I./common -I./src
LDFLAGS =
CCFLAGS =
CXXFLAGS = 
NVCCFLAGS = -g 

OBJECT_DIRECTORY := obj
SOURCE_DIRECTORY := src
DEMO_DIRECTORY := $(addprefix $(SOURCE_DIRECTORY)/,demo)
TEST_DIRECOTRY := $(addprefix $(SOURCE_DIRECTORY)/,test)
VPATH := $(OBJECT_DIRECTORY) $(SOURCE_DIRECTORY) $(DEMO_DIRECTORY) $(TEST_DIRECTORY)

OBJECTS := $(patsubst $(SOURCE_DIRECTORY)/%.cu, $(OBJECT_DIRECTORY)/%.o, $(wildcard $(SOURCE_DIRECTORY)/*.cu))
DEMO_OBJECTS := $(patsubst $(DEMO_DIRECTORY)/%.cu, $(OBJECT_DIRECTORY)/%.o, $(wildcard $(DEMO_DIRECTORY)/*.cu))
TEST_OBJECTS := $(patsubst $(TEST_DIRECTORY)/%.cu, $(OBJECT_DIRECTORY)/%.o, $(wildcard $(TEST_DIRECTORY)/*.cu))

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
	$(NVCC) -c $^ $(NVCCFLAGS) $(FLAGS) -o $@ 
	$(NVCC) -M $^ $(NVCCFLAGS) $(FLAGS) > $@.dep 
$(OBJECT_DIRECTORY)/%.o : %.c 
	$(NVCC) -c $^ $(CCFLAGS) $(FLAGS) -o $@ 
	$(NVCC) -M $^ $(CCFLAGS) $(FLAGS) > $@.dep 
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

$(DEMO) : $(OBJECTS) $(DEMO_OBJECTS)
	$(NVCC) $(OBJECTS) $(DEMO_OBJECTS) $(LDFLAGS) $< -o $@ 

$(TEST) : $(OBJECTS) $(TEST_OBJECTS)
	$(NVCC) $(LDFLAGS) -o $@

clean :  
	$(RM) $(OBJECTS) $(DEMO_OBJECTS) $(TEST_OBJECTS) $(OBJECT_DIRECTORY)/*.dep $(DEMO) $(TEST) 



