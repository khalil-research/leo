# --- SYSTEM ---

SYSTEM     = x86-64_linux
LIBFORMAT  = static_pic
BASISDIR    = /home/rahul/opt
PLATFORM   = linux64


# --- DIRECTORIES ---

CCC = g++

BASISILOG  = $(BASISDIR)/ibm/ILOG/CPLEX_Studio1210
CPOPTDIR   = $(BASISILOG)/cpoptimizer
CONCERTDIR = $(BASISILOG)/concert
CPLEXDIR   = $(BASISILOG)/cplex
BOOSTDIR   = $(BASISDIR)/boost

GUROBIDIR = /opt/gurobi/$(PLATFORM)
GUROBIINCDIR = $(GUROBIDIR)/include
GUROBILIBDIR = $(GUROBIDIR)/lib


# --- FLAGS ---

CCOPT = -std=c++11 -m64 -fPIC -fno-strict-aliasing -fexceptions -DIL_STD -Wno-deprecated-declarations -Wno-ignored-attributes
CPLEXLIBDIR   = $(CPLEXDIR)/lib/$(SYSTEM)/$(LIBFORMAT)
CONCERTLIBDIR = $(CONCERTDIR)/lib/$(SYSTEM)/$(LIBFORMAT)
CPOPTLIBDIR = $(CPOPTDIR)/lib/$(SYSTEM)/$(LIBFORMAT)

CONCERTINCDIR = $(CONCERTDIR)/include
CPLEXINCDIR   = $(CPLEXDIR)/include
CPOPTINCDIR   = $(CPOPTDIR)/include

# --- OPTIMIZATION FLAGS ---

DEBUG_OPT = -DNDEBUG -O3
#DEBUG_OPT = -g3 -O0

#PROF = -pg
PROF =


# Flags
CFLAGS  = -I$(BOOSTDIR)/include -c $(PROF) $(DEBUG_OPT)
LDFLAGS = -lm -pthread

# Uncomment if using CPLEX/CPOPTIMIZER
CFLAGS  += $(CCOPT) -I$(CPLEXINCDIR) -I$(CONCERTINCDIR) -I$(CPOPTINCDIR) -I./include $(DEBUG_OPT) -I$(BOOSTDIR)/include -c $(PROF)
LDFLAGS += -L$(CPOPTLIBDIR) -lcp -L$(CPLEXLIBDIR) -lilocplex -lcplex -L$(CONCERTLIBDIR) -lconcert -ldl

# Uncomment if using Gurobi
#CFLAGS += -I$(GUROBIINCDIR)
#LDFLAGS += -L$(GUROBILIBDIR) -lgurobi_c++ -lgurobi75 -lm 


# --- APPLICATION-SPECIFIC OPTIONS ---

# number of objective functions
NUM_OBJS=3
CFLAGS += -DNOBJS=$(NUM_OBJS)


# ---- COMPILE  ----
SRC_DIR   := src
OBJ_DIR   := obj

SRC_DIRS  := $(shell find $(SRC_DIR) -type d)
OBJ_DIRS  := $(addprefix $(OBJ_DIR)/,$(SRC_DIRS))

SOURCES   := $(shell find $(SRC_DIR) -name '*.cpp')
OBJ_FILES := $(addprefix $(OBJ_DIR)/, $(SOURCES:.cpp=.o))

vpath %.cpp $(SRC_DIRS)

# ---- TARGETS ----

EXECUTABLE=multiobj

all: $(EXECUTABLE)

$(EXECUTABLE): makedir $(SOURCES) $(OBJ_FILES) 
	$(CCC) $(OBJ_FILES) $(LDFLAGS) $(PROF) -o $@

$(OBJ_DIR)/%.o: %.cpp
	$(CCC) $(CFLAGS) $< -o $@

makedir: $(OBJ_DIRS)

$(OBJ_DIRS):
	@mkdir -p $@

clean:
	@rm -rf obj 
	@rm -rf $(EXECUTABLE)
