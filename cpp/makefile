# --- SYSTEM ---

SYSTEM     = x86-64_linux
LIBFORMAT  = static_pic
BASISDIR    = /opt

# --- DIRECTORIES ---

CCC = g++
BASISILOG  = $(BASISDIR)/ibm/ILOG/CPLEX_Studio1210
CPOPTDIR   = $(BASISILOG)/cpoptimizer
CONCERTDIR = $(BASISILOG)/concert
CPLEXDIR   = $(BASISILOG)/cplex
BOOSTDIR   = $(BASISDIR)/boost

# --- FLAGS ---

CCOPT = -m64 -fPIC -fno-strict-aliasing -fexceptions -DIL_STD -Wno-deprecated-declarations -std=c++17
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

CFLAGS += $(CCOPT) -I$(CPLEXINCDIR) -I$(CONCERTINCDIR) -I$(CPOPTINCDIR) -I./include $(DEBUG_OPT) -I$(BOOSTDIR)/include -c $(PROF)
# CFLAGS += $(CCOPT) -I./include $(DEBUG_OPT) -I$(BOOSTDIR)/include -c $(PROF)

LDFLAGS = -L$(CPOPTLIBDIR) -lcp -L$(CPLEXLIBDIR) -lilocplex -lcplex -L$(CONCERTLIBDIR) -lconcert -lm -ldl -pthread 
# LDFLAGS = -lm -pthread 


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
