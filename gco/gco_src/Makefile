CC=g++
CFLAGS=-fPIC

all: libgco.so

libgco.so: LinkedBlockList.o graph.o maxflow.o GCoptimization.o
	$(CC) -shared $(CFLAGS) LinkedBlockList.o graph.o maxflow.o GCoptimization.o -o libgco.so

LinkedBlockList.o: LinkedBlockList.cpp LinkedBlockList.h
	$(CC) $(CFLAGS) -c LinkedBlockList.cpp -o LinkedBlockList.o

graph.o: graph.cpp graph.h block.h
	$(CC) $(CFLAGS) -c graph.cpp -o graph.o

maxflow.o: block.h graph.h maxflow.cpp
	$(CC) $(CFLAGS) -c maxflow.cpp -o maxflow.o

GCoptimization.o: GCoptimization.cpp GCoptimization.h LinkedBlockList.h energy.h graph.h graph.o maxflow.o
	$(CC) $(CFLAGS) -c GCoptimization.cpp -o GCoptimization.o

linetest: linetest.cpp
	$(CC) -L. linetest.cpp -o linetest -lgco

gco_linenet: gco_linenet.cpp
	$(CC) -L. gco_linenet.cpp -o gco_linenet -lgco

clean:
	rm -f *.o

rm:
	rm -f *.o *.so linetest gco_linenet

