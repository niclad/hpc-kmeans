TEST_TARGET = kmeans_mpi
CC = mpic++
CVER = -std=c++11
CFLAGS = -Wall -g -O1
objects = $(TEST_TARGET).o readData.o tools.o kmeans.o

test: $(objects)
	$(CC) $(CVER) $(CFLAGS) -o $(TEST_TARGET) $(objects)

readData.o: readData.cpp readData.h
	$(CC) $(CVER) $(CFLAGS) -c readData.cpp

tools.o: tools.cpp tools.h
	$(CC) $(CVER) $(CFLAGS) -c tools.cpp

kmeans.o: kmeans.cpp kmeans.h
	$(CC) $(CVER) $(CFLAGS) -c kmeans.cpp

$(TEST_TARGET).o: $(TEST_TARGET).cpp
	$(CC) $(CVER) $(CFLAGS) -c $(TEST_TARGET).cpp


clean:
	rm $(objects) *.out