TEST_TARGET = kmeans_cuda
CC = nvcc
CVER = -std=c++11
CFLAGS = -g -O1
objects = $(TEST_TARGET).o readData.o tools.o

test: $(objects)
	$(CC) $(CVER) $(CFLAGS) -o $(TEST_TARGET) $(objects)

readData.o: readData.cu readData.h
	$(CC) $(CVER) $(CFLAGS) -c readData.cu

tools.o: tools.cu tools.h
	$(CC) $(CVER) $(CFLAGS) -c tools.cu

$(TEST_TARGET).o: $(TEST_TARGET).cu
	$(CC) $(CVER) $(CFLAGS) -c $(TEST_TARGET).cu


clean:
	rm $(objects) *.out