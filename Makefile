all:  value.o neuron.o layer.o NN.o main.o
	gcc value.o neuron.o layer.o NN.o main.o -std=c99 -o main -lm 

value.o:
	gcc -std=c99 -c value.c

neuron.o:
	gcc -std=c99 -c neuron.c

layer.o:
	gcc -std=c99 -c layer.c

NN.o:
	gcc -std=c99 -c NN.c

main.o:
	gcc -std=c99 -c main.c

clean:
	rm *.o

run: clean all
	./main

