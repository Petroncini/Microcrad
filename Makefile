all: value.o main.o
	gcc value.o main.o -std=c99 -o main -lm 

value.o:
	gcc -std=c99 -c value.c

main.o:
	gcc -std=c99 -c main.c

run: all
	./main

clean:
	rm *.o
