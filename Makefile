default: task3

task1: task1.cpp
	g++ task1.cpp `pkg-config --libs opencv` `pkg-config --cflags opencv`

task2: task2.cpp
	g++ face.cpp `pkg-config --libs opencv` `pkg-config --cflags opencv`

task3: task3.cpp
	g++ task3.cpp `pkg-config --libs opencv` `pkg-config --cflags opencv`

task4: task4.cpp
	g++ task4.cpp `pkg-config --libs opencv` `pkg-config --cflags opencv`

clean:
	rm -f a.out
	rm -f darts
	rm -f detected.jpg
