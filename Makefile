default: darts

darts: darts.cpp
	g++ darts.cpp `pkg-config --libs opencv` `pkg-config --cflags opencv`

face: face.cpp
	g++ face.cpp `pkg-config --libs opencv` `pkg-config --cflags opencv`

sobel: sobel.cpp
	g++ sobel.cpp `pkg-config --libs opencv` `pkg-config --cflags opencv`

clean:
	rm -f a.out
	rm -f darts
	rm -f detected.jpg
