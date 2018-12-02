default: darts

darts: darts.cpp
	g++ darts.cpp `pkg-config --libs opencv` `pkg-config --cflags opencv`

face: face.cpp
	g++ face.cpp `pkg-config --libs opencv` `pkg-config --cflags opencv`

hough: try_for_hough.cpp
	g++ try_for_hough.cpp `pkg-config --libs opencv` `pkg-config --cflags opencv`


clean:
	rm -f a.out
	rm -f darts
	rm -f detected.jpg
