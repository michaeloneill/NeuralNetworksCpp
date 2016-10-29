CXX=g++
CXXFLAGS=-Wall -Wextra -g -I/lsc/opt/armadillo-6.100/include


NNet.o: NNet.C NNet.H
	${CXX} -c $< -o $@ ${CXXFLAGS}
loadMNIST.o: loadMNIST.C loadMNIST.H
	${CXX} -c $< -o $@ ${CXXFLAGS}
matrixToFile.o: matrixToFile.C matrixToFile.H
	${CXX} -c $< -o $@ ${CXXFLAGS}
Tuning.o: Tuning.C Tuning.H
	${CXX} -c $< -o $@ ${CXXFLAGS}
Layers.o: Layers.C Layers.H
	${CXX} -c $< -o $@ ${CXXFLAGS}
ActivationFns.o: ActivationFns.C ActivationFns.H
	${CXX} -c $< -o $@ ${CXXFLAGS}
main.o: main.C
	${CXX} -c $< -o $@ ${CXXFLAGS}

CXXLIBS=-L/lsc/opt/armadillo-6.100/lib -larmadillo 

NNet: NNet.o loadMNIST.o matrixToFile.o Tuning.o Layers.o ActivationFns.o main.o
	${CXX} $^ ${CXXLIBS} -o $@ ${CXXFLAGS}

clean:
	rm NNet.o loadMNIST.o matrixToFile.o Tuning.o Layers.o ActivationFns.o main.o



