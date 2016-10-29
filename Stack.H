#ifndef STACK_H
#define STACK_H

template <class T1, class T2> struct Stack {

    Stack(){}
    Stack(T1 w, T2 b): w(w), b(b){}
    T1 w;
    T2 b;
    Stack& operator+=(const Stack& s){
	w += s.w;
	b += s.b;

	return *this;
    }
    //just for testing
    Stack(const Stack& s){
	w = s.w;
	b = s.b;
    }
};



#endif
