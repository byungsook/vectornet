// Example usage : minimize energy E(x, y) = 2 * x + 3 * (y + 1) + (x + 1)*(y + 2), where x, y \in{ 0, 1 }.

#include <stdio.h>
#include "QPBO.h"

int main()
{
	typedef int REAL;
	QPBO<REAL>* q;

	//q = new QPBO<REAL>(2, 1); // max number of nodes & edges
	//q->AddNode(2); // add two nodes

	//q->AddUnaryTerm(0, 0, 2); // add term 2*x
	//q->AddUnaryTerm(1, 3, 6); // add term 3*(y+1)
	//q->AddPairwiseTerm(0, 1, 2, 3, 4, 6); // add term (x+1)*(y+2)

	//q->Solve();
	//q->ComputeWeakPersistencies();

	//int x = q->GetLabel(0);
	//int y = q->GetLabel(1);
	//printf("Solution: x=%d, y=%d\n", x, y);

	// Minimize the following function of 3 binary variables:
	// E(x, y, z) = x - 2*y + 3*(1-z) - 4*x*y + 5*|y-z|
	// -5, x=y=z=1
	q = new QPBO<REAL>(3, 3); // max number of nodes & edges
	q->AddNode(3); // add two nodes

	q->AddUnaryTerm(0, 0, 1);  // add term x
	q->AddUnaryTerm(1, 0, -2); // add term -2*y
	q->AddUnaryTerm(2, 3, 0);  // add term 3*(1-z)
	
	q->AddPairwiseTerm(0, 1, 0, 0, 0, -4); // add term -4*x*y
	q->AddPairwiseTerm(1, 2, 0, 5, 5, 0);  // add term 5*|y-z|

	printf("Before = %f\n", q->ComputeTwiceEnergy()*0.5);

	q->Solve();
	//q->ComputeWeakPersistencies();

	printf("Minimum = %f\n", q->ComputeTwiceEnergy()*0.5);

	int x = q->GetLabel(0);
	int y = q->GetLabel(1);
	int z = q->GetLabel(2);
	printf("Solution: x=%d, y=%d, z=%d\n", x, y, z);


	return 0;
}
