//////////////////////////////////////////////////////////////////////////////
// Example illustrating the use of GCoptimization.cpp
//
/////////////////////////////////////////////////////////////////////////////

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <vector>
#include "GCoptimization.h"


GCoptimization::EnergyType smoothFn(int p1, int p2, int l1, int l2, void *data)
{
	float **pred = reinterpret_cast<float**>(data);
	float avg_pred = 0.5 * (pred[p1][p2] + pred[p2][p1]);
	float pred_distance = (l1 == l2) ? (1 - avg_pred) : avg_pred;
	return GCoptimization::EnergyType(pred_distance);
}

void test() {
	// pixels
	// 0 | 1
	// -----
	// 2 | 3
	// -----
	// 4 | 5
	// -----
	// 6 | 7
	// -----
	// 8 | 9
	// 0, 1 are in the same line, and 2, 3 are in the same line, and so on.
	// pred[0][1] = pred[1][0] = 1
	// pred[2][3] = pred[3][2] = 1 ..
	// otherwise, pred = 0
	
	int n_labels = 100;
	int n_sites = 20;
	int label_cost = 0;
	int n_iters = -1;

	GCoptimization::EnergyType *data = new GCoptimization::EnergyType[n_sites*n_labels];
	for (int i = 0; i < n_sites*n_labels; ++i) {
		data[i] = 0;
	}

	GCoptimization::EnergyType **pred = new GCoptimization::EnergyType*[n_sites];
	for (int i = 0; i < n_sites; ++i) {
		pred[i] = new float[n_sites];
		for (int j = 0; j < n_sites; ++j) {
			for (int l = 0; l < n_sites*0.5; ++l) {
				bool find = false;
				if (i >= l * 2 && i <= l * 2 + 1) {
					pred[i][j] = (j >= l * 2 && j <= l * 2 + 1) ? 1 : 0;
					find = true;
				}
				if (find) break;
			}
		}
	}
	
	int *labels = new int[n_sites];

	try {
		GCoptimizationGeneralGraph *gc = new GCoptimizationGeneralGraph(n_sites, n_labels);
		gc->setDataCost(data);
		gc->setSmoothCost(smoothFn, (void*)pred);
		for (int i = 0; i < n_sites - 1; ++i) {
			for (int j = i + 1; j < n_sites; ++j) {
				gc->setNeighbors(i, j, 1); // TODO: use spatial weight
			}
		}
		gc->setLabelCost(label_cost);

		//for (int i = 0; i < n_sites*0.5; ++i) {
		//	gc->setLabel(i * 2, i);
		//	gc->setLabel(i * 2 + 1, i);
		//}

		//for (int i = 0; i < n_sites*0.5; ++i) {
		//	gc->setLabel(i * 2, i);
		//	gc->setLabel(i * 2 + 1, (i+1) % n_sites);
		//}

		gc->setLabelOrder(true);
			
		printf("\nBefore optimization energy is %f", gc->compute_energy());
		gc->expansion(n_iters);
		//gc->swap(n_iters);
		printf("\nAfter optimization energy is %f", gc->compute_energy());

		for (int i = 0; i < n_sites; i++) {
			labels[i] = gc->whatLabel(i);
			printf("\nLabel %d: %d", i, labels[i]);
		}
		printf("\n");

		delete gc;
	}
	catch (GCException e) {
		e.Report();
	}

	delete[] labels;
	for (int i = 0; i < n_sites; ++i)
		delete[] pred[i];
	delete[] pred;
	delete[] data;
}


////////////////////////////////////////////////////////////////////////////////

int main(int argc, char **argv)
{
	test();

	// int width = 10;
	// int height = 5;
	// int num_pixels = width*height;
	// int num_labels = 7;


	// // smoothness and data costs are set up one by one, individually
	// GridGraph_Individually(width,height,num_pixels,num_labels);

	// // smoothness and data costs are set up using arrays
	// GridGraph_DArraySArray(width,height,num_pixels,num_labels);

	// // smoothness and data costs are set up using functions
	// GridGraph_DfnSfn(width,height,num_pixels,num_labels);

	// // smoothness and data costs are set up using arrays. 
	// // spatially varying terms are present
	// GridGraph_DArraySArraySpatVarying(width,height,num_pixels,num_labels);

	// //Will pretend our graph is 
	// //general, and set up a neighborhood system
	// // which actually is a grid
	// GeneralGraph_DArraySArray(width,height,num_pixels,num_labels);

	// //Will pretend our graph is general, and set up a neighborhood system
	// // which actually is a grid. Also uses spatially varying terms
	// GeneralGraph_DArraySArraySpatVarying(width,height,num_pixels,num_labels);

	// printf("\n  Finished %d (%d) clock per sec %d",clock()/CLOCKS_PER_SEC,clock(),CLOCKS_PER_SEC);

	return 0;
}

/////////////////////////////////////////////////////////////////////////////////

