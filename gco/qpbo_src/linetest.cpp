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
	//if (p1 > p2) {
	//	int tmp = p2;
	//	p2 = p1;
	//	p1 = tmp;
	//}
	//float avg_pred = pred[p1][p2];
	float pred_distance = (l1 == l2) ? (1 - avg_pred) : avg_pred;
	return GCoptimization::EnergyType(pred_distance);
}

void test2() {
	// pixels
	//      0
	// ------------
	//      1
	// ------------
	// 5|6|7/2|8|9
	// ------------
	//      3
	// ------------
	//      4
	// 0-4 / 5-9 in the same line
	// 2 and 7 are duplicated intersection point

	int n_labels = 100;
	int n_sites = 10;
	int label_cost = 0;
	int n_iters = -1;

	GCoptimization::EnergyType *data = new GCoptimization::EnergyType[n_sites*n_labels];
	for (int i = 0; i < n_sites*n_labels; ++i) {
		data[i] = 0;
	}

	GCoptimization::EnergyType **pred = new GCoptimization::EnergyType*[n_sites];
	for (int i = 0; i < n_sites; ++i) {
		pred[i] = new float[n_sites];
		for (int j = 0; j < n_sites; ++j)
			pred[i][j] = 0;
	}

	pred[0][1] = pred[0][1] = pred[0][2] = pred[0][3] = pred[0][4] = pred[0][7] = 1;
	pred[1][0] = pred[0][1] = pred[1][2] = pred[1][3] = pred[1][4] = pred[1][7] = 1;
	pred[2][0] = pred[2][1] = pred[2][2] = pred[2][3] = pred[2][4] = 
		pred[2][5] = pred[2][6] = pred[2][8] = pred[2][9] = 1;
	pred[3][0] = pred[3][1] = pred[3][2] = pred[3][3] = pred[3][4] = pred[3][7] = 1;
	pred[4][0] = pred[4][1] = pred[4][2] = pred[4][3] = pred[4][4] = pred[4][7] = 1;

	pred[5][5] = pred[5][6] = pred[5][7] = pred[5][8] = pred[5][9] = pred[5][2] = 1;
	pred[6][5] = pred[6][6] = pred[6][7] = pred[6][8] = pred[6][9] = pred[6][2] = 1;
	pred[7][0] = pred[7][1] = pred[7][3] = pred[7][4] =
		pred[7][5] = pred[7][6] = pred[7][7] = pred[7][8] = pred[7][9] = 1;
	pred[8][5] = pred[8][6] = pred[8][7] = pred[8][8] = pred[8][9] = pred[8][2] = 1;
	pred[9][5] = pred[9][6] = pred[9][7] = pred[9][8] = pred[9][9] = pred[9][2] = 1;


	int *labels = new int[n_sites];

	try {
		GCoptimizationGeneralGraph *gc = new GCoptimizationGeneralGraph(n_sites, n_labels);
		gc->setDataCost(data);
		gc->setSmoothCost(smoothFn, (void*)pred);
		for (int i = 0; i < n_sites - 1; ++i) {
			for (int j = i + 1; j < n_sites; ++j) {
				gc->setNeighbors(i, j, 1); // just assume every pixel is connected to each other
			}
		}
		gc->setNeighbors(2, 7, 0);
		gc->setLabelCost(label_cost);
		gc->setLabelOrder(true); // random order

		printf("\nBefore optimization energy is %f", gc->compute_energy());
		//gc->expansion(n_iters);
		gc->swap(n_iters);
		//gc->fusion(n_iters);
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
					printf("%d,%d,%f\n", i, j, pred[i][j]);
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

		gc->setLabelOrder(true); // random order
			
		printf("\nBefore optimization energy is %f", gc->compute_energy());
		//gc->expansion(n_iters);
		gc->swap(n_iters);
		//gc->fusion(n_iters);
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
	//test();
	test2();

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

