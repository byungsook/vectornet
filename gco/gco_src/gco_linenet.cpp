//////////////////////////////////////////////////////////////////////////////
// Example illustrating the use of GCoptimization.cpp
//
/////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <string>
#include "GCoptimization.h"

float smoothFn(int p1, int p2, int l1, int l2, void *data)
{
	float **pred = reinterpret_cast<float**>(data);
	//float avg_pred = 0.5 * (pred[p1][p2] + pred[p2][p1]);
	if (p1 > p2) {
		int tmp = p2;
		p2 = p1;
		p1 = tmp;
	}
	float avg_pred = pred[p1][p2];
	float pred_distance = (l1 == l2) ? (1 - avg_pred) : avg_pred;
	//return int(pred_distance * 1000);
	return pred_distance;
}

int main(int argc, char **argv)
{
	//std::cout << argv[1] << std::endl;
	std::ifstream is(argv[1]);

	if (!is.is_open()) {
		std::cout << "Unable to open pred file" << std::endl;
		return -1;
	}
	
	std::string pred_file_path, data_dir;
	int n_labels, n_sites, label_cost;
	float neighbor_sigma, prediction_sigma;
	is >> pred_file_path;
	is >> data_dir;
	is >> n_labels;
	is >> label_cost;
	is >> neighbor_sigma;
	is >> prediction_sigma;
	is >> n_sites;

	//std::cout << "pred_file_path:" << pred_file_path << std::endl;
	//std::cout << "data_dir:" << data_dir << std::endl;
	//std::cout << "n_labels:" << n_labels << std::endl;
	//std::cout << "label_cost:" << label_cost << std::endl;
	//std::cout << "neighbor_sigma:" << neighbor_sigma << std::endl;
	//std::cout << "pred_sigma:" << pred_sigma << std::endl;
	//std::cout << "n_sites:" << n_sites << std::endl;
	
	float **pred = new float*[n_sites];
	for (int i = 0; i < n_sites; ++i) {
		pred[i] = new float[n_sites]();
	}
	float **w = new float*[n_sites];
	for (int i = 0; i < n_sites; ++i) {
		w[i] = new float[n_sites]();
	}

	while (is.good()) {
		int i, j;
		float p, spatial;		
		is >> i >> j >> p >> spatial;
		//std::cout << i << " " << j << " " << p << " " << spatial << std::endl;
		pred[i][j] = p;
		w[i][j] = spatial;
	}

	// std::cout << "0 1 " << pred[0][1] << " " << w[0][1] << std::endl;
	// std::cout << "0 2 " << pred[0][2] << " " << w[0][2] << std::endl;

	// is.close();
	// return 0;


	int n_iters = 3;
	float *data = new float[n_sites*n_labels]();
	int *labels = new int[n_sites]();

	try {
		GCoptimizationGeneralGraph *gc = new GCoptimizationGeneralGraph(n_sites, n_labels);
		gc->setDataCost(data);
		gc->setSmoothCost(smoothFn, (void*)pred);
		for (int i = 0; i < n_sites - 1; ++i) {
			for (int j = i + 1; j < n_sites; ++j) {
				gc->setNeighbors(i, j, w[i][j]);
			}
		}
		gc->setLabelCost(label_cost);
		gc->setLabelOrder(true);

		std::string label_file_path = argv[1];
		label_file_path.replace(label_file_path.end() - 5, label_file_path.end(), ".label");
		std::ofstream os(label_file_path.c_str());
		if (!os.is_open()) {
			std::cout << "Unable to open label file" << std::endl;
			return -1;
		}

		// printf("\nBefore optimization energy is %f", gc->compute_energy());
		os << gc->compute_energy() << std::endl;
		// gc->expansion(n_iters);
		gc->swap(n_iters);
		// gc->fusion(n_iters);
		os << gc->compute_energy() << std::endl;
		// printf("\nAfter optimization energy is %f", gc->compute_energy());

		for (int i = 0; i < n_sites; i++) {
			labels[i] = gc->whatLabel(i);
			//printf("\nLabel %d: %d", i, labels[i]);
			os << labels[i] << " ";
		}
		//printf("\n");
		
		os.close();
		delete gc;
	}
	catch (GCException e) {
		e.Report();
	}

	delete[] labels;
	for (int i = 0; i < n_sites; ++i) {
		delete[] pred[i];
		delete[] w[i];
	}
	delete[] pred;
	delete[] w;
	delete[] data;

	return 0;
}