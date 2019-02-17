#include <iostream>
#include <fstream>
#include <random>
#include <vector>
#include <string>
#include "boost/multi_array.hpp"
#include "boost/tuple/tuple.hpp"
#include "boost/tuple/tuple_comparison.hpp"
#include <cassert>
#include <cmath>
#include <map>
#include <queue>
#include <ctime>
#include <set>

using namespace std;
ofstream ofile;
typedef boost::multi_array<int, 3> Lattice;
typedef Lattice::index coord;
typedef boost::tuple<int, int, int > site;

int seed = time(0);
default_random_engine e(seed);

inline int periodic(int i, const int limit, int add) {
    // This implementation of periodic boundary condition is from 
    // the 2D ising model example by Hjorth Jensen.
    return (i + limit + add) % (limit);
}

void read_input(unsigned int&, unsigned int&, unsigned int&, double&, double&, string&, vector<double>&);
void initialize(Lattice&, map<int, double>&, double&, double&, double&);
site choose_random_site(const unsigned int lsize);
void metropolis();
void wolff(unsigned int mcs, Lattice &lattice, const unsigned int lsize, const double K,
double& M, double& E);
vector<site> find_cluster(Lattice &lattice, const unsigned int lsize, const double K);
vector<site> find_neighbors(const site, const unsigned int);

void read_input(unsigned int &lsize, unsigned int &num_steps, 
unsigned int &start, double &init_K, double &final_K, string &filename, vector<double> &sweep) {
    // Function to read configurations for Monte Carlo pass.
    cout << "Enter lattice size: " << endl;
    cin >> lsize;
    cout << "Enter initial K: " << endl;
    cin >> init_K;
    cout << "Enter Final K: " << endl;
    cin >> final_K;
    cout << "Enter number of steps: " << endl;
    cin >> num_steps;
    cout << "Enter output filename: " << endl;
    cin >> filename;
    cout << "Enter initial_condition (0 for coldstart, 1 for random start)" << endl;
    cin >> start;

    double K_step = (final_K - init_K) / ((double) num_steps);
    double K = {0.0};
    for (auto i = 0; i != num_steps; ++i) {
        K = init_K + K_step * (double) i;
        sweep.push_back(K);
    }
    for (auto k : sweep) {
        cout << k << endl;
    }
}

void initialize( Lattice &lattice, map<int, double> &weights, double &K, 
double &M, double &E) {
    static uniform_int_distribution<int> u(0,1);
    unsigned int lsize = 0;
    lsize = lattice.size();
    for (auto i = 0; i != lsize; ++i) {
        for (auto j = 0; j != lsize; ++j) {
            for (auto k = 0; k != lsize; ++k) {
                // Call random number generator and initialize lattice
                if (u(e) == 1) {
                    lattice[i][j][k] = 1;
                } else {
                    lattice[i][j][k] = -1;
                }
                // Adjust magnetization accordingly
                M += (double) lattice[i][j][k];
            }
        }
    }
    for (auto i = 0; i != lsize; ++i) {
        for (auto j = 0; j != lsize; ++j) {
            for (auto k = 0; k != lsize; ++k) {
                E -= (double) lattice[i][j][k] * (
                    lattice[periodic(i, lsize, 1)][j][k] + 
                    lattice[i][periodic(j, lsize, 1)][k] +
                    lattice[i][j][periodic(k, lsize, 1)]
                );
            }
        }

    }
    // Initializing weights for metropolis
    weights.insert(pair<int, double>(-12, exp(-12 * K)));
    weights.insert(pair<int, double>(-8, exp(-8 * K)));
    weights.insert(pair<int, double>(-4, exp(-4 * K)));
    
    cout << "Initial M = " << M << endl;
    cout << "Initial E = " << E << endl;
}

site choose_random_site(const unsigned int lsize) {
    // Helper function for choosing random site.
    static uniform_int_distribution<int> u(0, lsize-1);
    int idx, idy, idz = {0};
    idx = u(e);
    idy = u(e);
    idz = u(e);
    site s(idx, idy, idz);
    return s;
}

vector<site> find_neighbors(const site s, const unsigned int lsize) {
    vector<site> neighbors;
    int idx, idy, idz = {0};
    idx = s.get<0>();
    idy = s.get<1>();
    idz = s.get<2>();
    for (auto i = 0; i != 3; ++i) {
        if (i == 0) {
            site n1(periodic(idx, lsize, 1), idy, idz);
            site n2(periodic(idx, lsize, -1), idy, idz);
            neighbors.push_back(n1);
            neighbors.push_back(n2);
        }
        if (i == 1) {
            site n1(idx, periodic(idy, lsize, 1), idz);
            site n2(idx, periodic(idy, lsize, -1), idz);
            neighbors.push_back(n1);
            neighbors.push_back(n2);
        }
        if (i == 2) {
            site n1(idx, idy, periodic(idz, lsize, 1));
            site n2(idx, idy, periodic(idz, lsize, -1));
            neighbors.push_back(n1);
            neighbors.push_back(n2);
        }
    }
    return neighbors;
}

double deltaE(site s, Lattice& lattice, const unsigned int lsize, const double K) {
    double dE = 0.0;
    vector<site> neighbors = find_neighbors(s, lsize);
    for (auto n : neighbors) {
        dE += lattice[s.get<0>()][s.get<1>()][s.get<2>()] *
              lattice[n.get<0>()][n.get<1>()][n.get<2>()];
    }
    dE = -dE * K * 2;
    return dE;
}

void metropolis();

vector<site> find_cluster(Lattice &lattice, const unsigned int lsize, const double K) {
    static uniform_real_distribution<double> u(0,1);
    double r = {0.0};
    site s = choose_random_site(lsize);
    queue<site> q;
    vector<site> added;
    q.push(s);
    added.push_back(s);
    while (!q.empty()) {
        s = q.front();
        q.pop();
        vector<site> neighbors = find_neighbors(s, lsize);
        for (auto n : neighbors) {
            double spin_n = (double) lattice[n.get<0>()][n.get<1>()][n.get<2>()];
            double spin_s = (double) lattice[s.get<0>()][s.get<1>()][s.get<2>()];
            double prob = 1 - exp( -2 * K * spin_n * spin_s);
            r = u(e);
            if (find(added.begin(), added.end(), n) == added.end() && r < prob) {
                q.push(n);
                added.push_back(n);
            }
        }
    }
    return added;
}

void wolff(unsigned int mcs, Lattice &lattice, const unsigned int lsize, const double K,
double& M, double& E) {
    double dE = 0.0;
    for (auto i = 0; i != mcs; ++i) {
        vector<site> clusters = find_cluster(lattice, lsize, K);
        cout << "Cluster size: " << clusters.size() << endl;
        for (auto s : clusters) {
            dE = deltaE(s, lattice, lsize, K);
            lattice[s.get<0>()][s.get<1>()][s.get<2>()] *= -1;
            M += 2 * lattice[s.get<0>()][s.get<1>()][s.get<2>()];
            // E += dE;
        }
    };
    cout << "M = " << M << endl;
}

void write_data(const double& M, const double& E, 
const string& output_filename) {
    ofstream result;
    result.open(output_filename);
    result << "M, M2, E, E2, abs(M)\n";
    
}

int main() {

    unsigned int lsize,  num_steps, start = {0};
    double init_K, final_K, M, E = {0.0};
    double K = {0.22};
    lsize = 100;
    unsigned int N = lsize * lsize * lsize;
    vector<double> sweep;
    map<int, double> weights;
    string output_filename;
    //read_input(lsize, num_steps, start, init_K, final_K, output_filename, sweep);
    Lattice lattice(boost::extents[lsize][lsize][lsize]);
    default_random_engine e;
    initialize(lattice, weights, K, M, E);
    cout << "Done Initializing" << endl;
    wolff(100000, lattice, lsize, K, M, E);
    double M_test, E_test = {0.0};
    for (auto i = 0; i != lsize; ++i) {
        for (auto j = 0; j != lsize; ++j) {
            for (auto k = 0; k != lsize; ++k) {
                M_test += (double) lattice[i][j][k];
            }
        }
    }
    for (auto i = 0; i != lsize; ++i) {
        for (auto j = 0; j != lsize; ++j) {
            for (auto k = 0; k != lsize; ++k) {
                E_test -= (double) lattice[i][j][k] * (
                    lattice[periodic(i, lsize, 1)][j][k] + 
                    lattice[i][periodic(j, lsize, 1)][k] +
                    lattice[i][j][periodic(k, lsize, 1)]
                );
            }
        }
    }
    cout << "M_test = " << M_test / (double) N << endl;
    cout << "E_test = " << E_test / (double) N << endl;

    return 0;
}