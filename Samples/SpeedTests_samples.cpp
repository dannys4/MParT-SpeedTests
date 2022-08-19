/**
TODO: ADD DESCRIPTION
 */


#include <random>
#include <fstream>
#include <stdio.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include <MParT/ConditionalMapBase.h>
#include <MParT/MapFactory.h>
#include <MParT/MultiIndices/MultiIndexSet.h>
#include <MParT/Utilities/ArrayConversions.h>

#include <random>
#include <chrono>

using namespace std::chrono;

using namespace mpart; 

template<typename MemorySpace>
struct Generator{

    const unsigned int seed = 2012;

    std::mt19937 mt;
    std::uniform_real_distribution<double> dist;

    Generator() : mt(2012), dist(-1,1){};

    Kokkos::View<double*,MemorySpace> Coeffs(unsigned int numCoeffs);
    Kokkos::View<double**, MemorySpace> Points(unsigned int dim, unsigned int numPts);
};

template<>
Kokkos::View<double*,Kokkos::HostSpace> Generator<Kokkos::HostSpace>::Coeffs(unsigned int numCoeffs)
{
    Kokkos::View<double*,Kokkos::HostSpace> output("Coefficients", numCoeffs);
    for(unsigned int i=0; i<numCoeffs; ++i)
        output(i) = dist(mt);

    return output;
}

template<>
Kokkos::View<double**,Kokkos::HostSpace> Generator<Kokkos::HostSpace>::Points(unsigned int dim, unsigned int numPts)
{
    Kokkos::View<double**,Kokkos::HostSpace> output("Points", dim, numPts);
    for(unsigned int i=0; i<dim; ++i){
        for(unsigned int j=0; j<numPts; ++j)
            output(i,j) = dist(mt);
    }

    return output;
}

#if defined(KOKKOS_ENABLE_CUDA )
template<typename MemorySpace>
Kokkos::View<double*,MemorySpace> Generator<MemorySpace>::Coeffs(unsigned int numCoeffs)
{   
    Kokkos::Random_XorShift64_Pool<> rand_pool(5374857);

    Kokkos::View<double*, MemorySpace> coeffs("Coefficients", numCoeffs);
    Kokkos::parallel_for(1, KOKKOS_LAMBDA(int const& i){
        
        auto rand_gen = rand_pool.get_state();
        for (int k = 0; k < numCoeffs; k++) 
            coeffs(k) = rand_gen.frand(-1,1);
        rand_pool.free_state(rand_gen);
    });

    Kokkos::fence();

    return coeffs;
}

template<typename MemorySpace>
Kokkos::View<double**,MemorySpace> Generator<MemorySpace>::Points(unsigned int dim, unsigned int numPts)
{   
    Kokkos::Random_XorShift64_Pool<> rand_pool(5374857);

    Kokkos::View<double**, MemorySpace> pts("Points", dim, numPts);
    Kokkos::parallel_for(dim, KOKKOS_LAMBDA(int const& i){
        auto rand_gen = rand_pool.get_state();
        for (int k = 0; k < numPts; k++) 
            pts(i,k) = rand_gen.frand(-1,1);
        rand_pool.free_state(rand_gen);
    });

    Kokkos::fence();

    return pts;
}

#endif 

int main(int argc, char* argv[]){

    assert(argc>=2);
    std::string backend = argv[1];
    
    Kokkos::InitArguments args;
    if(argc==3)
        args.num_threads = std::stoi(argv[2]);
    else
        args.num_threads = 1;

    Kokkos::initialize(args);
    {
    MapOptions opts;

    unsigned int nn = 7;

    Generator<Kokkos::DefaultExecutionSpace::memory_space> gen;

    auto tEvalMat_m = Eigen::VectorXd(nn);
    auto tLogDetMat_m = Eigen::VectorXd(nn);
    auto tEvalMat_s = Eigen::VectorXd(nn);
    auto tLogDetMat_s = Eigen::VectorXd(nn);

    unsigned int nk = 50;

    auto tEval = Eigen::MatrixXd(nn,nk);
    auto tLogDet = Eigen::MatrixXd(nn,nk);


    std::cout << "\nRunning Backend " << backend << std::endl;

    for (unsigned int n=0; n<nn;++n){
        unsigned int dim = 5;
        unsigned int order = 2;
        
        auto map = MapFactory::CreateTriangular<Kokkos::DefaultExecutionSpace::memory_space>(dim,dim,order,opts);
        
        auto numCoeffs = map->numCoeffs;
        unsigned int numPts = pow(10,(6-n));

        // auto tEval = Eigen::VectorXd(nk);
        // auto tLogDet = Eigen::VectorXd(nk);

        std::cout << "    NPts = " << numPts << ",  Trial: " << 0 << "/" << nk-1 << std::flush;

        for(unsigned int k=0; k<nk;++k){

            // Print the current trial.  use \b to overwrite the previous number
            for(int iii=0; iii< 3 + std::floor(log10(nk))+ std::floor(std::log10(std::max<int>(k,1))); ++iii)
                std::cout << "\b";
            std::cout << k << "/" << nk-1 << std::flush;

            Kokkos::View<double*> coeffs = gen.Coeffs(numCoeffs);            
            map->SetCoeffs(coeffs); 

            Kokkos::View<double**> pts = gen.Points(dim,numPts);
          
            auto start1 = high_resolution_clock::now();
            auto evals = map->Evaluate(pts);
            auto stop1 = high_resolution_clock::now();
            auto duration1 = duration_cast<microseconds>(stop1 - start1);
            tEval(n,k)=duration1.count();

            auto start2 = high_resolution_clock::now();
            auto logDet = map->LogDeterminant(pts);
            auto stop2 = high_resolution_clock::now();
            auto duration2 = duration_cast<microseconds>(stop2 - start2);
            tLogDet(n,k)=duration2.count();
        }
        // tEvalMat_m(n)=tEval.mean();
        // tLogDetMat_m(n)=tLogDet.mean();
        // tEvalMat_s(n)=std::sqrt((tEval - tEval.mean()).square().sum()/(tEval.size()-1));
        // tLogDetMat_s(n)=std::sqrt((tLogDet - tLogDet.mean()).square().sum()/(tLogDet.size()-1));
    }
    
    {
        std::stringstream filename;
        filename << "ST_CPP_eval_d5_to2_nt" << args.num_threads << "_" << backend << ".txt";

        std::ofstream file1(filename.str());  
        if(file1.is_open())  // si l'ouverture a réussi
        {   
        file1 << tEval << "\n";
        }
    }

    {
        std::stringstream filename;
        filename << "ST_CPP_logdet_d5_to2_nt" << args.num_threads << "_" << backend << ".txt";

        std::ofstream file2(filename.str());  
        if(file2.is_open())  // si l'ouverture a réussi
        {   
        file2 << tLogDet << "\n";
        }
    }
    

    }

    Kokkos::finalize();
	
    return 0;
}