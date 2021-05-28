// image-reconstruction.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <lbfgs.h>

#include <fftw3.h>

#include <algorithm>
#include <iostream>
#include <random>
#include <map>
#include <vector>
#include <exception>


static auto GetRandomInts(int numGoodPixels, int numImgPixels)
{
    std::vector<int> ri;

    std::default_random_engine dre;

    ri.resize(numGoodPixels);
    for (int j = 0; j < numGoodPixels; ++j) {
        ri[j] = j;
    }

    std::map<int, int> displaced;

    // Fisher-Yates shuffle Algorithm
    for (int j = 0; j < numGoodPixels; ++j)
    {
        std::uniform_int_distribution<int> di(j, numImgPixels - 1);
        int idx = di(dre);

        if (idx != j)
        {
            int& to_exchange = (idx < numGoodPixels)
                ? ri[idx]
                : displaced.try_emplace(idx, idx).first->second;
            std::swap(ri[j], to_exchange);
        }
    }

    std::sort(ri.begin(), ri.end());

    return ri;
}


struct LbfgsContext {
    cv::Size imageSize;
    std::vector <bool> ri;
    std::vector<uint8_t> b;
};

static int progress(
    void *instance,
    const lbfgsfloatval_t *x,
    const lbfgsfloatval_t *g,
    const lbfgsfloatval_t fx,
    const lbfgsfloatval_t xnorm,
    const lbfgsfloatval_t gnorm,
    const lbfgsfloatval_t step,
    int n,
    int k,
    int ls
)
{
    return 0;
}

static lbfgsfloatval_t evaluate(
    void *instance,
    const lbfgsfloatval_t *x,
    lbfgsfloatval_t *g,
    const int n,
    const lbfgsfloatval_t step
)
{
    auto context = static_cast<LbfgsContext*>(instance);

    //const cv::Mat x2(context->imageSize.height, 
    //    context->imageSize.width, 
    //    CV_64FC1, 
    //    const_cast<void*>(static_cast<const void*>(x)));

    //cv::Mat Ax2;
    //cv::idct(x2, Ax2);

    cv::Mat Ax2(context->imageSize.height, context->imageSize.width, CV_64FC1);
    fftw_plan idct_plan = fftw_plan_r2r_2d(
        context->imageSize.height, context->imageSize.width, const_cast<double*>(x),
        static_cast<double*>(static_cast<void*>(Ax2.data)),
        FFTW_REDFT01, FFTW_REDFT01, FFTW_MEASURE);
    fftw_execute(idct_plan);
    fftw_destroy_plan(idct_plan);
    //Ax2 /= sqrt(2.0*context->imageSize.height * context->imageSize.width);

    //cv::Mat Axb2 = cv::Mat::zeros(context->imageSize.height, context->imageSize.width, CV_64FC1);

    const auto coeff = 1. / sqrt(2.0*context->imageSize.height * context->imageSize.width);

    double fx = 0;
    auto it = context->b.begin();
    for (int i = 0; i < context->ri.size(); ++i)
    {
        //const int idx = context->ri[i];
        auto &v = static_cast<double*>(static_cast<void*>(Ax2.data))[i];
        if (context->ri[i])
        {
            const auto Ax = v * coeff - *(it++);
            v = Ax;
            fx += Ax * Ax;
        }
        else
        {
            v = 0;
        }
    }

    //cv::Mat AtAxb2(context->imageSize.height,
    //    context->imageSize.width,
    //    CV_64FC1,
    //    g);
    //cv::dct(Axb2, AtAxb2);
    //AtAxb2 *= 2;

    fftw_plan dct_plan = fftw_plan_r2r_2d(context->imageSize.height, context->imageSize.width, 
        static_cast<double*>(static_cast<void*>(Ax2.data)),
        g,
        FFTW_REDFT10, FFTW_REDFT10, FFTW_MEASURE);
    fftw_execute(dct_plan);
    fftw_destroy_plan(dct_plan);
    for (int i = 0; i < context->imageSize.height; i++) {
        for (int j = 0; j < context->imageSize.width; j++) {
            g[i*context->imageSize.width + j] *= coeff;  //  /= /*2.0 **/ sqrt(2.0*context->imageSize.width*context->imageSize.height);
        }
    }
    return fx;
};

int main(int argc, char** argv)
{
    try {
        cv::String filename;
        if (argc >= 2)
            filename = argv[1];
        else {
            try {
                filename = cv::samples::findFile("lena.jpg");
            }
            catch (const std::exception& ex) {
                std::string s(ex.what());
                s = s.substr(s.find(") ") + 2);
                s = s.substr(0, s.find("modules"));
                filename = s + "samples/data/lena.jpg";
            }
        }
        cv::Mat src = cv::imread(filename, cv::IMREAD_GRAYSCALE);

        imshow("Original", src);

        const int numImgPixels = src.rows * src.cols;

        const int numGoodPixels = numImgPixels / 10;

        LbfgsContext context;

        context.imageSize.width = src.cols;
        context.imageSize.height = src.rows;

        auto ri = GetRandomInts(numGoodPixels, numImgPixels);

        context.ri.resize(numImgPixels);
        context.b.reserve(numGoodPixels);
        for (auto& v : ri)
        {
            context.ri[v] = true;
            context.b.push_back(src.data[v]);
        }


        cv::Mat squeezed = cv::Mat::zeros(src.rows, src.cols, CV_8UC1);
        for (int i = 0; i < ri.size(); ++i)
        {
            const int idx = ri[i];
            squeezed.data[idx] = context.b[i];
        }

        imshow("Squeezed", squeezed);

        //////////////////////////////////////////////////////////////////////////

        const double param_c = 5;

        // Initialize solution vector
        lbfgsfloatval_t fx;
        lbfgsfloatval_t *x = lbfgs_malloc(numImgPixels);
        if (x == nullptr) {
            //
        }
        for (int i = 0; i < numImgPixels; i++) {
            x[i] = 1;
        }

        // Initialize the parameters for the optimization.
        lbfgs_parameter_t param;
        lbfgs_parameter_init(&param);
        param.orthantwise_c = param_c; // this tells lbfgs to do OWL-QN
        param.linesearch = LBFGS_LINESEARCH_BACKTRACKING;
        int lbfgs_ret = lbfgs(numImgPixels, x, &fx, evaluate, progress, &context, &param);

        //cv::Mat Xat2(context.imageSize.height, context.imageSize.width, CV_64FC1, x);
        //cv::Mat Xa;
        //idct(Xat2, Xa);

        cv::Mat Xa(context.imageSize.height, context.imageSize.width, CV_64FC1);
        fftw_plan idct_plan = fftw_plan_r2r_2d(
            context.imageSize.height, context.imageSize.width, const_cast<double*>(x),
            static_cast<double*>(static_cast<void*>(Xa.data)),
            FFTW_REDFT01, FFTW_REDFT01, FFTW_MEASURE);
        fftw_execute(idct_plan);
        fftw_destroy_plan(idct_plan);
        Xa /= sqrt(2.0*context.imageSize.height * context.imageSize.width);

        lbfgs_free(x);

        cv::Mat dst;
        Xa.convertTo(dst, CV_8U);

        imshow("Restored", dst);

        cv::waitKey();

    }
    catch (const std::exception& ex) {
        std::cerr << typeid(ex).name() << ": " << ex.what() << '\n';
    }
}
