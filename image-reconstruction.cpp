// image-reconstruction.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <lbfgs.h>

#include <iostream>
#include <random>
#include <map>
#include <vector>
#include <exception>

struct LbfgsContext {
    cv::Size imageSize;
    std::vector<int> ri;
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

    cv::Mat x2 = cv::Mat(context->imageSize.width, context->imageSize.height, CV_64FC1, const_cast<void*>(static_cast<const void*>(x))).t();

    cv::Mat Ax2;
    cv::idct(x2, Ax2);

    cv::Mat Axb2 = cv::Mat::zeros(x2.rows, x2.cols, CV_64FC1);

    double fx = 0;
    for (int i = 0; i < context->ri.size(); ++i)
    {
        const int idx = context->ri[i];
        const auto Ax = Ax2.reshape(1, 1).at<double>(idx) - context->b[i];
        fx += Ax * Ax;
        Axb2.reshape(1, 1).at<double>(idx) = Ax;
    }

    cv::Mat AtAxb2;
    cv::dct(Axb2, AtAxb2);
    AtAxb2 *= 2;
    cv::transpose(AtAxb2, AtAxb2);

    memcpy(g, AtAxb2.data, n * sizeof(double));

    return fx;
};

int main(int argc, char** argv)
{
    const char* default_file = "/Users/Usrer/Pictures/3.jpg";
    const char* filename = argc >= 2 ? argv[1] : default_file;

    try {

        cv::Mat src = cv::imread(filename, cv::IMREAD_GRAYSCALE);

        imshow("Original", src);

        const int numImgPixels = src.rows * src.cols;

        const int numGoodPixels = numImgPixels / 10;

        LbfgsContext context;

        context.imageSize.width = src.cols;
        context.imageSize.height = src.rows;

        {
            std::default_random_engine dre;

            context.ri.resize(numGoodPixels);
            for (int j = 0; j < numGoodPixels; ++j)
                context.ri[j] = j;

            std::map<int, int> displaced;

            // Fisher-Yates shuffle Algorithm
            for (int j = 0; j < numGoodPixels; ++j)
            {
                std::uniform_int_distribution<int> di(j, numImgPixels - 1);
                int idx = di(dre);

                if (idx != j)
                {
                    int& to_exchange = (idx < numGoodPixels) ? context.ri[idx] : displaced.try_emplace(idx, idx).first->second;
                    std::swap(context.ri[j], to_exchange);
                }
            }
        }

        context.b.reserve(numGoodPixels);
        for (auto& v : context.ri)
        {
            context.b.push_back(src.reshape(1, 1).at<uint8_t>(v));
        }


        cv::Mat squeezed = cv::Mat::zeros(src.rows, src.cols, CV_8UC1);
        for (int i = 0; i < context.ri.size(); ++i)
        {
            const int idx = context.ri[i];
            squeezed.reshape(1, 1).at<uint8_t>(idx) = context.b[i];
        }

        imshow("Squeezed", squeezed);

        //////////////////////////////////////////////////////////////////////////

        const double param_c = 5;

        // Initialize solution vector
        lbfgsfloatval_t fx;
        lbfgsfloatval_t *x = lbfgs_malloc(numImgPixels);
        if (x == NULL) {
            //
        }
        for (int i = 0; i < numImgPixels; i++) {
            x[i] = 1;
        }

        // Initialize the parameters for the optimization.
        lbfgs_parameter_t param;
        lbfgs_parameter_init(&param);
        param.orthantwise_c = (lbfgsfloatval_t)param_c; // this tells lbfgs to do OWL-QN
        param.linesearch = LBFGS_LINESEARCH_BACKTRACKING;
        int lbfgs_ret = lbfgs(numImgPixels, x, &fx, evaluate, progress, &context, &param);

        cv::Mat Xat2 = cv::Mat(context.imageSize.width, context.imageSize.height, CV_64FC1, const_cast<void*>(static_cast<const void*>(x))).t();
        lbfgs_free(x);

        cv::Mat Xa;
        idct(Xat2, Xa);

        cv::Mat dst;
        Xa.convertTo(dst, CV_8U);

        imshow("Restored", dst);

        cv::waitKey();

    }
    catch (const std::exception& ex) {
        std::cerr << typeid(ex).name() << ": " << ex.what() << '\n';
    }
}
