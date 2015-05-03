#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

using std::vector;
using namespace cv;
//---------------------------------------------------
class WMILTrack{
public:
	WMILTrack();
	~WMILTrack();
private:
	int featureMinNumRect;
	int featureMaxNumRect;
	int featureNum;
	vector<vector<Rect>> features;
	vector<vector<float>> featuresWeight;
	float rOuterPositive;
	vector<Rect> samplePositiveBox;
	vector<Rect> sampleNegativeBox;
	int rSearchWindow;
	Mat imageIntegral;
	Mat samplePositiveFeatureValue;
	Mat sampleNegativeFeatureValue;
	Mat pospred;
	Mat negpred;
	Mat r;
	vector<float> muPositive;
	vector<float> sigmaPositive;
	vector<float> muNegative;
	vector<float> sigmaNegative;
	float learnRate;
	vector<Rect> detectBox;
	vector<float> sampleBoxWeight;
	vector<int> selector;
	Mat detectFeatureValue;
	RNG rng;
	int init_negnumtrain;
	int numSel;
	

private:
	void HaarFeature(Rect& _objectBox, int _numFeature);
	void sampleRect(Mat& _image, 
		            Rect& _objectBox, 
					float _rInner, 
					float _rOuter, 
					int _maxSampleNum, 
					vector<Rect>& _sampleBox);
	void sampleRect(Mat& _image, 
		            Rect& _objectBox, 
					float _srw, 
					vector<Rect>& _sampleBox);
	void getFeatureValue(Mat& _imageIntegral, 
						vector<Rect>& _sampleBox, 
						Mat& _sampleFeatureValue);
	void getFeatureValue(Mat& _imageIntegral, 
		                 vector<Rect>& _sampleBox, 
						 Mat& _sampleFeatureValue,
						 vector<int>& selector);
	void WeakclassifierUpdate(Mat& _sampleFeatureValue, 
							vector<float>& _mu, 
							vector<float>& _sigma, 
							float _learnRate);
	void classifierUpdate(Mat& _sampleFeatureValue, 
						vector<float>& _mu, 
						vector<float>& _sigma, 
						float _learnRate,
						vector<int>& selector);
	void WeakClassifier(vector<float>& _muPos, 
						vector<float>& _sigmaPos, 
						vector<float>& _muNeg, 
						vector<float>& _sigmaNeg,
						Mat& _sampleFeatureValue, 
						Mat&_radio,
						vector<int>& selector);
	void clfWMilBoostUpdate(Mat& _sampleFeatureValuePos, 
										Mat& _sampleFeatureValueNeg,
										Mat&_radioPos,
										Mat&_radioNeg,
										vector<float>& _sampleBoxWeight,
										int numSel,
										vector<int>& selector);
	void PosInstanceWeight(vector<Rect>& _sampleBox,
		                   Rect& _objectBox,
						   vector<float>& _sampleBoxWeight);
	float sigmoidpos(float x);
	float sigmoidneg(float x);

public:
	void processFrame(Mat& _frame, Rect& _objectBox);
	void init(Mat& _frame, Rect& _objectBox);
};