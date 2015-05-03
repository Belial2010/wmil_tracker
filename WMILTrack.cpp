#include "WMILTrack.h"
#include <math.h>
#include <iostream>
#include <assert.h>
#include <fstream>
#include <sstream>
using namespace cv;
using namespace std;

WMILTrack::WMILTrack(void)
{
	featureMinNumRect = 2;
	featureMaxNumRect = 4;	// number of rectangle from 2 to 4
	featureNum =150 ;	// number of all weaker classifiers, i.e,feature pool
	rOuterPositive = 4.0f;	// radical scope of positive samples
	rSearchWindow =25; // size of search window
	muPositive = vector<float>(featureNum, 0.0f);
	muNegative = vector<float>(featureNum, 0.0f);
	sigmaPositive = vector<float>(featureNum, 1.0f);
	sigmaNegative = vector<float>(featureNum, 1.0f);
	learnRate = 0.85f;	// Learning rate parameter
	init_negnumtrain = 50;//number of trained negative samples
	numSel = 15; //number of selected weak classifier 
}

WMILTrack::~WMILTrack(void)
{
	
}

void WMILTrack::HaarFeature(Rect& _objectBox, 
	                        int _numFeature)
/*Description: compute Haar features
  Arguments:
  -_objectBox: [x y width height] object rectangle
  -_numFeature: total number of features.The default is 50.
*/
{
	features = vector<vector<Rect>>(_numFeature, vector<Rect>());
	featuresWeight = vector<vector<float>>(_numFeature, vector<float>());

	int numRect;
	Rect rectTemp;
	float weightTemp;

	for (int i=0; i<_numFeature; i++)
	{
		numRect = cvFloor(rng.uniform((double)featureMinNumRect, (double)featureMaxNumRect));

		//int c = 1;
		for (int j=0; j<numRect; j++)
		{

			rectTemp.x = cvFloor(rng.uniform(0.0, (double)(_objectBox.width - 3)));
			rectTemp.y = cvFloor(rng.uniform(0.0, (double)(_objectBox.height - 3)));
			rectTemp.width = cvCeil(rng.uniform(0.0, (double)(_objectBox.width - rectTemp.x - 2)));
			rectTemp.height = cvCeil(rng.uniform(0.0, (double)(_objectBox.height - rectTemp.y - 2)));
			features[i].push_back(rectTemp);

			
			//weightTemp = (float)pow(-1.0, c);
			weightTemp = 2*(rng.uniform(0.0,1.0))-1;
			//cout<<weightTemp<<endl;
			featuresWeight[i].push_back(weightTemp);

		}
	}
}

void WMILTrack::sampleRect(Mat& _image, 
							Rect& _objectBox, 
							float _rInner, 
							float _rOuter, 
							int _maxSampleNum, 
							vector<Rect>& _sampleBox)
{
	int rowsz = _image.rows - _objectBox.height - 1;
	int colsz = _image.cols - _objectBox.width - 1;
	float inradsq = _rInner*_rInner;
	float outradsq = _rOuter*_rOuter;


	int dist;

	int minrow = max(0,(int)_objectBox.y-(int)_rInner+1);
	int maxrow = min((int)rowsz-1,(int)_objectBox.y+(int)_rInner);
	int mincol = max(0,(int)_objectBox.x-(int)_rInner+1);
	int maxcol = min((int)colsz-1,(int)_objectBox.x+(int)_rInner);

	int i = 0;

	float prob = ((float)(_maxSampleNum))/(maxrow-minrow+1)/(maxcol-mincol+1);

	int r;
	int c;

	_sampleBox.clear();//important
	Rect rec(0,0,0,0);

	for( r=minrow; r<=(int)maxrow; r++ )
		for( c=mincol; c<=(int)maxcol; c++ )
		{
			dist = (_objectBox.y-r)*(_objectBox.y-r) + (_objectBox.x-c)*(_objectBox.x-c);

			if( rng.uniform(0.,1.)<prob && dist < inradsq && dist >= outradsq ){

				rec.x = c;
				rec.y = r;
				rec.width = _objectBox.width;
				rec.height= _objectBox.height;

				_sampleBox.push_back(rec);				

				i++;
			}
		}

		_sampleBox.resize(i);
}

void WMILTrack::sampleRect(Mat& _image, 
							Rect& _objectBox, 
							float _srw, 
							vector<Rect>& _sampleBox)
{
	int rowsz = _image.rows - _objectBox.height - 1;
	int colsz = _image.cols - _objectBox.width - 1;
	float inradsq = _srw*_srw;	


	int dist;

	int minrow = max(0,(int)_objectBox.y-(int)_srw);
	int maxrow = min((int)rowsz-1,(int)_objectBox.y+(int)_srw);
	int mincol = max(0,(int)_objectBox.x-(int)_srw);
	int maxcol = min((int)colsz-1,(int)_objectBox.x+(int)_srw);

	int i = 0;

	int r;
	int c;

	Rect rec(0,0,0,0);
	_sampleBox.clear();//important

	for( r=minrow; r<=(int)maxrow; r++ )
		for( c=mincol; c<=(int)maxcol; c++ ){
			dist = (_objectBox.y-r)*(_objectBox.y-r) + (_objectBox.x-c)*(_objectBox.x-c);

			if( dist < inradsq ){

				rec.x = c;
				rec.y = r;
				rec.width = _objectBox.width;
				rec.height= _objectBox.height;

				_sampleBox.push_back(rec);				

				i++;
			}
		}

		_sampleBox.resize(i);

}

void WMILTrack::getFeatureValue(Mat& _imageIntegral, 
								vector<Rect>& _sampleBox, 
								Mat& _sampleFeatureValue)
{
	int sampleBoxSize = _sampleBox.size();
	_sampleFeatureValue.create(featureNum, sampleBoxSize, CV_32F);
	float tempValue;
	int xMin;
	int xMax;
	int yMin;
	int yMax;

	for (int i=0; i<featureNum; i++)
	{
		for (int j=0; j<sampleBoxSize; j++)
		{
			tempValue = 0.0f;
			for (size_t k=0; k<features[i].size(); k++)
			{
				xMin = _sampleBox[j].x-1+ features[i][k].x;
				xMax = _sampleBox[j].x-1+ features[i][k].x + features[i][k].width-1;
				yMin = _sampleBox[j].y-1+ features[i][k].y;
				yMax = _sampleBox[j].y-1+ features[i][k].y + features[i][k].height-1;
				tempValue += featuresWeight[i][k] * 
					(_imageIntegral.at<float>(yMin, xMin) +
					_imageIntegral.at<float>(yMax, xMax) -
					_imageIntegral.at<float>(yMin, xMax) -
					_imageIntegral.at<float>(yMax, xMin));
			}
			_sampleFeatureValue.at<float>(i,j) = tempValue;
		}
	}
}

void WMILTrack::getFeatureValue(Mat& _imageIntegral, 
								vector<Rect>& _sampleBox, 
								Mat& _sampleFeatureValue,
								vector<int>& selector)
{
	int sampleBoxSize = _sampleBox.size();
	_sampleFeatureValue.create(selector.size(), sampleBoxSize, CV_32F);
	float tempValue;
	int xMin;
	int xMax;
	int yMin;
	int yMax;

	for (int i=0; i<selector.size(); i++)
	{
		for (int j=0; j<sampleBoxSize; j++)
		{
			tempValue = 0.0f;
			for (size_t k=0; k<features[selector[i]].size(); k++)
			{
				xMin = _sampleBox[j].x-1 + features[selector[i]][k].x;
				xMax = _sampleBox[j].x-1 + features[selector[i]][k].x + features[selector[i]][k].width-1;
				yMin = _sampleBox[j].y-1 + features[selector[i]][k].y;
				yMax = _sampleBox[j].y-1 + features[selector[i]][k].y + features[selector[i]][k].height-1;
				tempValue += featuresWeight[selector[i]][k] * 
					(_imageIntegral.at<float>(yMin, xMin) +
					_imageIntegral.at<float>(yMax, xMax) -
					_imageIntegral.at<float>(yMin, xMax) -
					_imageIntegral.at<float>(yMax, xMin));
			}
			_sampleFeatureValue.at<float>(i,j) = tempValue;
		}
	}
}

void WMILTrack::WeakclassifierUpdate(Mat& _sampleFeatureValue, 
									vector<float>& _mu, 
									vector<float>& _sigma, 
									float _learnRate)
{
	Scalar muTemp;
	Scalar sigmaTemp;

	for (int i=0; i<featureNum; i++)
	{
		meanStdDev(_sampleFeatureValue.row(i), muTemp, sigmaTemp);
		_sigma[i] = (float)sqrt( _learnRate*_sigma[i]*_sigma[i]	+ (1.0f-_learnRate)*sigmaTemp.val[0]*sigmaTemp.val[0]+_learnRate*(1.0f-_learnRate)*(_mu[i]-muTemp.val[0])*(_mu[i]-muTemp.val[0]));	// equation 6 in paper
		_mu[i] = _mu[i]*_learnRate + (1.0f-_learnRate)*muTemp.val[0];	// equation 6 in paper
	}
}

void WMILTrack::classifierUpdate(Mat& _sampleFeatureValue, 
								vector<float>& _mu, 
								vector<float>& _sigma, 
								float _learnRate,
								vector<int>& selector
								)
{
	Scalar muTemp;
	Scalar sigmaTemp;

	for (int i=0; i<numSel; i++)
	{
		meanStdDev(_sampleFeatureValue.row(selector[i]), muTemp, sigmaTemp);
		_sigma[i] = (float)sqrt( _learnRate*_sigma[i]*_sigma[i]	+ (1.0f-_learnRate)*sigmaTemp.val[0]*sigmaTemp.val[0]+_learnRate*(1.0f-_learnRate)*(_mu[i]-muTemp.val[0])*(_mu[i]-muTemp.val[0]));	// equation 6 in paper
		_mu[i] = _mu[i]*_learnRate + (1.0f-_learnRate)*muTemp.val[0];	// equation 6 in paper
	}
}

void WMILTrack::WeakClassifier(vector<float>& _muPos, 
					vector<float>& _sigmaPos, 
					vector<float>& _muNeg, 
					vector<float>& _sigmaNeg,
					Mat& _sampleFeatureValue, 
					Mat&_radio,
					vector<int>& selector)
{
	float sumRadio;
	float pPos;
	float pNeg;
	int sampleBoxNum = _sampleFeatureValue.cols;
	_radio.create(selector.size(),sampleBoxNum,CV_32F);
			
	for (int i=0; i<selector.size(); i++)
	{
		for (int j=0; j<sampleBoxNum; j++)
		{
			//cout<<_sampleFeatureValue.at<float>(selector[i],j)<<endl;
		pPos = exp((_sampleFeatureValue.at<float>(i,j)-_muPos[selector[i]])*(_sampleFeatureValue.at<float>(i,j)-_muPos[selector[i]])/-(2.0f*_sigmaPos[selector[i]]*_sigmaPos[selector[i]]+1e-30) )/(_sigmaPos[selector[i]]+1e-30);
		pNeg = exp( (_sampleFeatureValue.at<float>(i,j)-_muNeg[selector[i]])*(_sampleFeatureValue.at<float>(i,j)-_muNeg[selector[i]])/-(2.0f*_sigmaNeg[selector[i]]*_sigmaNeg[selector[i]]+1e-30) ) / (_sigmaNeg[selector[i]]+1e-30);
		_radio.at<float>(i,j) = log(pPos+1e-30) - log(pNeg+1e-30);
		}
	}
}

void WMILTrack::PosInstanceWeight(vector<Rect>& _sampleBox,
									Rect& _objectBox,
									vector<float>& _sampleBoxWeight )
{
	int sampleBoxSize = _sampleBox.size();
	_sampleBoxWeight.clear();//important
	float weight;
	for (int i=0;i<sampleBoxSize;i++)
	{
		weight=exp(-((_sampleBox[i].x-_objectBox.x)*(_sampleBox[i].x-_objectBox.x)+(_sampleBox[i].y-_objectBox.y)*(_sampleBox[i].y-_objectBox.y))+1e-9f);
		_sampleBoxWeight.push_back(weight);	
	}
}

float WMILTrack::sigmoidpos(float x)
{
	return 1.0f/(1.0f+exp(-x));
}

float WMILTrack::sigmoidneg(float x)
{
	return 1.0f/(1.0f+exp(x));
}

void WMILTrack::clfWMilBoostUpdate(Mat& _sampleFeatureValuePos, 
	                               Mat& _sampleFeatureValueNeg,
								   Mat&_radioPos,
								   Mat&_radioNeg,
								   vector<float>& _sampleBoxWeight,
								   int numSel,
								   vector<int>& selector)
{
	//cout<<_radioPos.at<float>(0,0)<<endl;
	selector.clear();
	int numpos=_sampleFeatureValuePos.cols;
	int numneg=_sampleFeatureValueNeg.cols;
	int liklMinIndex;
	float liklMin;
	float* Hpos=new float[numpos];
	for(int i=0;i<numpos;i++)
		Hpos[i]=0.0f;
	float* Hneg=new float[numneg];
	for(int i=0;i<numneg;i++)
		Hneg[i]=0.0f;

	int _sampleBoxSize=_sampleBoxWeight.size();
	vector<float>psigf;
	psigf.clear();
	float PosTmp;
	vector<float>nsigf;
	nsigf.clear();
	float NegTmp;
	vector<float>pll;
	pll.clear();
	float PllTmp;
	vector<float>nll;
	nll.clear();
	float NllTmp;
	float poslikltmp;
    float neglikltmp;
	vector<float> poslikl;
	poslikl.clear();
	vector<float> neglikl;
	neglikl.clear();
	float likltmp;
	vector<float> likl;
	likl.clear();
	vector<float> liklIndex;
	liklIndex.clear();
	float SumPos,SumNeg;
	selector.clear();
	for(int s=0;s<numSel;s++)
	{
		//cout<<s+1<<"次迭代"<<endl;
		psigf.clear();
		SumPos=0.0f;		
		for (int j=0;j<numpos;j++)
		{
			//cout<<j+1<<endl;
			//cout<<_sampleBoxWeight[j]<<" "<<sigmoidpos(Hpos[j])<<endl;
			//cout<<Hpos[j]<<endl;
			PosTmp=_sampleBoxWeight[j]*sigmoidpos(Hpos[j]);
			//cout<<"Pos="<<PosTmp<<endl;
			SumPos+=PosTmp;
			//cout<<"Sum="<<SumPos<<endl;
			psigf.push_back(PosTmp);
		}
	    assert(psigf.size()==numpos);
		//cout<<"at last sumpos="<<SumPos<<endl;

	
		//cout<<psigf.size()<<endl;
		nsigf.clear();
		SumNeg=0.0f;
		for (int j=0;j<numneg;j++)
		{
			NegTmp=sigmoidneg(Hneg[j]);
			//cout<<NegTmp<<endl;
			SumNeg+=NegTmp;
			nsigf.push_back(NegTmp);
		}
		assert(nsigf.size()==numneg);
		//cout<<"at last sumneg="<<SumNeg<<endl;
		//cout<<"Sum="<<endl;
		//cout<<SumPos<<endl;
		//cout<<SumNeg<<endl;
		pll.clear();
		for (int j=0;j<numpos;j++)
		{
			PllTmp=-(psigf[j]*(1-psigf[j]))/SumPos;
			pll.push_back(PllTmp);
		}
		assert(pll.size()==numpos);

		nll.clear();
		for (int j=0;j<numneg;j++)
		{
			NllTmp=(nsigf[j]*(1-nsigf[j]))/SumNeg;
			nll.push_back(NllTmp);
		}
	    assert(nll.size()==numneg);
		//cout<<"likl"<<endl;
		//cout<<_radioPos.rows<<endl;
		//cout<<_radioPos.cols<<endl;
		poslikl.clear();
		for (int i=0;i<_radioPos.rows;i++)
		{
			poslikltmp=0.0f;
			for (int j=0;j<_radioPos.cols;j++)
			{
				poslikltmp+=_radioPos.at<float>(i,j)*pll[j];				
			}
			poslikl.push_back(poslikltmp);
		}
		assert(poslikl.size()==_radioPos.rows);

		neglikl.clear();
		for (int i=0;i<_radioNeg.rows;i++)
		{
			neglikltmp=0.0f;
			for (int j=0;j<_radioNeg.cols;j++)
			{
				//cout<<_radioNeg.at<float>(i,j)<<endl;
				neglikltmp+=_radioNeg.at<float>(i,j)*nll[j];				
			}
			neglikl.push_back(neglikltmp);
		}
		assert(neglikl.size()==_radioNeg.rows);

		
		//cout<<"likl="<<s<<endl;
		assert(poslikl.size()==neglikl.size());
		assert(poslikl.size()==featureNum);
		likl.clear();
		liklIndex.clear();
		for (int i=0;i<featureNum;i++)
		{
			likltmp=poslikl[i]+neglikl[i];
			likl.push_back(likltmp);
			liklIndex.push_back(i);		
		}
		float tmp;
		int tmpIndex;
		for (int i=1;i<likl.size();i++)
		{
			int j;
			tmp=likl[i];
			tmpIndex=liklIndex[i];
			for (j=i;j>0&&likl[j-1]>tmp;j--)
			{
				likl[j]=likl[j-1];
				liklIndex[j]=liklIndex[j-1];
			}
			likl[j]=tmp;
			liklIndex[j]=tmpIndex;
		}

		//cout<<"Min"<<likl[0]<<endl;
		//cout<<"MinIndex"<<liklIndex[0]+1<<endl;
		//cout<<"MinIndex"<<liklIndex[1]+1<<endl;
		int count=0;
		int sum;
		for (int k=0;k<liklIndex.size();k++)
		{
			sum=0;
			for (int j=0;j<selector.size();j++)
			{
				if(liklIndex[k]==selector[j])
				   sum+=1;
			}
			if (0==sum)
			{
				selector.push_back(liklIndex[k]);
				liklMinIndex=liklIndex[k];
				break;
			}
			
		}
	    //selector.push_back(liklMinIndex);

		//选择likl最小值的索引
		
		for(int i=0;i<numpos;i++)
		{
			Hpos[i]+=_radioPos.at<float>(liklMinIndex,i);
			//cout<<"更新后的"<<Hpos[i]<<endl;
		}

		for(int i=0;i<numneg;i++)
		{
			Hneg[i]+=_radioNeg.at<float>(liklMinIndex,i);			
		}
		//cout<<liklMinIndex+1<<endl;
		//for (int i=0;i<numpos;i++)
		//{
		//	cout<<Hpos[i]<<" "<<Hneg[i]<<endl;
		//}
		//cout<<"  "<<endl;
	}

	delete []Hpos;
	delete []Hneg;
}

void WMILTrack::init(Mat& _frame,Rect& _objectBox)
{
	cout<<_objectBox.x<<endl;
	cout<<_objectBox.y<<endl;
	// compute feature template
	HaarFeature(_objectBox, featureNum);

	// compute sample templates
	sampleRect(_frame, 
		       _objectBox, 
			   rOuterPositive, 
			   0, 
			   100000, 
			   samplePositiveBox);
	cout<<samplePositiveBox[0].x<<endl;
	sampleRect(_frame, 
		       _objectBox, 
			   rSearchWindow*2, 
			   rOuterPositive*1.5, 
			   init_negnumtrain, 
			   sampleNegativeBox);
	cout<<sampleNegativeBox[0].x<<endl;
	//weight of the positive instance 
	
	integral(_frame, imageIntegral, CV_32F);
	cout<<imageIntegral.at<float>(0,0)<<endl;
	selector.clear();
	for (int i=0;i<featureNum;i++)
	{
		selector.push_back(i);
	}
	//for (int i=0;i<selector.size();i++)
	//{
	//	cout<<selector[i]<<"";
	//}
	//getFeatureValue(imageIntegral, samplePositiveBox, samplePositiveFeatureValue);
	//for (int i=0;i<10;i++)
	//{
	//	cout<<samplePositiveFeatureValue.at<float>(0,i)<<endl;
	//}
	//cout<<"new"<<endl;
	getFeatureValue(imageIntegral, samplePositiveBox, samplePositiveFeatureValue,selector);
	//cout<<samplePositiveFeatureValue.at<float>(0,0)<<endl;
	//for (int i=0;i<10;i++)
	//{
	//	cout<<samplePositiveFeatureValue.at<float>(0,i)<<endl;
	//}
	//getFeatureValue(imageIntegral, sampleNegativeBox, sampleNegativeFeatureValue);
	getFeatureValue(imageIntegral, sampleNegativeBox, sampleNegativeFeatureValue,selector);
	WeakclassifierUpdate(samplePositiveFeatureValue, muPositive, sigmaPositive, learnRate);
	WeakclassifierUpdate(sampleNegativeFeatureValue, muNegative, sigmaNegative, learnRate);
	//cout<<samplePositiveFeatureValue.rows<<endl;
	//cout<<samplePositiveFeatureValue.cols<<endl;

	WeakClassifier(muPositive, 
					sigmaPositive, 
					muNegative, 
					sigmaNegative, 
					samplePositiveFeatureValue, 
					pospred,
					selector);
	//cout<<muPositive[0]<<endl;
	//cout<<sigmaPositive[0]<<endl;
	//cout<<muNegative[0]<<endl;
	//cout<<sigmaNegative[0]<<endl;
	//cout<<samplePositiveFeatureValue.at<float>(0,0)<<endl;
	//cout<<pospred.at<float>(0,0)<<endl;
	WeakClassifier(muPositive, 
					sigmaPositive, 
					muNegative, 
					sigmaNegative, 
					sampleNegativeFeatureValue, 
					negpred,
					selector);
	PosInstanceWeight(samplePositiveBox,_objectBox,sampleBoxWeight);
 //   fstream f;
	//f.open("w.txt", fstream::in);
	//float data;
	//sampleBoxWeight.clear();
	//for (int i=0;i<45;i++)
	//{
	//	f>>data;	
	//	sampleBoxWeight.push_back(data);
	//	//cout<<data<<endl;		
	//}
	//cout<<sampleBoxWeight.size()<<endl;
	//f.close();
	//f.open("pos.txt", fstream::in);
	//for (int i=0;i<pospred.rows;i++)
	//{
	//	for (int j=0;j<pospred.cols;j++)
	//	{
	//		f>>pospred.at<float>(i,j);
	//		//cout<<pospred.at<float>(i,j)<<endl;
	//	}	
	//}
	//f.close();
	//f.open("neg.txt",fstream::in);
	//Mat negpred1;
	//negpred1.create(150,42,CV_32F);
	//for (int i=0;i<150;i++)
	//{
	//	for (int j=0;j<42;j++)
	//	{
	//		f>>negpred1.at<float>(i,j);
	//		//cout<<negpred1.at<float>(i,j)<<endl;
	//	}	
	//}
	//f.close();
	//clfWMilBoostUpdate(samplePositiveFeatureValue,
	//	sampleNegativeFeatureValue,   
	//	pospred,
	//	negpred1,
	//	sampleBoxWeight,
	//	numSel,
	//	selector);
	
	
	//sscanf (cstring, "%s %s %s", param1,param2,param3);
	clfWMilBoostUpdate(samplePositiveFeatureValue,
					   sampleNegativeFeatureValue,   
					   pospred,
					   negpred,
					   sampleBoxWeight,
					   numSel,
					   selector);
	for (int i=0;i<selector.size();i++)
	{
		cout<<selector[i]<<endl;
	}

}

void WMILTrack::processFrame(Mat& _frame, 
							 Rect& _objectBox)
{
	// predict
	sampleRect(_frame, 
				_objectBox, 
				rSearchWindow, 
				0, 
				100000, 
				detectBox);
	//sampleRect(_frame, _objectBox, rSearchWindow,detectBox);
	integral(_frame, imageIntegral, CV_32F);
	getFeatureValue(imageIntegral, detectBox, detectFeatureValue,selector);
	WeakClassifier(muPositive, 
					sigmaPositive, 
					muNegative, 
					sigmaNegative, 
					detectFeatureValue, 
					r,
					selector);
	//classifierUpdate(detectFeatureValue, muPositive, sigmaPositive, learnRate,selector);
	vector<float>rmax;
	rmax.clear();
	
	for(int i=0;i<r.cols;i++)
	{
		float tmp=0.0f;
		for(int j=0;j<r.rows;j++)
		{
			tmp+=r.at<float>(j,i);
		}
		rmax.push_back(tmp);
	}
	
	//
	float radioMax=rmax[0];
	int radioMaxIndex=0;
	for(int i=1;i<rmax.size();i++)
	{
		if (rmax[i]>radioMax)
		{
			radioMax=rmax[i];
			radioMaxIndex=i;
		}
	}

	_objectBox = detectBox[radioMaxIndex];


	// update
	sampleRect(_frame, _objectBox, rOuterPositive, 0.0, 1000000, samplePositiveBox);
	sampleRect(_frame, _objectBox, rSearchWindow*1.5, rOuterPositive+4.0, 100, sampleNegativeBox);
	PosInstanceWeight(samplePositiveBox,_objectBox,sampleBoxWeight);
	selector.clear();
	for (int i=0;i<featureNum;i++)
	{
		selector.push_back(i);
	}
	getFeatureValue(imageIntegral, samplePositiveBox, samplePositiveFeatureValue,selector);
	getFeatureValue(imageIntegral, sampleNegativeBox, sampleNegativeFeatureValue,selector);
	WeakclassifierUpdate(samplePositiveFeatureValue, muPositive, sigmaPositive, learnRate);
	WeakclassifierUpdate(sampleNegativeFeatureValue, muNegative, sigmaNegative, learnRate);
	WeakClassifier(muPositive, 
					sigmaPositive, 
					muNegative, 
					sigmaNegative, 
					samplePositiveFeatureValue, 
					pospred,
					selector);
	WeakClassifier(muPositive, 
					sigmaPositive, 
					muNegative, 
					sigmaNegative, 
					sampleNegativeFeatureValue, 
					negpred,
					selector);
	//select the most discriminative weak classifiers 
	clfWMilBoostUpdate(samplePositiveFeatureValue,
						sampleNegativeFeatureValue,   
						pospred,
						negpred,
						sampleBoxWeight,
						numSel,
						selector);
}



