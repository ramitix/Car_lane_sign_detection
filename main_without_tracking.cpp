#include <opencv2/opencv.hpp>
#include "opencv2/highgui.hpp"
#include <iostream>
#include <vector>
#include <string>
#include "opencv2/videoio.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/opencv_modules.hpp"
#include "opencv2/calib3d/calib3d.hpp"

#include <math.h>
#include <iomanip>
using namespace cv;
using namespace std;

long getMatches(const Mat& Car1, const Mat& Car2);
double getPSNR(const Mat& I1, const Mat& I2);
Point GetWrappedPoint(Mat M, const Point& p);
void draw_locations(Mat & img, vector< Rect > & locations, const Scalar & color,string text);


#define VIDEO_FILE_NAME "/Users/ramitix/Documents/opencv_lane_detection/opencv_lane_detection/video.mp4"
#define CASCADE_FILE_NAME "/Users/ramitix/Documents/opencv_lane_detection/opencv_lane_detection/cars3.xml"
//#define CASCADE1_FILE_NAME "/Users/ramitix/Documents/opencv_lane_detection/opencv_lane_detection/traffic_light.xml"
//#define CASCADE2_FILE_NAME "/Users/ramitix/Documents/opencv_lane_detection/opencv_lane_detection/stop_sign.xml"
//#define CASCADE3_FILE_NAME "/Users/ramitix/Documents/opencv_lane_detection/opencv_lane_detection/pedestrian.xml"
#define CASCADE4_FILE_NAME "/Users/ramitix/Documents/opencv_lane_detection/opencv_lane_detection/left-sign.xml"
#define CASCADE5_FILE_NAME "/Users/ramitix/Documents/opencv_lane_detection/opencv_lane_detection/right-sign.xml"

#define CAR_IMAGE "/Users/ramitix/Documents/opencv_lane_detection/opencv_lane_detection/car.png"
#define LEFT_SIGN_IMAGE "/Users/ramitix/Documents/opencv_lane_detection/opencv_lane_detection/left.png"
#define RIGHT_SIGN_IMAGE "/Users/ramitix/Documents/opencv_lane_detection/opencv_lane_detection/right.png"

#define WINDOW_NAME_1 "WINDOW1"
#define WINDOW_NAME_2 "WINDOW2"

int main()
{
    cout << CV_VERSION <<endl;
	VideoCapture cap;
	Mat mFrame, mGray, mCanny, imageROI,mGray1, mGray2, carTrack , mask, IPM_ROI, IPM, IPM_Gray, IPM1, IPM2 ,IPM_Gray2, mFrame2;
	CascadeClassifier cars, traffic_light, stop_sign, pedestrian,sign, sign2;
	vector<Rect> cars_found, traffic_light_found, stop_sign_found ,pedestrian_found ,sign_found, sign_found2, cars_tracking;
    vector<Mat> cars_tracking_img;
    vector<int> car_timer;
    
    

	cars.load(CASCADE_FILE_NAME);
    // traffic_light.load(CASCADE1_FILE_NAME);
    // stop_sign.load(CASCADE2_FILE_NAME);
    // pedestrian.load(CASCADE3_FILE_NAME);
    sign.load(CASCADE4_FILE_NAME);
    sign2.load(CASCADE5_FILE_NAME);
    
	cap.open(VIDEO_FILE_NAME);


    
    double fps = 0;
    int level=0,a=mFrame.rows;
    
    // Number of frames to capture
    int num_frames = 60;
    int started_frames = 0;
    
    // Start and end times
    time_t start, end;
    
    // Variable for storing video frames
    Mat frame;
    
    cout << "Capturing " << num_frames << " frames" << endl ;
    
    // Start time
    time(&start);


	while (cap.read(mFrame))
	{
        
        

        started_frames++;
        
        if (started_frames==num_frames){
        // End Time
        time(&end);
            
        
        // Time elapsed
        double seconds = difftime (end, start);
        cout << "Time taken : " << seconds << " seconds" << endl;
        
        // Calculate frames per second
        fps  = num_frames / seconds;
        
        cout <<"fps : "<<fps<<endl;
        started_frames=0;
        time(&start);
        }
        
        
		// Apply the classifier to the frame
        
        mFrame2 = mFrame.clone();
        imageROI = mFrame(Rect(0,mFrame.rows/2,mFrame.cols,mFrame.rows/2));
        IPM_ROI = imageROI(Rect(0,65,imageROI.cols,(imageROI.rows-65)));
        IPM_ROI = IPM_ROI.clone();


		cvtColor(imageROI, mGray, COLOR_BGR2GRAY);
        cvtColor(mFrame, mGray2, COLOR_BGR2GRAY);
        
        //imshow("before", mGray);
        mGray.copyTo(mGray1);
		//equalizeHist(mGray, mGray);
        //imshow("after", mGray);
        
        //cars cascade
		cars.detectMultiScale(mGray, cars_found, 1.1, 5, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
		draw_locations(mFrame, cars_found, Scalar(0, 255, 0),"Car");
        draw_locations(mFrame2, cars_found, Scalar(0, 255, 0),"Car");
        
//        //traffic lights cascade
//        traffic_light.detectMultiScale(mGray, traffic_light_found, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
//        draw_locations(mFrame, traffic_light_found, Scalar(0, 255, 255),"traffic light");
//        
//        //stop sign cascade
//        stop_sign.detectMultiScale(mGray, stop_sign_found, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
//        draw_locations(mFrame, stop_sign_found, Scalar(0, 0, 255),"Stop Sign");
//        
//        
//        //pedestrian cascade
//        pedestrian.detectMultiScale(mGray, pedestrian_found, 1.1, 1, 0 | CASCADE_SCALE_IMAGE, Size(20,50));
//        draw_locations(mFrame, pedestrian_found, Scalar(255, 0, 0),"Pedestrian");
        
        //stop sign cascade
        sign.detectMultiScale(mGray2, sign_found, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
        draw_locations(mFrame, sign_found, Scalar(0, 143, 255),"Left Arrow");
        
        sign2.detectMultiScale(mGray2, sign_found2, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
        draw_locations(mFrame, sign_found2, Scalar(0, 143, 255),"Right Arrow");
        
        
        
        Point2f inputQuad[4];
        Point2f outputQuad[4];
        
        imshow("3333", IPM_ROI);
        
        Mat IPM_Matrix( 2, 4, CV_32FC1 );
        Mat IPM_Matrix_inverse;
        
        // Set the IPM_Matrix matrix the same type and size as input
        IPM_Matrix = Mat::zeros( mFrame.rows, mFrame.cols, mFrame.type() );
        
        // The 4 points that select quadilateral on the input , from top-left in clockwise order
        // These four pts are the sides of the rect box used as input
        inputQuad[0] = Point2f( 0,0);
        inputQuad[1] = Point2f( IPM_ROI.cols,0);
        inputQuad[2] = Point2f( IPM_ROI.cols,IPM_ROI.rows);
        inputQuad[3] = Point2f( 0,IPM_ROI.rows);           //
        // The 4 points where the mapping is to be done , from top-left in clockwise order
        outputQuad[0] = Point2f( 0,0 );
        outputQuad[1] = Point2f( mFrame.cols,0);
        outputQuad[2] = Point2f( mFrame.cols-250,mFrame.rows);
        outputQuad[3] = Point2f( 250,mFrame.rows);
        
        // Get the Perspective Transform Matrix i.e. IPM_Matrix
        IPM_Matrix = getPerspectiveTransform( inputQuad, outputQuad );
        invert(IPM_Matrix,IPM_Matrix_inverse);

        // Apply the Perspective Transform just found to the src image
 
        warpPerspective(IPM_ROI,IPM,IPM_Matrix,mFrame.size() );


        imshow("IPM", IPM);
        cvtColor(IPM, IPM_Gray, COLOR_BGR2GRAY);
        GaussianBlur(IPM_Gray, IPM_Gray, Size(7,7), 1.5, 1.5);
        //imshow("IPM BEFORE CANNY", IPM_Gray);
        Canny(IPM_Gray, IPM_Gray, 5, 40, 3);
        //imshow("IPM AFTER CANNY", IPM_Gray);
        IPM.copyTo(IPM1);
        IPM.copyTo(IPM2);
        

        //nested loops to eliminate the angled "lines" edges and trim the IPM
        
        for (int i=0; i<IPM_Gray.rows; i++){
            uchar* data= IPM_Gray.ptr<uchar>(i);
            for (int j=0; j<IPM_Gray.cols; j++)
            {
                if(i<0 || i>480)
                {
                    // process each pixel
                    data[j]= data[j]>level?level:0;
                }else{
                    if(data[j]<=255 && data[j]>240 ){
                        for(int m=j;m<j+20;m++){
                            a=m;
                            data[m]=0;
                        }
                        j=a;
                        break;
                    }
                }
            }
        }
        
        for (int i=0; i<IPM_Gray.rows; i++){
            uchar* data= IPM_Gray.ptr<uchar>(i);
            for(int j=IPM_Gray.cols;j>0;j--){
                if(data[j]<=255 && data[j]>240){
                    for(int m=j;m>j-20;m--){
                        data[m]=0;
                    }
                    j=j-20;
                    break;
                }	
            }   
        }

        GaussianBlur( IPM_Gray,IPM_Gray, Size( 5, 5 ), 1.5, 1.5 );
        imshow("IPM BINRARY AFTER FILTERING", IPM_Gray);
      
        
        
        vector<Vec4i> lines;
        HoughLinesP(IPM_Gray,lines,1, 0.01, 120 ,10,600 );
       // HoughLinesP(IPM_Gray,lines,1, 0.01, 120  );

 

        vector<Point> laneShade,laneShade1,laneShade2;
       	float d=0.00,d1=0.00;
        int s=0;
        int n=mFrame.cols;
        Point e,f,g,h,A,B,C,D;
        float angle;float a;
        for( size_t i = 0; i < lines.size(); i++ ){
            float p=0,t=0;
            Vec4i l = lines[i];
            if((l[0]-l[2])==0){
                a=-CV_PI/2;
                angle=-90;
            }else{
                t=(l[1]-l[3])/(l[0]-l[2]);
                a=atan(t);
                angle=a*180/CV_PI;
            }

            if(angle>50 ||  angle<(-50)){
                
                p=(l[0]+l[2])/2;
                line(IPM1,Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255,0,0), 3,CV_AA);
                if(p<320){
                    if(p>s){
                        s=p;
                        d=320-(s);
                        e=Point(l[0],l[1]);
                        f=Point(l[2],l[3]);
                        A= GetWrappedPoint(IPM_Matrix_inverse,e);
                        B =GetWrappedPoint(IPM_Matrix_inverse,f);
                        A.y += 245;
                        B.y += 245;

                        double lengthAB = sqrt((A.x - B.x)*(A.x - B.x) + (A.y - B.y)*(A.y - B.y));
                        if(A.y > B.y){
                        A.x = B.x + (B.x - A.x) / lengthAB * -350;
                        A.y = B.y + (B.y - A.y) / lengthAB * -350;
                            
                        }else{
                            B.x = B.x + (B.x - A.x) / lengthAB * 350;
                            B.y = B.y + (B.y - A.y) / lengthAB * 350;
                        
                        }

                    }
                    
                }
                if(p>320){
                    if(p<n){
                        n=p;
                        d1=(n)-320;
                        g=Point(l[0],l[1]);
                        h=Point(l[2],l[3]);
                        C= GetWrappedPoint(IPM_Matrix_inverse,g);
                        C.y +=245;
                        D =GetWrappedPoint(IPM_Matrix_inverse,h);
                        D.y +=245;
                        double lengthCD = sqrt((C.x - D.x)*(C.x - D.x) + (C.y - D.y)*(C.y - D.y));
                        if(C.x > D.x){
                        C.x = D.x + (D.x - C.x) / lengthCD * -350;
                        C.y = D.y + (D.y - C.y) / lengthCD * -350;
                        }else{
                            D.x = D.x + (D.x - C.x) / lengthCD * +350;
                            D.y = D.y + (D.y - C.y) / lengthCD * +350;
                        
                        }
                    }
                    
                }
                
            }
        }


        line(IPM2,e, f, Scalar(0,255,255), 3,CV_AA);
        line(IPM2,g, h, Scalar(0,145,255), 3,CV_AA);
        
        if(A.x < B.x){
            laneShade.push_back(B);
            laneShade.push_back(A);
        }else{
            laneShade.push_back(A);
            laneShade.push_back(B);
        }
        
        if(C.x > D.x){
            laneShade.push_back(C);
            laneShade.push_back(D);
        }else{
            laneShade.push_back(D);
            laneShade.push_back(C);
        }
        
        laneShade1.push_back(Point((laneShade[0].x+laneShade[3].x)/2,laneShade[0].y+20));
        laneShade1.push_back(Point((laneShade[0].x+laneShade[3].x)/2 +45,laneShade[1].y));
        laneShade1.push_back(Point((laneShade[0].x+laneShade[3].x)/2 -45,laneShade[2].y));
        laneShade1.push_back(Point((laneShade[0].x+laneShade[3].x)/2,laneShade[3].y+20));
        
        laneShade2.push_back(Point((laneShade[0].x+laneShade[3].x)/2,laneShade[0].y+20));
        laneShade2.push_back(Point((laneShade[0].x+laneShade[3].x)/2 +25,laneShade[2].y));
        laneShade2.push_back(Point((laneShade[0].x+laneShade[3].x)/2 -25,laneShade[2].y));
        laneShade2.push_back(Point((laneShade[0].x+laneShade[3].x)/2,laneShade[3].y+20));
        

        Point zero  = Point(0,0);
        if(laneShade[0]!=zero && laneShade[1]!=zero && laneShade[2]!=zero && laneShade[3]!=zero && laneShade[2].y>0){
        Mat laneMask= mFrame.clone();
        fillConvexPoly(laneMask, laneShade, Scalar(0,200,0));  //(255,144,30)
        fillConvexPoly(mFrame, laneShade1, Scalar(0,200,0));
        fillConvexPoly(mFrame, laneShade2, Scalar(255,255,255));
        addWeighted(mFrame, 0.6, laneMask, 0.4, 3, mFrame);
        }

        
        imshow("HOUGH BEFORE FILTERING",IPM1);
        imshow("HOUGH AFTER FILTERING",IPM2);
        
        
        
        

		imshow(WINDOW_NAME_1, mFrame);
        imshow(WINDOW_NAME_2, mFrame2);
        

		waitKey(10);
	}

	return 0;
}

// long getMatches(const Mat& Car1, const Mat& Car2){
    
//     vector<KeyPoint> keypoints1, keypoints2;
//     Mat desc1, desc2;
//     Ptr<AKAZE> akaze = AKAZE::create();
    
//     akaze->detectAndCompute(Car1, Mat(), keypoints1, desc1);
//     akaze->detectAndCompute(Car2, Mat(), keypoints2, desc2);
//     Mat img_keypoints_1,img_keypoints_2;
//     vector< DMatch > good_matches;
    
//     BFMatcher matcher(NORM_L2);
//     std::vector< DMatch > matches;
//     matcher.match( desc1, desc2, matches );
    
//     Mat img_matches;
    
//     if(matches.size()!=0){
//         double max_dist = 0; double min_dist = 250;
//         //   -- Quick calculation of max and min distances between keypoints
//         for( int i = 0; i < desc1.rows; i++ )
//         { double dist = matches[i].distance;
//             if( dist < min_dist ) min_dist = dist;
//             if( dist > max_dist ) max_dist = dist;
//         }
        
//         //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
        
//         for( int i = 0; i < desc1.rows; i++ )
//         { if( matches[i].distance < 3*min_dist )
//         { good_matches.push_back( matches[i]); }
//         }
        
//         drawMatches( Car1, keypoints1, Car2, keypoints2,
//                     good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
//                     vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
        
//         imshow("Matches", img_matches );
        
        


//     }
//     return (good_matches.size());
// }

// double getPSNR(const Mat& I1, const Mat& I2)
// {
//     Mat s1;
//     absdiff(I1, I2, s1);       // |I1 - I2|
//     s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
//     s1 = s1.mul(s1);           // |I1 - I2|^2
    
//     Scalar s = sum(s1);         // sum elements per channel
    
//     double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels
    
//     if( sse <= 1e-10) // for small values return zero
//         return 0;
//     else
//     {
//         double  mse =sse /(double)(I1.channels() * I1.total());
//         double psnr = 10.0*log10((255*255)/mse);
//         return psnr;
//     }
// }

// Point GetWrappedPoint(Mat M, const Point& p)
// {
//     cv::Mat_<double> src(3/*rows*/,1 /* cols */);
    
//     src(0,0)=p.x;
//     src(1,0)=p.y;
//     src(2,0)=1.0;
    
//     cv::Mat_<double> dst = M*src;
//     dst(0,0) /= dst(2,0);
//         dst(1,0) /= dst(2,0);
//     return Point(dst(0,0),dst(1,0));
// }

void draw_locations(Mat & img, vector< Rect > &locations, const Scalar & color, string text)
{

    Mat img1, car, carMask ,carMaskInv,car1,roi1, LeftArrow , LeftMask, RightArrow,RightMask;


    img.copyTo(img1);
    string dis;

	if (!locations.empty())
	{

        double distance= 0;
        
        for( int i = 0 ; i < locations.size() ; ++i){
            
            if (text=="Car"){
                car = imread(CAR_IMAGE);
                carMask = car.clone();
                cvtColor(carMask, carMask, CV_BGR2GRAY);
                locations[i].y = locations[i].y + img.rows/2; // shift the bounding box
                distance = (0.0397*2)/((locations[i].width)*0.00007);// 2 is avg. width of the car
                Size size(locations[i].width/1.5, locations[i].height/3);
                resize(car,car,size, INTER_NEAREST);
                resize(carMask,carMask,size, INTER_NEAREST);
                Mat roi = img.rowRange(locations[i].y-size.height, (locations[i].y+locations[i].height/3)-size.height).colRange(locations[i].x, (locations[i].x  +locations[i].width/1.5));
                bitwise_and(car, roi, car);
                car.setTo(color, carMask);
                add(roi,car,car);
                car.copyTo(img1.rowRange(locations[i].y-size.height, (locations[i].y+locations[i].height/3)-size.height).colRange(locations[i].x, (locations[i].x  +locations[i].width/1.5)));
                
            }else if((text=="Pedestrian")){
                distance = (0.0397*0.5)/((locations[i].width)*0.00007);//0.5 is avg. width of a person
            }else if((text=="Stop Sign")){
                distance = (0.0397*0.75)/((locations[i].width)*0.00007);//0.75 is avg. width of the stopsign
            }else if((text=="Left Arrow")){
                LeftArrow = imread(LEFT_SIGN_IMAGE);
                LeftMask = LeftArrow.clone();
                cvtColor(LeftMask, LeftMask, CV_BGR2GRAY);
                //locations[i].y = locations[i].y + img.rows/2; // shift the bounding box
                Size size(locations[i].width/2, locations[i].height/1.5);
                resize(LeftArrow,LeftArrow,size, INTER_NEAREST);
                resize(LeftMask,LeftMask,size, INTER_NEAREST);
                distance = (0.0397*0.4)/((locations[i].width)*0.00007);//0.35 is avg. width of the   Chevron Arrow sign

                if (locations[i].y-size.height>0){
                    
                Mat roi1 = img.rowRange(locations[i].y-size.height,(locations[i].y+locations[i].height/1.5)-size.height).colRange(locations[i].x+5, (locations[i].x+5+locations[i].width/2));
                bitwise_and(LeftArrow, roi1, LeftArrow);
                LeftArrow.setTo(color, LeftMask);
                add(roi1,LeftArrow,LeftArrow);
                LeftArrow.copyTo(img1.rowRange(locations[i].y-size.height,(locations[i].y+locations[i].height/1.5)-size.height).colRange(locations[i].x+5 ,(locations[i].x +5+locations[i].width/2 )));
                }
                
            }else if((text=="Right Arrow")){
                RightArrow = imread(RIGHT_SIGN_IMAGE);
                RightMask = RightArrow.clone();
                cvtColor(RightMask, RightMask, CV_BGR2GRAY);
                //locations[i].y = locations[i].y + img.rows/2; // shift the bounding box
                Size size(locations[i].width/2, locations[i].height/1.5);
                resize(RightArrow,RightArrow,size, INTER_NEAREST);
                resize(RightMask,RightMask,size, INTER_NEAREST);
                distance = (0.0397*0.4)/((locations[i].width)*0.00007);//0.35 is avg. width of the   Chevron Arrow sign

                if (locations[i].y-size.height>0){
                    
                    Mat roi1 = img.rowRange(locations[i].y-size.height,(locations[i].y+locations[i].height/1.5)-size.height).colRange(locations[i].x+5, (locations[i].x+5+locations[i].width/2));
                    bitwise_and(RightArrow, roi1, RightArrow);
                    RightArrow.setTo(color, RightMask);
                    add(roi1,RightArrow,RightArrow);
                    RightArrow.copyTo(img1.rowRange(locations[i].y-size.height,(locations[i].y+locations[i].height/1.5)-size.height).colRange(locations[i].x+5 ,(locations[i].x +5+locations[i].width/2 )));
                }
                
            }
            stringstream stream;
            stream << fixed << setprecision(2) << distance;
            dis = stream.str() + "m";
            rectangle(img,locations[i], color, -1);
        }
        addWeighted(img1, 0.8, img, 0.2, 0, img);
        
        for( int i = 0 ; i < locations.size() ; ++i){
        
            rectangle(img,locations[i],color,1.8);
            
            putText(img, text, Point(locations[i].x+1,locations[i].y+8), FONT_HERSHEY_DUPLEX, 0.3, color, 1);
            putText(img, dis, Point(locations[i].x,locations[i].y+locations[i].height-5), FONT_HERSHEY_DUPLEX, 0.3, Scalar(255, 255, 255), 1);
            
            
            if (text=="Car"){
                locations[i].y = locations[i].y - img.rows/2; // shift the bounding box
            }
        
        }
        
	}
}




