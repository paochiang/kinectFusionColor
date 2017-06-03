#include<string>
#include<windows.h>
#include<iostream>
#include<fstream>
#include<Kinect.h>
#include <NuiKinectFusionApi.h>
#include<opencv2/opencv.hpp>

using namespace std;
using namespace cv;

vector<Matrix4> camerasMatrix;

#ifndef SAFE_FUSION_RELEASE_IMAGE_FRAME
#define SAFE_FUSION_RELEASE_IMAGE_FRAME(p) { if (p) { static_cast<void>(NuiFusionReleaseImageFrame(p)); (p)=NULL; } }
#endif
#ifndef SAFE_DELETE_ARRAY
#define SAFE_DELETE_ARRAY(p) { if (p) { delete[] (p); (p)=NULL; } }
#endif

typedef struct BorderBox {
	float x_min;
	float x_max;
	float y_min;
	float y_max;
	float z_min;
	float z_max;
}BorderBox;

// Safe release for interfaces
template<class Interface>
inline void SafeRelease(Interface *& pInterfaceToRelease)
{
	if (pInterfaceToRelease != NULL)
	{
		pInterfaceToRelease->Release();
		pInterfaceToRelease = NULL;
	}
}
/// Set Identity in a Matrix4
void SetIdentityMatrix(Matrix4 &mat)
{
	mat.M11 = 1; mat.M12 = 0; mat.M13 = 0; mat.M14 = 0;
	mat.M21 = 0; mat.M22 = 1; mat.M23 = 0; mat.M24 = 0;
	mat.M31 = 0; mat.M32 = 0; mat.M33 = 1; mat.M34 = 0;
	mat.M41 = 0; mat.M42 = 0; mat.M43 = 0; mat.M44 = 1;
}

void UpdateIntrinsics(NUI_FUSION_IMAGE_FRAME * pImageFrame, NUI_FUSION_CAMERA_PARAMETERS * params)
{
	if (pImageFrame != nullptr && pImageFrame->pCameraParameters != nullptr && params != nullptr)
	{
		pImageFrame->pCameraParameters->focalLengthX = params->focalLengthX;
		pImageFrame->pCameraParameters->focalLengthY = params->focalLengthY;
		pImageFrame->pCameraParameters->principalPointX = params->principalPointX;
		pImageFrame->pCameraParameters->principalPointY = params->principalPointY;
	}

	// Confirm we are called correctly
	_ASSERT(pImageFrame != nullptr && pImageFrame->pCameraParameters != nullptr && params != nullptr);
}

class KinectFusion
{
public:
	KinectFusion();
	KinectFusion(Mat r, Mat t);
	~KinectFusion();	
	HRESULT CreateFirstConnected();
	HRESULT InitializeKinectFusion();
	HRESULT ResetReconstruction();
	HRESULT SetupUndistortion();
	HRESULT OnCoordinateMappingChanged();
	bool CoordinateChangeCheck();
	HRESULT MapColorToDepth();
	void IncreaseFusion(Mat curDepth, Mat curColor);
	void ProcessFusion();
	void GetResult(const string fileName);
	void CloseSensor() {if (m_pNuiSensor) {m_pNuiSensor->Close();}};
	void SetWorldToCameraT(const Mat r, const Mat t, Matrix4& dst);	
	void DepthToCloud(const UINT16* dephdata, string fileName, string fileName2);
	void FloatDepthToCloud(float* depthData, string fileName, string fileName2);

	int                         m_cFrameCounter;		//camera tracking中已经有的frame的个数
	bool						m_bTrackingFailed;		//camera tracking是否失败标志
	int							m_cLostFrameCounter;	//连续帧之间丢失的帧数

private:
	IKinectSensor*              m_pNuiSensor;			// Current Kinect	
	ICoordinateMapper*          m_pMapper;				//坐标系转换
	WAITABLE_HANDLE             m_coordinateMappingChangedEvent;//坐标系映射变换标志

	static const UINT                         m_cDepthWidth = 512;
	static const UINT                         m_cDepthHeight = 424;
	static const UINT						  m_cColorWidth = 1920;
	static const UINT						  m_cColorHeight = 1080;
	static const int						  cBytesPerPixel = 4; // for depth float and int-per-pixel raycast images
	static const int						  cVisibilityTestQuantShift = 2; // shift by 2 == divide by 4
	static const UINT16						  cDepthVisibilityTestThreshold = 50; //50 mm

	UINT16*									 depthData;	//原始的深度图的值


	INuiFusionColorReconstruction*			 m_pVolume;							//Volume
	NUI_FUSION_CAMERA_PARAMETERS			 m_cameraParameters;				//camera参数 focalx,focaly, principalPointX, principalPointsY;
	Matrix4									 m_worldToCameraTransform;			//global到camera坐标系的转换
	Matrix4									 m_defaultWorldToVolumeTransform;	//默认的global到camera坐标系的转换矩阵	
	NUI_FUSION_RECONSTRUCTION_PARAMETERS	 m_reconstructionParams;	//volume 的参数

	float									 m_fMinDepthThreshold;		//对获得的depth图，通过这两个阈值进行第一步过滤，单位为米
	float									 m_fMaxDepthThreshold;
	unsigned short							 m_cMaxIntegrationWeight;	//将深度图整合进global model时用到的临时平均参数,越小噪声点越多，适合动态；越大融合越慢，细节更多，噪声点更少。
	int										 m_deviceIndex;				//当使用GPU时，选择的设备的索引
	NUI_FUSION_RECONSTRUCTION_PROCESSOR_TYPE m_processorType;			//使用GPU或CPU

	ColorSpacePoint*			m_pColorCoordinates;		//颜色坐标
	DepthSpacePoint*			m_pDepthDistortionMap;		//深度图坐标（每个值代表深度图上的坐标）
	UINT*                       m_pDepthDistortionLT;		////标记camera frame下的坐标是否可以投影到深度图上，若不行，这将对应位置（此处用一维表示）的值设置为很大；若可以，将对应位置值赋值为位置坐标值（变成一维），用于过滤原始深度图
	UINT16*                     m_pDepthImagePixelBuffer;	//经过原始的深度图经过m_pDepthDistortionLT过滤后的深度值，不可取的地方赋值为0
	UINT16*                     m_pDepthVisibilityTestMap;

	NUI_FUSION_IMAGE_FRAME*     m_pDepthFloatImage;			//将m_pDepthImagePixelBuffer用m_fMinDepthThreshold和m_fMaxDepthThreshold过滤后的深度数据。
	NUI_FUSION_IMAGE_FRAME*		m_pSmoothDepthFloatImage;	//将m_pDepthFloatImage平滑后的深度数据帧
	NUI_FUSION_IMAGE_FRAME*     m_pPointCloud;				//第i帧光线投影结果，(作为本帧的深度结果)用于和下一帧进行tracking
	NUI_FUSION_IMAGE_FRAME*     m_pShadedSurface;			//三角化后的模型
	NUI_FUSION_IMAGE_FRAME*     m_pColorImage;				//color数据
	NUI_FUSION_IMAGE_FRAME*		m_pResampledColorImageDepthAligned; //深度图对应的彩色值
	NUI_FUSION_IMAGE_FRAME*		m_pCapturedSurfaceColor;	////第i帧光线投影的点云的颜色

	bool						m_bHaveValidCameraParameters;	//是否有合法的相机参数
	bool                        m_bInitializeError;				//初始化是否成功标志
	bool                        m_bMirrorDepthFrame;			//是否产生深度镜像标志
	bool						m_bTranslateResetPoseByMinDepthThreshold;	//是否将volume往z轴正方向（world frame坐标系）平移
	
	Mat m_r;			//world to camera Rotate matrix
	Mat m_t;			//world to camera Translate matrix
	BorderBox bb;
};

KinectFusion::KinectFusion() {
	m_pNuiSensor = NULL;
	depthData = new UINT16[m_cDepthHeight * m_cDepthWidth];
	m_pMapper = NULL;
	m_coordinateMappingChangedEvent = NULL;
	m_pColorCoordinates = new ColorSpacePoint[m_cDepthHeight * m_cDepthWidth];
	bb.x_min = -0.5f;
	bb.x_max = 0.5f;
	bb.y_min = -0.5f;
	bb.y_max = 0.5f;
	bb.z_min = 0.00f;
	bb.z_max = 1.00f;
	// Define a cubic Kinect Fusion reconstruction volume,
	// with the Kinect at the center of the front face and the volume directly in front of Kinect.
	m_reconstructionParams.voxelsPerMeter = 256;// 1000mm / 256vpm = ~3.9mm/voxel    
	m_reconstructionParams.voxelCountX = (bb.x_max - bb.x_min) * m_reconstructionParams.voxelsPerMeter;   // 384 / 256vpm = 1.5m wide reconstruction
	m_reconstructionParams.voxelCountY = (bb.y_max - bb.y_min) * m_reconstructionParams.voxelsPerMeter;   // Memory = 384*384*384 * 4bytes per voxel
	m_reconstructionParams.voxelCountZ = (bb.z_max - bb.z_min) * m_reconstructionParams.voxelsPerMeter;   // This will require a GPU with at least 256MB

	m_fMinDepthThreshold = NUI_FUSION_DEFAULT_MINIMUM_DEPTH;   // min depth in meters
	m_fMaxDepthThreshold = NUI_FUSION_DEFAULT_MAXIMUM_DEPTH;    // max depth in meters
	// This parameter is the temporal averaging parameter for depth integration into the reconstruction
	m_cMaxIntegrationWeight = NUI_FUSION_DEFAULT_INTEGRATION_WEIGHT;	// Reasonable for static scenes
	m_deviceIndex = -1;	//自动选择GPU设备
	m_processorType = NUI_FUSION_RECONSTRUCTION_PROCESSOR_TYPE_AMP; //使用GPU或CPU

	SetIdentityMatrix(m_worldToCameraTransform);
	SetIdentityMatrix(m_defaultWorldToVolumeTransform);

	// We don't know these at object creation time, so we use nominal values.
	// These will later be updated in response to the CoordinateMappingChanged event.
	m_cameraParameters.focalLengthX = NUI_KINECT_DEPTH_NORM_FOCAL_LENGTH_X;		//这个值在开始是不知道的，但在后面会的得到并需要更新
	m_cameraParameters.focalLengthY = NUI_KINECT_DEPTH_NORM_FOCAL_LENGTH_Y;
	m_cameraParameters.principalPointX = NUI_KINECT_DEPTH_NORM_PRINCIPAL_POINT_X;
	m_cameraParameters.principalPointY = NUI_KINECT_DEPTH_NORM_PRINCIPAL_POINT_Y;
	m_pVolume = NULL;
	m_pDepthFloatImage = nullptr;
	m_pSmoothDepthFloatImage = nullptr;
	m_pDepthImagePixelBuffer = nullptr;
	m_pPointCloud = nullptr;
	m_pShadedSurface = nullptr;
	m_pDepthDistortionMap = nullptr;
	m_pDepthDistortionLT = nullptr;
	m_bHaveValidCameraParameters = false;
	m_bInitializeError = false;
	m_bMirrorDepthFrame = false;
	m_bTrackingFailed = false;
	m_cFrameCounter = 0;
	m_cLostFrameCounter = 0;
	m_bTranslateResetPoseByMinDepthThreshold = true;
	m_pColorImage = nullptr;
	m_pResampledColorImageDepthAligned = nullptr;
	m_pCapturedSurfaceColor = nullptr;
	m_pDepthVisibilityTestMap = nullptr;

}
KinectFusion::KinectFusion(Mat r, Mat t) {
	r.copyTo(m_r);
	t.copyTo(m_t);
	m_pNuiSensor = NULL;
	depthData = new UINT16[m_cDepthHeight * m_cDepthWidth];
	m_pMapper = NULL;
	m_coordinateMappingChangedEvent = NULL;
	m_pColorCoordinates = new ColorSpacePoint[m_cDepthHeight * m_cDepthWidth];
	// Define a cubic Kinect Fusion reconstruction volume,
	// with the Kinect at the center of the front face and the volume directly in front of Kinect.

	bb.x_min = -0.50f;
	bb.x_max = 0.50f;
	bb.y_min = -0.50f;
	bb.y_max = 0.50f;
	bb.z_min = -0.25f;
	bb.z_max = 0.25f;

	m_reconstructionParams.voxelsPerMeter = 256;// 1000mm / 256vpm = ~3.9mm/voxel    
	m_reconstructionParams.voxelCountX = (bb.x_max - bb.x_min) * m_reconstructionParams.voxelsPerMeter;   // 384 / 256vpm = 1.5m wide reconstruction
	m_reconstructionParams.voxelCountY = (bb.y_max - bb.y_min) * m_reconstructionParams.voxelsPerMeter;   // Memory = 384*384*384 * 4bytes per voxel
	m_reconstructionParams.voxelCountZ = (bb.z_max - bb.z_min) * m_reconstructionParams.voxelsPerMeter;   // This will require a GPU with at least 256MB

	m_fMinDepthThreshold = NUI_FUSION_DEFAULT_MINIMUM_DEPTH;   // min depth in meters
	m_fMaxDepthThreshold = NUI_FUSION_DEFAULT_MAXIMUM_DEPTH;    // max depth in meters
	// This parameter is the temporal averaging parameter for depth integration into the reconstruction
	m_cMaxIntegrationWeight = NUI_FUSION_DEFAULT_INTEGRATION_WEIGHT;	// Reasonable for static scenes
	m_deviceIndex = -1;	//自动选择GPU设备
	m_processorType = NUI_FUSION_RECONSTRUCTION_PROCESSOR_TYPE_AMP; //使用GPU或CPU

	//SetIdentityMatrix(m_worldToCameraTransform);
	SetWorldToCameraT(m_r, m_t, m_worldToCameraTransform);
	SetIdentityMatrix(m_defaultWorldToVolumeTransform);

	// We don't know these at object creation time, so we use nominal values.
	// These will later be updated in response to the CoordinateMappingChanged event.
	m_cameraParameters.focalLengthX = NUI_KINECT_DEPTH_NORM_FOCAL_LENGTH_X;		//这个值在开始是不知道的，但在后面会的得到并需要更新
	m_cameraParameters.focalLengthY = NUI_KINECT_DEPTH_NORM_FOCAL_LENGTH_Y;
	m_cameraParameters.principalPointX = NUI_KINECT_DEPTH_NORM_PRINCIPAL_POINT_X;
	m_cameraParameters.principalPointY = NUI_KINECT_DEPTH_NORM_PRINCIPAL_POINT_Y;
	m_pVolume = NULL;
	m_pDepthFloatImage = nullptr;
	m_pSmoothDepthFloatImage = nullptr;
	m_pDepthImagePixelBuffer = nullptr;
	m_pPointCloud = nullptr;
	m_pShadedSurface = nullptr;
	m_pDepthDistortionMap = nullptr;
	m_pDepthDistortionLT = nullptr;
	m_bHaveValidCameraParameters = false;
	m_bInitializeError = false;
	m_bMirrorDepthFrame = false;
	m_bTrackingFailed = false;
	m_cFrameCounter = 0;
	m_cLostFrameCounter = 0;
	m_bTranslateResetPoseByMinDepthThreshold = true;
	m_pColorImage = nullptr;
	m_pResampledColorImageDepthAligned = nullptr;
	m_pCapturedSurfaceColor = nullptr;
	m_pDepthVisibilityTestMap = nullptr;

}
KinectFusion::~KinectFusion() {	
	SafeRelease(m_pMapper);
	if (nullptr != m_pMapper)
		m_pMapper->UnsubscribeCoordinateMappingChanged(m_coordinateMappingChangedEvent);
	if (m_pNuiSensor) {
		m_pNuiSensor->Close();
	}
	SafeRelease(m_pNuiSensor);
	SafeRelease(m_pVolume);
	SAFE_DELETE_ARRAY(depthData);
	SAFE_DELETE_ARRAY(m_pColorCoordinates);
	SAFE_DELETE_ARRAY(m_pDepthDistortionMap);
	SAFE_DELETE_ARRAY(m_pDepthDistortionLT);
	SAFE_DELETE_ARRAY(m_pDepthImagePixelBuffer);
	SAFE_DELETE_ARRAY(m_pDepthVisibilityTestMap);
	SAFE_FUSION_RELEASE_IMAGE_FRAME(m_pDepthFloatImage);
	SAFE_FUSION_RELEASE_IMAGE_FRAME(m_pSmoothDepthFloatImage);
	SAFE_FUSION_RELEASE_IMAGE_FRAME(m_pPointCloud);
	SAFE_FUSION_RELEASE_IMAGE_FRAME(m_pShadedSurface);
	SAFE_FUSION_RELEASE_IMAGE_FRAME(m_pColorImage);
	SAFE_FUSION_RELEASE_IMAGE_FRAME(m_pResampledColorImageDepthAligned);
	SAFE_FUSION_RELEASE_IMAGE_FRAME(m_pCapturedSurfaceColor);

}
HRESULT KinectFusion::CreateFirstConnected()
{
	HRESULT hr;

	hr = GetDefaultKinectSensor(&m_pNuiSensor);
	if (FAILED(hr))
	{
		return hr;
	}

	if (m_pNuiSensor) {
		hr = m_pNuiSensor->Open();
		if (SUCCEEDED(hr))
			hr = m_pNuiSensor->get_CoordinateMapper(&m_pMapper);
		if (SUCCEEDED(hr))
			hr = m_pMapper->SubscribeCoordinateMappingChanged(&m_coordinateMappingChangedEvent);
	}
	if (nullptr == m_pNuiSensor || FAILED(hr))
	{
		cout << "No ready Kinect found!" << endl;
		return E_FAIL;
	}
	return hr;
}
HRESULT KinectFusion::InitializeKinectFusion() {
	HRESULT hr = S_OK;
	//设备检查
	// Check to ensure suitable DirectX11 compatible hardware exists before initializing Kinect Fusion
	WCHAR description[MAX_PATH];    //The description of the device.
	WCHAR instancePath[MAX_PATH];	//The DirectX instance path of the GPU being used for reconstruction.
	UINT memorySize = 0;
	if (FAILED(hr = NuiFusionGetDeviceInfo(m_processorType,	m_deviceIndex,	&description[0], ARRAYSIZE(description),	&instancePath[0],	ARRAYSIZE(instancePath),	&memorySize)))
	{
		if (hr == E_NUI_BADINDEX)
		{
			// This error code is returned either when the device index is out of range for the processor 
			// type or there is no DirectX11 capable device installed. As we set -1 (auto-select default) 
			// for the device index in the parameters, this indicates that there is no DirectX11 capable 
			// device. The options for users in this case are to either install a DirectX11 capable device
			// (see documentation for recommended GPUs) or to switch to non-real-time CPU based 
			// reconstruction by changing the processor type to NUI_FUSION_RECONSTRUCTION_PROCESSOR_TYPE_CPU.
			cout << "No DirectX11 device detected, or invalid device index - Kinect Fusion requires a DirectX11 device for GPU-based reconstruction." << endl;
		}
		else
			cout << "Failed in call to NuiFusionGetDeviceInfo." << endl;
		return hr;
	}
	//创建Fusion 容积重建 Volume
	hr = NuiFusionCreateColorReconstruction(&m_reconstructionParams, m_processorType, m_deviceIndex, &m_worldToCameraTransform, &m_pVolume);

	if (FAILED(hr))
	{
		if (E_NUI_GPU_FAIL == hr)
			cout<<"Device "<<m_deviceIndex<<" not able to run Kinect Fusion, or error initializing."<<endl;
		else if (E_NUI_GPU_OUTOFMEMORY == hr)			
			cout<<"Device "<<m_deviceIndex<<" out of memory error initializing reconstruction - try a smaller reconstruction volume."<<endl;
		else if (NUI_FUSION_RECONSTRUCTION_PROCESSOR_TYPE_CPU != m_processorType)
			cout<<"Failed to initialize Kinect Fusion reconstruction volume on device"<<m_deviceIndex<<endl;
		else
			cout<<"Failed to initialize Kinect Fusion reconstruction volume on CPU."<<endl;
		return hr;
	}

	//保存创建的volume的WorldToVolumeTransform矩阵
	hr = m_pVolume->GetCurrentWorldToVolumeTransform(&m_defaultWorldToVolumeTransform);
	if (FAILED(hr))
	{
		cout<<"Failed in call to GetCurrentWorldToVolumeTransform."<<endl;
		return hr;
	}
	//是否将volume在z轴（基于world frame坐标系）往正方向平移一段voxels(因为kinect最小的depth值也会大于0.5m左右，故就算在volume里重建这一部分，也没有模型)
	if (m_bTranslateResetPoseByMinDepthThreshold)
	{
		//重新构建volume
		hr = ResetReconstruction();
		if (FAILED(hr))
		{
			return hr;
		}
	}

	// 创建浮点深度帧
	hr = NuiFusionCreateImageFrame(NUI_FUSION_IMAGE_TYPE_FLOAT, m_cDepthWidth, m_cDepthHeight, &m_cameraParameters, &m_pDepthFloatImage);
	if (FAILED(hr))
	{
		cout<<"Failed to initialize Kinect Fusion m_pDepthFloatImage."<<endl;
		return hr;
	}
	// 创建平滑浮点深度帧
	hr = NuiFusionCreateImageFrame(NUI_FUSION_IMAGE_TYPE_FLOAT, m_cDepthWidth, m_cDepthHeight, &m_cameraParameters, &m_pSmoothDepthFloatImage);
	if (FAILED(hr))
	{
		cout << "Failed to initialize Kinect Fusion m_pSmoothDepthFloatImage." << endl;
		return hr;
	}
	// 创建光线投影点云帧
	hr = NuiFusionCreateImageFrame(NUI_FUSION_IMAGE_TYPE_POINT_CLOUD, m_cDepthWidth, m_cDepthHeight, &m_cameraParameters, &m_pPointCloud);
	if (FAILED(hr))
	{
		cout<<"Failed to initialize Kinect Fusion m_pPointCloud."<<endl;
		return hr;
	}
	// 表面点云着色帧
	hr = NuiFusionCreateImageFrame(NUI_FUSION_IMAGE_TYPE_COLOR, m_cDepthWidth, m_cDepthHeight, &m_cameraParameters, &m_pShadedSurface);
	if (FAILED(hr))
	{
		cout<<"Failed to initialize Kinect Fusion m_pShadedSurface."<<endl;
		return hr;
	}
	// 创建颜色帧
	hr = NuiFusionCreateImageFrame(NUI_FUSION_IMAGE_TYPE_COLOR, m_cColorWidth, m_cColorHeight, &m_cameraParameters, &m_pColorImage);
	if (FAILED(hr))
	{
		cout << "Failed to initialize Kinect Fusion m_pShadedSurface." << endl;
		return hr;
	}
	// 创建深度图大小的彩色图帧
	if (FAILED(hr = NuiFusionCreateImageFrame(NUI_FUSION_IMAGE_TYPE_COLOR, m_cDepthWidth, m_cDepthHeight, &m_cameraParameters, &m_pResampledColorImageDepthAligned)))
	{
		cout << "Failed to initialize Kinect Fusion m_pShadedSurface"<<endl;
		return hr;
	}
	//创建光线投影点云帧颜色
	if (FAILED(hr = NuiFusionCreateImageFrame(NUI_FUSION_IMAGE_TYPE_COLOR, m_cDepthWidth, m_cDepthHeight, &m_cameraParameters, &m_pCapturedSurfaceColor)))
	{
		return hr;
	}

	//分配深度图相关的内存
	_ASSERT(m_pDepthImagePixelBuffer == nullptr);
	m_pDepthImagePixelBuffer = new(std::nothrow) UINT16[m_cDepthHeight * m_cDepthWidth];
	if (nullptr == m_pDepthImagePixelBuffer)
	{
		cout<<"Failed to initialize Kinect Fusion depth image pixel buffer."<<endl;
		return hr;
	}
	_ASSERT(m_pDepthDistortionMap == nullptr);
	m_pDepthDistortionMap = new(std::nothrow) DepthSpacePoint[m_cDepthHeight * m_cDepthWidth];
	if (nullptr == m_pDepthDistortionMap)
	{
		cout<<"Failed to initialize Kinect Fusion depth image distortion buffer."<<endl;
		return E_OUTOFMEMORY;
	}
	SAFE_DELETE_ARRAY(m_pDepthDistortionLT);
	m_pDepthDistortionLT = new(std::nothrow) UINT[m_cDepthHeight * m_cDepthWidth];
	if (nullptr == m_pDepthDistortionLT)
	{
		cout<<"Failed to initialize Kinect Fusion depth image distortion Lookup Table."<<endl;
		return E_OUTOFMEMORY;
	}


	SAFE_DELETE_ARRAY(m_pDepthVisibilityTestMap);
	m_pDepthVisibilityTestMap = new(std::nothrow) UINT16[(m_cColorWidth >> cVisibilityTestQuantShift) * (m_cColorHeight >> cVisibilityTestQuantShift)];

	if (nullptr == m_pDepthVisibilityTestMap)
	{
		cout << "Failed to initialize Kinect Fusion depth points visibility test buffer." << endl;
		return E_OUTOFMEMORY;
	}
	// If we have valid parameters, let's go ahead and use them.
	if (m_cameraParameters.focalLengthX != 0)
		SetupUndistortion();
	return hr;
}
HRESULT KinectFusion::ResetReconstruction()
{
	if (nullptr == m_pVolume)
		return E_FAIL;
	HRESULT hr = S_OK;
	if (m_r.empty() || m_t.empty()) {
		SetIdentityMatrix(m_worldToCameraTransform);
		cout << "non ready r and t!" << endl;
	}
	else {
		SetWorldToCameraT(m_r, m_t, m_worldToCameraTransform);
	}

	Matrix4 worldToVolumeTransform = m_defaultWorldToVolumeTransform;

	worldToVolumeTransform.M11 = 256; worldToVolumeTransform.M12 = 0; worldToVolumeTransform.M13 = 0; worldToVolumeTransform.M14 = 0;
	worldToVolumeTransform.M21 = 0; worldToVolumeTransform.M22 = 256; worldToVolumeTransform.M23 = 0; worldToVolumeTransform.M24 = 0;
	worldToVolumeTransform.M31 = 0; worldToVolumeTransform.M32 = 0; worldToVolumeTransform.M33 = 256; worldToVolumeTransform.M34 = 0;
	worldToVolumeTransform.M41 = -(bb.x_min * m_reconstructionParams.voxelsPerMeter); worldToVolumeTransform.M42 = -(bb.y_min * m_reconstructionParams.voxelsPerMeter); worldToVolumeTransform.M43 = -(bb.z_min * m_reconstructionParams.voxelsPerMeter); worldToVolumeTransform.M44 = 1;

	m_defaultWorldToVolumeTransform.M11 = 256; m_defaultWorldToVolumeTransform.M12 = 0; m_defaultWorldToVolumeTransform.M13 = 0; m_defaultWorldToVolumeTransform.M14 = 0;
	m_defaultWorldToVolumeTransform.M21 = 0; m_defaultWorldToVolumeTransform.M22 = 256; m_defaultWorldToVolumeTransform.M23 = 0; m_defaultWorldToVolumeTransform.M24 = 0;
	m_defaultWorldToVolumeTransform.M31 = 0; m_defaultWorldToVolumeTransform.M32 = 0; m_defaultWorldToVolumeTransform.M33 = 256; m_defaultWorldToVolumeTransform.M34 = 0;
	m_defaultWorldToVolumeTransform.M41 = -(bb.x_min * m_reconstructionParams.voxelsPerMeter); m_defaultWorldToVolumeTransform.M42 = -(bb.y_min * m_reconstructionParams.voxelsPerMeter); m_defaultWorldToVolumeTransform.M43 = -(bb.z_min * m_reconstructionParams.voxelsPerMeter); m_defaultWorldToVolumeTransform.M44 = 1;

	//将volume在Z轴上往正方向平移最小深度大小的voxels，这样对于最小深度的world points，它在volume里的z轴对应为0.（volume可以理解一个操作区域，只对此区域内的点云进行处理）
	if (m_bTranslateResetPoseByMinDepthThreshold)
	{
		
		// 将volume在Z轴上平移最小深度大小的voxels(从0到最小深度内不会有重建的模型，故不需要再这些部分创建volume)
		//float minDist = (m_fMinDepthThreshold < m_fMaxDepthThreshold) ? m_fMinDepthThreshold : m_fMaxDepthThreshold;
		//worldToVolumeTransform.M43 -= (minDist * m_reconstructionParams.voxelsPerMeter);
		//worldToVolumeTransform.M42 += (0.05 * m_reconstructionParams.voxelsPerMeter);
		//worldToVolumeTransform.M41 += (0.06 * m_reconstructionParams.voxelsPerMeter);
		hr = m_pVolume->ResetReconstruction(&m_worldToCameraTransform, &worldToVolumeTransform);
		m_pVolume->GetCurrentWorldToVolumeTransform(&worldToVolumeTransform);
		m_pVolume->GetCurrentWorldToCameraTransform(&m_worldToCameraTransform);
	}
	else
		hr = m_pVolume->ResetReconstruction(&m_worldToCameraTransform, &worldToVolumeTransform);
	m_cLostFrameCounter = 0;
	m_cFrameCounter = 0;
	if (SUCCEEDED(hr))
	{
		m_bTrackingFailed = false;
		cout << "Reconstruction has been reset." << endl;
	}
	else
		cout << "Failed to reset reconstruction." << endl;
	return hr;
}
HRESULT KinectFusion::SetupUndistortion()
{
	HRESULT hr = E_UNEXPECTED;

	//深度图坐标系原点不能在图像中心，否则此摄像机参数就不合法
	if (m_cameraParameters.principalPointX != 0)
	{
		//深度图的四个边坐标：左上（0，0），右上（1，0）（因为k矩阵里参数分别都除以深度图的宽和高），左下（0，1），右下（1，1）反投影成camera frame 下且z为1（1m）的空间点
		CameraSpacePoint cameraFrameCorners[4] = //at 1 meter distance. Take into account that depth frame is mirrored
		{
			/*LT*/{ -m_cameraParameters.principalPointX / m_cameraParameters.focalLengthX, m_cameraParameters.principalPointY / m_cameraParameters.focalLengthY, 1.f },
			/*RT*/{ (1.f - m_cameraParameters.principalPointX) / m_cameraParameters.focalLengthX, m_cameraParameters.principalPointY / m_cameraParameters.focalLengthY, 1.f },
			/*LB*/{ -m_cameraParameters.principalPointX / m_cameraParameters.focalLengthX, (m_cameraParameters.principalPointY - 1.f) / m_cameraParameters.focalLengthY, 1.f },
			/*RB*/{ (1.f - m_cameraParameters.principalPointX) / m_cameraParameters.focalLengthX, (m_cameraParameters.principalPointY - 1.f) / m_cameraParameters.focalLengthY, 1.f }
		};

		//将4个1m处的空间点边界内的空间划分为个数和深度图大小相同的空间点，然后将这些点投影回深度图上。
		for (UINT rowID = 0; rowID < m_cDepthHeight; rowID++)
		{
			const float rowFactor = float(rowID) / float(m_cDepthHeight - 1);
			const CameraSpacePoint rowStart =
			{
				cameraFrameCorners[0].X + (cameraFrameCorners[2].X - cameraFrameCorners[0].X) * rowFactor,
				cameraFrameCorners[0].Y + (cameraFrameCorners[2].Y - cameraFrameCorners[0].Y) * rowFactor,
				1.f
			};

			const CameraSpacePoint rowEnd =
			{
				cameraFrameCorners[1].X + (cameraFrameCorners[3].X - cameraFrameCorners[1].X) * rowFactor,
				cameraFrameCorners[1].Y + (cameraFrameCorners[3].Y - cameraFrameCorners[1].Y) * rowFactor,
				1.f
			};

			const float stepFactor = 1.f / float(m_cDepthWidth - 1);
			const CameraSpacePoint rowDelta =
			{
				(rowEnd.X - rowStart.X) * stepFactor,
				(rowEnd.Y - rowStart.Y) * stepFactor,
				0
			};

			_ASSERT(m_cDepthWidth == NUI_DEPTH_RAW_WIDTH);
			CameraSpacePoint cameraCoordsRow[NUI_DEPTH_RAW_WIDTH];

			CameraSpacePoint currentPoint = rowStart;
			for (UINT i = 0; i < m_cDepthWidth; i++)
			{
				cameraCoordsRow[i] = currentPoint;
				currentPoint.X += rowDelta.X;
				currentPoint.Y += rowDelta.Y;
			}

			hr = m_pMapper->MapCameraPointsToDepthSpace(m_cDepthWidth, cameraCoordsRow, m_cDepthWidth, &m_pDepthDistortionMap[rowID * m_cDepthWidth]);
			if (FAILED(hr))
			{
				cout<<"Failed to initialize Kinect Coordinate Mapper."<<endl;
				return hr;
			}
		}

		if (nullptr == m_pDepthDistortionLT)
		{
			cout<<"Failed to initialize Kinect Fusion depth image distortion Lookup Table."<<endl;
			return E_OUTOFMEMORY;
		}

		//若反投影回的深度图位置不合法，这将此处位置的深度图标记为不可从空间坐标投影回来，用于后面过滤采集到的深度图
		UINT* pLT = m_pDepthDistortionLT;
		for (UINT i = 0; i < m_cDepthHeight * m_cDepthWidth; i++, pLT++)
		{
			//nearest neighbor depth lookup table 
			UINT x = UINT(m_pDepthDistortionMap[i].X + 0.5f);
			UINT y = UINT(m_pDepthDistortionMap[i].Y + 0.5f);

			*pLT = (x < m_cDepthWidth && y < m_cDepthHeight) ? x + y * m_cDepthWidth : UINT_MAX;
		}
		m_bHaveValidCameraParameters = true;
	}
	else
	{
		m_bHaveValidCameraParameters = false;
	}
	return S_OK;
}
HRESULT KinectFusion::OnCoordinateMappingChanged()
{
	HRESULT hr = E_UNEXPECTED;

	// Calculate the down sampled image sizes, which are used for the AlignPointClouds calculation frames
	CameraIntrinsics intrinsics = {};

	m_pMapper->GetDepthCameraIntrinsics(&intrinsics);

	float focalLengthX = intrinsics.FocalLengthX / NUI_DEPTH_RAW_WIDTH;
	float focalLengthY = intrinsics.FocalLengthY / NUI_DEPTH_RAW_HEIGHT;
	float principalPointX = intrinsics.PrincipalPointX / NUI_DEPTH_RAW_WIDTH;
	float principalPointY = intrinsics.PrincipalPointY / NUI_DEPTH_RAW_HEIGHT;

	if (m_cameraParameters.focalLengthX == focalLengthX && m_cameraParameters.focalLengthY == focalLengthY &&
		m_cameraParameters.principalPointX == principalPointX && m_cameraParameters.principalPointY == principalPointY)
		return S_OK;

	m_cameraParameters.focalLengthX = focalLengthX;
	m_cameraParameters.focalLengthY = focalLengthY;
	m_cameraParameters.principalPointX = principalPointX;
	m_cameraParameters.principalPointY = principalPointY;

	_ASSERT(m_cameraParameters.focalLengthX != 0);

	UpdateIntrinsics(m_pDepthFloatImage, &m_cameraParameters);
	UpdateIntrinsics(m_pPointCloud, &m_cameraParameters);
	UpdateIntrinsics(m_pShadedSurface, &m_cameraParameters);
	UpdateIntrinsics(m_pSmoothDepthFloatImage, &m_cameraParameters);
	UpdateIntrinsics(m_pColorImage, &m_cameraParameters);
	UpdateIntrinsics(m_pResampledColorImageDepthAligned, &m_cameraParameters);
	UpdateIntrinsics(m_pCapturedSurfaceColor, &m_cameraParameters);

	if (nullptr == m_pDepthDistortionMap)
	{
		cout << "Failed to initialize Kinect Fusion depth image distortion buffer." << endl;
		return E_OUTOFMEMORY;
	}

	hr = SetupUndistortion();
	return hr;
}
bool KinectFusion::CoordinateChangeCheck() {
	if (nullptr == m_pNuiSensor)
	{
		cout << "cannot get kinect sensor!" << endl;
		exit(0);
	}
	//检查相机参数变化
	if (m_coordinateMappingChangedEvent != NULL && WAIT_OBJECT_0 == WaitForSingleObject((HANDLE)m_coordinateMappingChangedEvent, 0))
	{
		cout << "camere corrdinate map chainge!" << endl;
		OnCoordinateMappingChanged();
		ResetEvent((HANDLE)m_coordinateMappingChangedEvent);
		return true;
	}
	return false;
}
void KinectFusion::SetWorldToCameraT(const Mat r, const Mat t, Matrix4& dst) {
	if (r.rows != 3 || r.cols != 3 || r.type() != CV_32FC1
		|| t.rows != 3 || t.cols != 1 || t.type() != CV_32FC1)
	{
		std::cout << "SetWorldToCameraT: check inputs!(r, t format error!)" << std::endl;
		exit(0);
	}

	dst.M11 = r.at<float>(0, 0); dst.M12 = r.at<float>(1, 0); dst.M13 = r.at<float>(2, 0); dst.M14 = 0.0f;
	dst.M21 = r.at<float>(0, 1); dst.M22 = r.at<float>(1, 1); dst.M23 = r.at<float>(2, 1); dst.M24 = 0.0f;
	dst.M31 = r.at<float>(0, 2); dst.M32 = r.at<float>(1, 2); dst.M33 = r.at<float>(2, 2); dst.M34 = 0.0f;
	dst.M41 = t.at<float>(0, 0); dst.M42 = t.at<float>(1, 0); dst.M43 = t.at<float>(2, 0); dst.M44 = 1.0f;
}
HRESULT KinectFusion::MapColorToDepth()
{
	HRESULT hr = S_OK;

	if (nullptr == m_pColorImage || nullptr == m_pResampledColorImageDepthAligned
		|| nullptr == m_pColorCoordinates || nullptr == m_pDepthVisibilityTestMap)
	{
		return E_FAIL;
	}

	NUI_FUSION_BUFFER *srcColorBuffer = m_pColorImage->pFrameBuffer;
	NUI_FUSION_BUFFER *destColorBuffer = m_pResampledColorImageDepthAligned->pFrameBuffer;

	if (nullptr == srcColorBuffer || nullptr == destColorBuffer)
	{
		cout << "Error accessing color textures." << endl;
		return E_NOINTERFACE;
	}

	if (FAILED(hr) || srcColorBuffer->Pitch == 0)
	{
		cout << "Error accessing color texture pixels." << endl;
		return  E_FAIL;
	}

	if (FAILED(hr) || destColorBuffer->Pitch == 0)
	{
		cout << "Error accessing color texture pixels." << endl;
		return  E_FAIL;
	}

	int *rawColorData = reinterpret_cast<int*>(srcColorBuffer->pBits);
	int *colorDataInDepthFrame = reinterpret_cast<int*>(destColorBuffer->pBits);

	// Get the coordinates to convert color to depth space
	hr = m_pMapper->MapDepthFrameToColorSpace(NUI_DEPTH_RAW_WIDTH * NUI_DEPTH_RAW_HEIGHT, m_pDepthImagePixelBuffer,
		NUI_DEPTH_RAW_WIDTH * NUI_DEPTH_RAW_HEIGHT, m_pColorCoordinates);

	if (FAILED(hr))
	{
		return hr;
	}

	// construct dense depth points visibility test map so we can test for depth points that are invisible in color space
	const UINT16* const pDepthEnd = m_pDepthImagePixelBuffer + NUI_DEPTH_RAW_WIDTH * NUI_DEPTH_RAW_HEIGHT;
	const ColorSpacePoint* pColorPoint = m_pColorCoordinates;
	const UINT testMapWidth = UINT(m_cColorWidth >> cVisibilityTestQuantShift);
	const UINT testMapHeight = UINT(m_cColorHeight >> cVisibilityTestQuantShift);
	ZeroMemory(m_pDepthVisibilityTestMap, testMapWidth * testMapHeight * sizeof(UINT16));
	for (const UINT16* pDepth = m_pDepthImagePixelBuffer; pDepth < pDepthEnd; pDepth++, pColorPoint++)
	{
		const UINT x = UINT(pColorPoint->X + 0.5f) >> cVisibilityTestQuantShift;
		const UINT y = UINT(pColorPoint->Y + 0.5f) >> cVisibilityTestQuantShift;
		if (x < testMapWidth && y < testMapHeight)
		{
			const UINT idx = y * testMapWidth + x;
			const UINT16 oldDepth = m_pDepthVisibilityTestMap[idx];
			const UINT16 newDepth = *pDepth;
			if (!oldDepth || oldDepth > newDepth)
			{
				m_pDepthVisibilityTestMap[idx] = newDepth;
			}
		}
	}


	// Loop over each row and column of the destination color image and copy from the source image
	// Note that we could also do this the other way, and convert the depth pixels into the color space, 
	// avoiding black areas in the converted color image and repeated color images in the background
	// However, then the depth would have radial and tangential distortion like the color camera image,
	// which is not ideal for Kinect Fusion reconstruction.

	if (m_bMirrorDepthFrame)
	{
		for (unsigned int y = 0; y < m_cDepthHeight; y++)
		{
			const UINT depthWidth = m_cDepthWidth;
			const UINT depthImagePixels = m_cDepthHeight * m_cDepthWidth;
			const UINT colorHeight = m_cColorHeight;
			const UINT colorWidth = m_cColorWidth;
			const UINT testMapWidth = UINT(colorWidth >> cVisibilityTestQuantShift);

			UINT destIndex = y * depthWidth;
			for (UINT x = 0; x < depthWidth; ++x, ++destIndex)
			{
				int pixelColor = 0;
				const UINT mappedIndex = m_pDepthDistortionLT[destIndex];
				if (mappedIndex < depthImagePixels)
				{
					// retrieve the depth to color mapping for the current depth pixel
					const ColorSpacePoint colorPoint = m_pColorCoordinates[mappedIndex];

					// make sure the depth pixel maps to a valid point in color space
					const UINT colorX = (UINT)(colorPoint.X + 0.5f);
					const UINT colorY = (UINT)(colorPoint.Y + 0.5f);
					if (colorX < colorWidth && colorY < colorHeight)
					{
						const UINT16 depthValue = m_pDepthImagePixelBuffer[mappedIndex];
						const UINT testX = colorX >> cVisibilityTestQuantShift;
						const UINT testY = colorY >> cVisibilityTestQuantShift;
						const UINT testIdx = testY * testMapWidth + testX;
						const UINT16 depthTestValue = m_pDepthVisibilityTestMap[testIdx];
						_ASSERT(depthValue >= depthTestValue);
						if (depthValue - depthTestValue < cDepthVisibilityTestThreshold)
						{
							// calculate index into color array
							const UINT colorIndex = colorX + (colorY * colorWidth);
							pixelColor = rawColorData[colorIndex];
						}
					}
				}
				colorDataInDepthFrame[destIndex] = pixelColor;
			}
		}
	}
	else
	{
		for (unsigned int y = 0; y < m_cDepthHeight; y++)
		{
			const UINT depthWidth = m_cDepthWidth;
			const UINT depthImagePixels = m_cDepthHeight * m_cDepthWidth;
			const UINT colorHeight = m_cColorHeight;
			const UINT colorWidth = m_cColorWidth;
			const UINT testMapWidth = UINT(colorWidth >> cVisibilityTestQuantShift);

			// Horizontal flip the color image as the standard depth image is flipped internally in Kinect Fusion
			// to give a viewpoint as though from behind the Kinect looking forward by default.
			UINT destIndex = y * depthWidth;
			UINT flipIndex = destIndex + depthWidth - 1;
			for (UINT x = 0; x < depthWidth; ++x, ++destIndex, --flipIndex)
			{
				int pixelColor = 0;
				const UINT mappedIndex = m_pDepthDistortionLT[destIndex];
				if (mappedIndex < depthImagePixels)
				{
					// retrieve the depth to color mapping for the current depth pixel
					const ColorSpacePoint colorPoint = m_pColorCoordinates[mappedIndex];

					// make sure the depth pixel maps to a valid point in color space
					const UINT colorX = (UINT)(colorPoint.X + 0.5f);
					const UINT colorY = (UINT)(colorPoint.Y + 0.5f);
					if (colorX < colorWidth && colorY < colorHeight)
					{
						const UINT16 depthValue = m_pDepthImagePixelBuffer[mappedIndex];
						const UINT testX = colorX >> cVisibilityTestQuantShift;
						const UINT testY = colorY >> cVisibilityTestQuantShift;
						const UINT testIdx = testY * testMapWidth + testX;
						const UINT16 depthTestValue = m_pDepthVisibilityTestMap[testIdx];
						_ASSERT(depthValue >= depthTestValue);
						if (depthValue - depthTestValue < cDepthVisibilityTestThreshold)
						{
							// calculate index into color array
							const UINT colorIndex = colorX + (colorY * colorWidth);
							pixelColor = rawColorData[colorIndex];
						}
					}
				}
				colorDataInDepthFrame[flipIndex] = pixelColor;
			}
		}
	}

	return hr;
}

void KinectFusion::IncreaseFusion(Mat curDepth, Mat curColor) {
	//get depth data
	for (int row = 0; row < m_cDepthHeight; row++) {
		for (int col = 0; col < m_cDepthWidth; col++) {
			depthData[row * m_cDepthWidth + col] = curDepth.at<unsigned short>(row, col);
		}
	}
	//get color data
	for (int i = 0; i < m_cColorHeight * m_cColorWidth; i++) {
		m_pColorImage->pFrameBuffer->pBits[i * 4] = curColor.data[i * 4];
		m_pColorImage->pFrameBuffer->pBits[i * 4 + 1] = curColor.data[i * 4 + 1];
		m_pColorImage->pFrameBuffer->pBits[i * 4 + 2] = curColor.data[i * 4 + 2];
		m_pColorImage->pFrameBuffer->pBits[i * 4 + 3] = curColor.data[i * 4 + 3];
	}

	//remap depth
	const UINT bufferLength = m_cDepthHeight * m_cDepthWidth;
	UINT16 * pDepth = m_pDepthImagePixelBuffer;
	for (UINT i = 0; i < bufferLength; i++, pDepth++)
	{
		const UINT id = m_pDepthDistortionLT[i];
		*pDepth = id < bufferLength ? depthData[id] : 0;
	}

	//将深度图映射到彩色图空间
	HRESULT hr = m_pMapper->MapDepthFrameToColorSpace(m_cDepthWidth * m_cDepthHeight, m_pDepthImagePixelBuffer, m_cDepthWidth * m_cDepthHeight, m_pColorCoordinates);
	
	if (SUCCEEDED(hr)) {
		//得到与深度图对其的彩色图
		MapColorToDepth();

		//kinectfusion操作
		ProcessFusion();
		//Matrix4 worldToVolumeTransform;
		//m_pVolume->GetCurrentWorldToVolumeTransform(&worldToVolumeTransform);

		//cout << "11" << endl;
	}	
}

void KinectFusion::DepthToCloud(const UINT16* dephdata, string fileName, string fileName2) {

	CameraSpacePoint* csp = new CameraSpacePoint[m_cDepthHeight * m_cDepthWidth];
	HRESULT hr = m_pMapper->MapDepthFrameToCameraSpace(m_cDepthHeight * m_cDepthWidth, dephdata, m_cDepthHeight * m_cDepthWidth, csp);

	int count = 0;
	for (int i = 0; i < m_cDepthHeight * m_cDepthWidth; i++)
	{
		CameraSpacePoint p = csp[i];
		if (p.X != -std::numeric_limits<float>::infinity() && p.Y != -std::numeric_limits<float>::infinity() && p.Z != -std::numeric_limits<float>::infinity())
		{
			count++;
		}
	}

	ofstream ofs;
	ofs.open(fileName);
	ofstream ofs2;
	ofs2.open(fileName2);
	if (!ofs2.is_open()) {
		cout << "ofs2 open file error!" << endl;
	}
	if (ofs.is_open() && ofs2.is_open()) {
		ofs << "ply\n";
		ofs << "format ascii 1.0\n";
		ofs << "element vertex " << count << "\n";
		ofs << "property float x\n";
		ofs << "property float y\n";
		ofs << "property float z\n";
		ofs << "property float nx\n";
		ofs << "property float ny\n";
		ofs << "property float nz\n";
		ofs << "property uchar diffuse_red\n";
		ofs << "property uchar diffuse_green\n";
		ofs << "property uchar diffuse_blue\n";
		ofs << "property uchar alpha\n";
		ofs << "end_header\n";

		ofs2 << "ply\n";
		ofs2 << "format ascii 1.0\n";
		ofs2 << "element vertex " << count << "\n";
		ofs2 << "property float x\n";
		ofs2 << "property float y\n";
		ofs2 << "property float z\n";
		ofs2 << "property float nx\n";
		ofs2 << "property float ny\n";
		ofs2 << "property float nz\n";
		ofs2 << "property uchar diffuse_red\n";
		ofs2 << "property uchar diffuse_green\n";
		ofs2 << "property uchar diffuse_blue\n";
		ofs2 << "property uchar alpha\n";
		ofs2 << "end_header\n";

		for (int i = 0; i < m_cDepthHeight * m_cDepthWidth; i++)
		{
			
			CameraSpacePoint p = csp[i];
			if (p.X != -std::numeric_limits<float>::infinity() && p.Y != -std::numeric_limits<float>::infinity() && p.Z != -std::numeric_limits<float>::infinity())
			{
				float cameraX = static_cast<float>(p.X);
				float cameraY = static_cast<float>(p.Y);
				float cameraZ = static_cast<float>(p.Z);
				ofs << cameraX << " " << cameraY << " " << cameraZ << " ";
				ofs << "0 0 0 ";
				ofs << "255 255 255 ";
				ofs << "255" << endl;

				float data[3] = { csp[i].X, csp[i].Y, csp[i].Z };
				Mat point(3, 1, CV_32FC1, data);
				Mat out = m_r.inv() * (point - m_t);
				ofs2 << out.at<float>(0, 0) << " " << out.at<float>(1, 0) << " " << out.at<float>(2, 0) << " ";
				ofs2 << "0 0 0 ";
				ofs2 << "255 255 255 ";
				ofs2 << "255" << endl;
			}
		}
		ofs.close();
		ofs2.close();
		cout << "depthtocloud ok!" << endl;
	}
	else {
		cout << "DepthToCloud: open file error!" << endl;
		return;
	}
}

//void KinectFusion::FloatDepthToCloud(float* depthData, string fileName, string fileName2) {
//
//	int count = 0;
//	for (int i = 0; i < m_cDepthHeight * m_cDepthWidth; i++)
//	{
//		float p = csp[i];
//		if (p.X != -std::numeric_limits<float>::infinity() && p.Y != -std::numeric_limits<float>::infinity() && p.Z != -std::numeric_limits<float>::infinity())
//		{
//			count++;
//		}
//	}
//	ofstream ofs;
//	ofs.open(fileName);
//	ofstream ofs2;
//	ofs2.open(fileName2);
//	if (!ofs2.is_open()) {
//		cout << "ofs2 open file error!" << endl;
//	}
//	if (ofs.is_open() && ofs2.is_open()) {
//		ofs << "ply\n";
//		ofs << "format ascii 1.0\n";
//		ofs << "element vertex " << count << "\n";
//		ofs << "property float x\n";
//		ofs << "property float y\n";
//		ofs << "property float z\n";
//		ofs << "property float nx\n";
//		ofs << "property float ny\n";
//		ofs << "property float nz\n";
//		ofs << "property uchar diffuse_red\n";
//		ofs << "property uchar diffuse_green\n";
//		ofs << "property uchar diffuse_blue\n";
//		ofs << "property uchar alpha\n";
//		ofs << "end_header\n";
//
//		ofs2 << "ply\n";
//		ofs2 << "format ascii 1.0\n";
//		ofs2 << "element vertex " << count << "\n";
//		ofs2 << "property float x\n";
//		ofs2 << "property float y\n";
//		ofs2 << "property float z\n";
//		ofs2 << "property float nx\n";
//		ofs2 << "property float ny\n";
//		ofs2 << "property float nz\n";
//		ofs2 << "property uchar diffuse_red\n";
//		ofs2 << "property uchar diffuse_green\n";
//		ofs2 << "property uchar diffuse_blue\n";
//		ofs2 << "property uchar alpha\n";
//		ofs2 << "end_header\n";
//
//		for (int i = 0; i < m_cDepthHeight * m_cDepthWidth; i++)
//		{
//
//			CameraSpacePoint p = csp[i];
//			if (p.X != -std::numeric_limits<float>::infinity() && p.Y != -std::numeric_limits<float>::infinity() && p.Z != -std::numeric_limits<float>::infinity())
//			{
//				float cameraX = static_cast<float>(p.X);
//				float cameraY = static_cast<float>(p.Y);
//				float cameraZ = static_cast<float>(p.Z);
//				ofs << cameraX << " " << cameraY << " " << cameraZ << " ";
//				ofs << "0 0 0 ";
//				ofs << "255 255 255 ";
//				ofs << "255" << endl;
//
//				float data[3] = { csp[i].X, csp[i].Y, csp[i].Z };
//				Mat point(3, 1, CV_32FC1, data);
//				Mat out = m_r.inv() * (point - m_t);
//				ofs2 << out.at<float>(0, 0) << " " << out.at<float>(1, 0) << " " << out.at<float>(2, 0) << " ";
//				ofs2 << "0 0 0 ";
//				ofs2 << "255 255 255 ";
//				ofs2 << "255" << endl;
//			}
//		}
//		ofs.close();
//		ofs2.close();
//		cout << "depthtocloud ok!" << endl;
//	}
//	else {
//		cout << "DepthToCloud: open file error!" << endl;
//		return;
//	}
//}
void KinectFusion::ProcessFusion() {
	if (m_bInitializeError)
		return;
	HRESULT hr = S_OK;
	if (nullptr == m_pVolume)
	{
		cout << "Kinect Fusion reconstruction volume not initialized. Please try reducing volume size or restarting." << endl;
		return;
	}

	//CameraSpacePoint* csp = new CameraSpacePoint[m_cDepthHeight * m_cDepthWidth];
	//hr = m_pMapper->MapDepthFrameToCameraSpace(m_cDepthHeight * m_cDepthWidth, m_pDepthImagePixelBuffer, m_cDepthHeight * m_cDepthWidth, csp);
	//if (SUCCEEDED(hr)) {
	//	int count = 0;
	//	for (int i = 0; i < m_cDepthHeight * m_cDepthWidth; i++)
	//	{
	//		CameraSpacePoint p = csp[i];
	//		if (p.X != -std::numeric_limits<float>::infinity() && p.Y != -std::numeric_limits<float>::infinity() && p.Z != -std::numeric_limits<float>::infinity())
	//		{
	//			float data[3] = { csp[i].X, csp[i].Y, csp[i].Z };
	//			Mat point(3, 1, CV_32FC1, data);
	//			Mat out = m_r.inv() * (point - m_t);
	//			if (out.at<float>(0, 0) <= bb.x_min || out.at<float>(0, 0) >= bb.x_max || out.at<float>(1, 0) <= bb.y_min || out.at<float>(1, 0) >= bb.y_max || out.at<float>(2, 0) <= bb.z_min || out.at<float>(2, 0) >= bb.z_max) {
	//				m_pDepthImagePixelBuffer[i] = 0;
	//			}
	//		}
	//	}
	//}
	//else {
	//	cout << "ProcessFusion: map depth to camera error!" << endl;
	//}

	//DepthToCloud(m_pDepthImagePixelBuffer, "depthCloud.ply", "worldcheck.ply");
	//由原深度数据构造浮点深度图数据
	hr = m_pVolume->DepthToDepthFloatFrame(m_pDepthImagePixelBuffer, m_cDepthHeight * m_cDepthWidth * sizeof(UINT16), m_pDepthFloatImage, m_fMinDepthThreshold, m_fMaxDepthThreshold, m_bMirrorDepthFrame);
	//hr = m_pVolume->DepthToDepthFloatFrame(depthData, m_cDepthImagePixels * sizeof(UINT16), m_pDepthFloatImage, m_fMinDepthThreshold, m_fMaxDepthThreshold, true);
	if (FAILED(hr))
	{
		cout << "Kinect Fusion NuiFusionDepthToDepthFloatFrame call failed." << endl;
		return;
	}
	//FloatDepthToCloud(reinterpret_cast<float*>(m_pDepthFloatImage->pFrameBuffer->pBits), "m_pDepthFloatImage.ply", "worldcheckloat.ply");
	 //平滑深度数据
	hr = m_pVolume->SmoothDepthFloatFrame(m_pDepthFloatImage, m_pSmoothDepthFloatImage, 1, 0.03f);
	if (FAILED(hr)){
		cout << "Kinect Fusion SmoothDepthFloatFrame call failed." << endl;
		return;
	}

	//处理当前帧， 进行 camera tracking 和 update the Kinect Fusion Volume
	FLOAT alignmentEnergy = 1.0f;
	if (SUCCEEDED(hr)) 
		hr = m_pVolume->ProcessFrame(m_pSmoothDepthFloatImage, m_pResampledColorImageDepthAligned, NUI_FUSION_DEFAULT_ALIGN_ITERATION_COUNT, m_cMaxIntegrationWeight, NUI_FUSION_DEFAULT_COLOR_INTEGRATION_OF_ALL_ANGLES, &alignmentEnergy, &m_worldToCameraTransform);
	
	// 检查 camera tracking 是否失败. 
	if (FAILED(hr))
	{
		if (hr == E_NUI_FUSION_TRACKING_ERROR)
		{
			m_cLostFrameCounter++;
			m_bTrackingFailed = true;
			cout << "Kinect Fusion camera tracking failed! Align the camera to the last tracked position. " << endl;
		}
		else
		{
			cout << "Kinect Fusion ProcessFrame call failed!" << endl;	
			return;
		}		
	}
	else
	{
		Matrix4 calculatedCameraPose;
		hr = m_pVolume->GetCurrentWorldToCameraTransform(&calculatedCameraPose);
		if (SUCCEEDED(hr))
		{
			if (m_bTrackingFailed)
				cout << "lostFrameCounter:" << m_cLostFrameCounter << endl;
			// Set the pose
			m_worldToCameraTransform = calculatedCameraPose;
			m_cLostFrameCounter = 0;
			m_bTrackingFailed = false;
			camerasMatrix.push_back(calculatedCameraPose);

		}
	}

	// 计算光线投影点云帧
	// Raycast all the time, even if we camera tracking failed, to enable us to visualize what is happening with the system
	hr = m_pVolume->CalculatePointCloud(m_pPointCloud, m_pCapturedSurfaceColor, &m_worldToCameraTransform);
	if (FAILED(hr))
	{
		cout << "Kinect Fusion CalculatePointCloud call failed." << endl;
		return;
	}

	// 点云渲染
	// Map X axis to blue channel, Y axis to green channel and Z axis to red channel,
	// normalizing each to the range [0, 1].
	Matrix4 worldToBGRTransform = { 0.0f };
	worldToBGRTransform.M11 = m_reconstructionParams.voxelsPerMeter / m_reconstructionParams.voxelCountX;
	worldToBGRTransform.M22 = m_reconstructionParams.voxelsPerMeter / m_reconstructionParams.voxelCountY;
	worldToBGRTransform.M33 = m_reconstructionParams.voxelsPerMeter / m_reconstructionParams.voxelCountZ;
	worldToBGRTransform.M41 = 0.5f;
	worldToBGRTransform.M42 = 0.5f;
	worldToBGRTransform.M43 = 0.0f;
	worldToBGRTransform.M44 = 1.0f;
	hr = NuiFusionShadePointCloud(m_pPointCloud, &m_worldToCameraTransform, &worldToBGRTransform, m_pShadedSurface, nullptr);
	if (FAILED(hr))
	{
		cout << "Kinect Fusion NuiFusionShadePointCloud call failed." << endl;
		return;
	}
	m_cFrameCounter++;
}
void KinectFusion::GetResult(const string fileName) {
	Mat w2vR(3, 3, CV_32FC1);
	Mat w2vT(3, 1, CV_32FC1);
	w2vR.at<float>(0, 0) = m_defaultWorldToVolumeTransform.M11; w2vR.at<float>(0, 1) = m_defaultWorldToVolumeTransform.M21; w2vR.at<float>(0, 2) = m_defaultWorldToVolumeTransform.M31;
	w2vR.at<float>(1, 0) = m_defaultWorldToVolumeTransform.M12; w2vR.at<float>(1, 1) = m_defaultWorldToVolumeTransform.M22; w2vR.at<float>(1, 2) = m_defaultWorldToVolumeTransform.M32;
	w2vR.at<float>(2, 0) = m_defaultWorldToVolumeTransform.M13; w2vR.at<float>(2, 1) = m_defaultWorldToVolumeTransform.M23; w2vR.at<float>(2, 2) = m_defaultWorldToVolumeTransform.M33;	
	w2vT.at<float>(0, 0) = m_defaultWorldToVolumeTransform.M41; w2vT.at<float>(1, 0) = m_defaultWorldToVolumeTransform.M42; w2vT.at<float>(2, 0) = m_defaultWorldToVolumeTransform.M43;

	//mesh
	INuiFusionColorMesh* colorMesh;
	HRESULT hr = m_pVolume->CalculateMesh(1, &colorMesh);
	if (FAILED(hr)) {
		cout << "CalculateMesh failed!" << endl;
		return;
	}
	unsigned int count = colorMesh->VertexCount();
	unsigned int nCount = colorMesh->NormalCount();
	unsigned int tCount = colorMesh->TriangleVertexIndexCount()/3;
	cout << "vertex :" << count << endl;
	cout << "normal:" << nCount << endl;
	cout << "triangles:" << tCount << endl;
	const Vector3 *pVertices;// = new Vector3[count];
	hr = colorMesh->GetVertices(&pVertices);
	if (FAILED(hr)) {
		cout << "get Vertices failed!" << endl;
		return;
	}
	const Vector3* pNormals;
	hr = colorMesh->GetNormals(&pNormals);
	if (FAILED(hr)) {
		cout << "get Normals failed!" << endl;
		return;
	}
	const int* colors;
	hr = colorMesh->GetColors(&colors);
	if (FAILED(hr)) {
		cout << "get Colors failed!" << endl;
		return;
	}	
	ofstream ofs;
	ofs.open(fileName);
	if (ofs.is_open()) {
		ofs << "ply\n";
		ofs << "format ascii 1.0\n";
		int showCameCount = (camerasMatrix.size() % 10 == 0 ? camerasMatrix.size() / 10 : (camerasMatrix.size() / 10 + 1));
		ofs << "element vertex " << count + 8 + showCameCount * 4 << "\n";
		ofs << "property float x\n";
		ofs << "property float y\n";
		ofs << "property float z\n";
		ofs << "property float nx\n";
		ofs << "property float ny\n";
		ofs << "property float nz\n";
		ofs << "property uchar diffuse_red\n";
		ofs << "property uchar diffuse_green\n";
		ofs << "property uchar diffuse_blue\n";
		ofs << "property uchar alpha\n";
		ofs << "element face " << tCount << "\n";
		ofs << "property list uchar int vertex_index\n";
		ofs << "end_header\n";
		
		for (int i = 0; i < camerasMatrix.size(); i = i + 10) {
			Mat c_to_w_R, r1(3, 3, CV_32FC1);
			Mat c_to_w_T, t1(3, 1, CV_32FC1);
			r1.at<float>(0, 0) = camerasMatrix[i].M11; r1.at<float>(0, 1) = camerasMatrix[i].M21; r1.at<float>(0, 2) = camerasMatrix[i].M31;
			r1.at<float>(1, 0) = camerasMatrix[i].M12; r1.at<float>(1, 1) = camerasMatrix[i].M22; r1.at<float>(1, 2) = camerasMatrix[i].M32;
			r1.at<float>(2, 0) = camerasMatrix[i].M13; r1.at<float>(2, 1) = camerasMatrix[i].M23; r1.at<float>(2, 2) = camerasMatrix[i].M33;
			t1.at<float>(0, 0) = camerasMatrix[i].M41; t1.at<float>(1, 0) = camerasMatrix[i].M42; t1.at<float>(2, 0) = camerasMatrix[i].M43;
			c_to_w_R = r1.inv();
			c_to_w_T = -(c_to_w_R * t1);
			ofs << c_to_w_T.at<float>(0, 0) << " " << c_to_w_T.at<float>(1, 0) << " " << c_to_w_T.at<float>(2, 0) << " ";
			ofs << "0 0 0 0 0 0 255\n";

			Mat xx = Mat::zeros(3, 1, CV_32FC1);
			xx.at<float>(0, 0) = 0.05f;
			xx = c_to_w_R * xx + c_to_w_T;
			ofs << xx.at<float>(0, 0) << " " << xx.at<float>(1, 0) << " " << xx.at<float>(2, 0) << " ";
			ofs << "0 0 0 255 0 0 255\n";

			xx = Mat::zeros(3, 1, CV_32FC1);
			xx.at<float>(1, 0) = 0.05f;
			xx = c_to_w_R * xx + c_to_w_T;
			ofs << xx.at<float>(0, 0) << " " << xx.at<float>(1, 0) << " " << xx.at<float>(2, 0) << " ";
			ofs << "0 0 0 0 255 0 255\n";

			xx = Mat::zeros(3, 1, CV_32FC1);
			xx.at<float>(2, 0) = 0.05f;
			xx = c_to_w_R * xx + c_to_w_T;
			ofs << xx.at<float>(0, 0) << " " << xx.at<float>(1, 0) << " " << xx.at<float>(2, 0) << " ";
			ofs << "0 0 0 0 0 255 255\n";			
		}

		Mat d1(3, 1, CV_32FC1);
		d1.at<float>(0, 0) = 0.0f; d1.at<float>(1, 0) = 0.0f; d1.at<float>(2, 0) = 0.0f;
		d1 = w2vR.inv() * (d1 - w2vT);
		ofs << d1.at<float>(0, 0) << " " << d1.at<float>(1, 0) << " " << d1.at<float>(2, 0) << " ";
		ofs << "0 0 0 255 0 255 255" << endl;

		Mat d2(3, 1, CV_32FC1);
		d2.at<float>(0, 0) = 0.0f; d2.at<float>(1, 0) = 0.0f; d2.at<float>(2, 0) = 256.0f;
		d2 = w2vR.inv() * (d2 - w2vT);
		ofs << d2.at<float>(0, 0) << " " << d2.at<float>(1, 0) << " " << d2.at<float>(2, 0) << " ";
		ofs << "0 0 0 255 0 255 255" << endl;

		Mat d3(3, 1, CV_32FC1);
		d3.at<float>(0, 0) = 0.0f; d3.at<float>(1, 0) = 256.0f; d3.at<float>(2, 0) = 0.0f;
		d3 = w2vR.inv() * (d3 - w2vT);
		ofs << d3.at<float>(0, 0) << " " << d3.at<float>(1, 0) << " " << d3.at<float>(2, 0) << " ";
		ofs << "0 0 0 255 0 255 255" << endl;

		Mat d4(3, 1, CV_32FC1);
		d4.at<float>(0, 0) = 0.0f; d4.at<float>(1, 0) = 256.0f; d4.at<float>(2, 0) = 256.0f;
		d4 = w2vR.inv() * (d4 - w2vT);
		ofs << d4.at<float>(0, 0) << " " << d4.at<float>(1, 0) << " " << d4.at<float>(2, 0) << " ";
		ofs << "0 0 0 255 0 255 255" << endl;

		Mat d5(3, 1, CV_32FC1);
		d5.at<float>(0, 0) = 256.0f; d5.at<float>(1, 0) = 0.0f; d5.at<float>(2, 0) = 0.0f;
		d5 = w2vR.inv() * (d5 - w2vT);
		ofs << d5.at<float>(0, 0) << " " << d5.at<float>(1, 0) << " " << d5.at<float>(2, 0) << " ";
		ofs << "0 0 0 255 0 255 255" << endl;

		Mat d6(3, 1, CV_32FC1);
		d6.at<float>(0, 0) = 256.0f; d6.at<float>(1, 0) = 0.0f; d6.at<float>(2, 0) = 256.0f;
		d6 = w2vR.inv() * (d6 - w2vT);
		ofs << d6.at<float>(0, 0) << " " << d6.at<float>(1, 0) << " " << d6.at<float>(2, 0) << " ";
		ofs << "0 0 0 255 0 255 255" << endl;

		Mat d7(3, 1, CV_32FC1);
		d7.at<float>(0, 0) = 256.0f; d7.at<float>(1, 0) = 256.0f; d7.at<float>(2, 0) = 0.0f;
		d7 = w2vR.inv() * (d7 - w2vT);
		ofs << d7.at<float>(0, 0) << " " << d7.at<float>(1, 0) << " " << d7.at<float>(2, 0) << " ";
		ofs << "0 0 0 255 0 255 255" << endl;

		Mat d8(3, 1, CV_32FC1);
		d8.at<float>(0, 0) = 256.0f; d8.at<float>(1, 0) = 256.0f; d8.at<float>(2, 0) = 256.0f;
		d8 = w2vR.inv() * (d8 - w2vT);
		ofs << d8.at<float>(0, 0) << " " << d8.at<float>(1, 0) << " " << d8.at<float>(2, 0) << " ";
		ofs << "0 0 0 255 0 255 255" << endl;

		for (unsigned i = 0; i < count; i++)
		{
			//float x = pVertices[i].x * m_worldToCameraTransform.M11 + pVertices[i].y * m_worldToCameraTransform.M21 + pVertices[i].z * m_worldToCameraTransform.M31 + m_worldToCameraTransform.M41;
			//float y = pVertices[i].x * m_worldToCameraTransform.M12 + pVertices[i].y * m_worldToCameraTransform.M22 + pVertices[i].z * m_worldToCameraTransform.M32 + m_worldToCameraTransform.M42;
			//float z = pVertices[i].x * m_worldToCameraTransform.M13 + pVertices[i].y * m_worldToCameraTransform.M23 + pVertices[i].z * m_worldToCameraTransform.M33 + m_worldToCameraTransform.M43;
			//ofs << x << ' ' << y << ' ' << z << ' ';
			ofs << pVertices[i].x << ' ' << pVertices[i].y << ' ' << pVertices[i].z << ' ';
			ofs << pNormals[i].x << ' ' << pNormals[i].y << ' ' << pNormals[i].z << ' ';
			ofs << ((colors[i] >> 16) & 255) << ' ' << ((colors[i] >> 8) & 255) << ' ' << ((colors[i] >> 16) & 255) << ' ' << static_cast<int>(255) << endl;
		}
		for (unsigned i = 0 + showCameCount * 4 + 8; i < nCount + showCameCount * 4 + 8; i = i + 3) {
			ofs << "3 " << i << " " << i + 1 << " " << i + 2 << endl;
		}
		ofs.close();
	}
	else {
		cout << "GetResult: open file error!" << endl;
		return;
	}
}

void WriteImgList(const string fileName, const string pathPrefix, const int start, const int end) {
	ofstream ofs;
	ofs.open(fileName);
	if (ofs.is_open()) {
		for (int i = start; i <= end; i++) {
			stringstream ss;
			ss << i << endl;
			string out;
			ss >> out;
			while (out.length() < 4) {
				out = "0" + out;
			}
			ofs << pathPrefix + out + ".png" << endl;
		}
		ofs.close();
	}
	else {
		cout << "writeImgList: open file error!" << endl;
		exit(0);
	}
}

void GetImgsPath(string listDir, vector<string>& imgsPath) {
	//read it to the storage	
	ifstream readPic(listDir);
	if (!readPic)
	{
		cout << "error: Cannot open the dir!" << endl;
		exit(0);
	}
	string tmpStr;
	while (getline(readPic, tmpStr))
	{
		imgsPath.push_back(tmpStr);
	}
	readPic.close();
}

void GetCamerasPos() {
	ofstream ofs(".\\camePos.out");
	ofs << "# Bundle file v0.3" << endl;
	ofs << camerasMatrix.size() << ' ' << static_cast<int>(0) << endl;
	for (int i = 0; i < camerasMatrix.size(); i++) {
		float came[5][3];
		came[0][0] = 364.630;
		came[0][1] = 0;
		came[0][2] = 0;
		came[1][0] = camerasMatrix[i].M11;
		came[1][1] = camerasMatrix[i].M21;
		came[1][2] = camerasMatrix[i].M31;
		came[2][0] = camerasMatrix[i].M12;
		came[2][1] = camerasMatrix[i].M22;
		came[2][2] = camerasMatrix[i].M32;
		came[3][0] = camerasMatrix[i].M13;
		came[3][1] = camerasMatrix[i].M23;
		came[3][2] = camerasMatrix[i].M33;
		came[4][0] = camerasMatrix[i].M14;
		came[4][1] = camerasMatrix[i].M24;
		came[4][2] = camerasMatrix[i].M34;

		ofs << came[0][0] << ' ' << came[0][1] << ' ' << came[0][2] << endl;
		ofs << came[1][0] << ' ' << came[1][1] << ' ' << came[1][2] << endl;
		ofs << came[2][0] << ' ' << came[2][1] << ' ' << came[2][2] << endl;
		ofs << came[3][0] << ' ' << came[3][1] << ' ' << came[3][2] << endl;
		ofs << came[4][0] << ' ' << came[4][1] << ' ' << came[4][2] << endl;
	}
	ofs.close();
}

void ReadRAndT(Mat&R, Mat&T, const string fileName){
	R.create(3, 3, CV_32FC1);
	T.create(3, 1, CV_32FC1);
	ifstream ifs;
	ifs.open(fileName);
	if (ifs.is_open()) {
		for (int i = 0; i < 3; i++) {
			float a, b, c;
			ifs >> a >> b >> c;
			//cout << a << ",  " << b << ",  " << c << endl;
			R.at<float>(i, 0) = a; R.at<float>(i, 1) = b; R.at<float>(i, 2) = c;
		}
		float a, b, c;
		ifs >> a >> b >> c;
		//cout << a << ",  " << b << ",  " << c << endl;
		T.at<float>(0, 0) = a; T.at<float>(1, 0) = b; T.at<float>(2, 0) = c;
		ifs.close();
	}
	else {
		cout << "ReadRAndT: open file error!" << endl;
		exit(0);
	}
}

int  main() {
	//read r and t
	Mat r, t;
	ReadRAndT(r, t, ".\\RAndT.txt");
	//prepare depth and color image data;
	WriteImgList(".\\inDepth.txt", "../../data/tong/depth_", 1, 409);
	WriteImgList(".\\inColor.txt", "../../data/tong/color_", 1, 409);
	vector<string> depthImgsPath, colorImgsPath;
	GetImgsPath(".\\inDepth.txt", depthImgsPath);
	GetImgsPath(".\\inColor.txt", colorImgsPath);

	KinectFusion kf(r, t);
	HRESULT hr = kf.CreateFirstConnected();
	if (FAILED(hr)) {
		cout << "CreateFirstConnected error !" << endl;
		return 0;
	}
	hr = kf.InitializeKinectFusion();
	if (FAILED(hr)) {
		cout << "InitializeKinectFusion error!" << endl;
		return 0;
	}
	while (!kf.CoordinateChangeCheck());
	kf.CloseSensor();
	for (int i = 0; i < depthImgsPath.size(); i++) {
		cout << "i:" << i << endl;
		Mat tdepth = imread(depthImgsPath[i], CV_LOAD_IMAGE_UNCHANGED);
		Mat tcolor = imread(colorImgsPath[i], CV_LOAD_IMAGE_UNCHANGED);
		CV_Assert(!tdepth.empty());
		CV_Assert(!tcolor.empty());
		kf.IncreaseFusion(tdepth, tcolor);
		if (kf.m_cLostFrameCounter >= 10)
			break;
	}
	kf.GetResult(".\\result.ply");
	//getCamerasPos();
	cout << "帧数：" << kf.m_cFrameCounter << endl;
	return 0;
}


