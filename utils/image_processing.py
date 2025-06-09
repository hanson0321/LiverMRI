import SimpleITK as sitk
import numpy as np
import cv2
from skimage.filters import frangi, threshold_otsu
from skimage import exposure
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import generate_uid
import os
import datetime

def register_images(fixed_image_sitk, moving_image_sitk):
    """
    將移動影像對位到固定影像空間。
    採用多階段對位策略：
    1. 仿射對位 (Affine Registration) 作為粗對位。
    2. B-Spline 對位 (B-Spline Registration) 作為精細的非剛性對位。
    """
    print("--- 開始執行多階段影像對位 ---")

    # --- 階段一：仿射對位 (Affine Registration) ---
    print("  [階段 1/2] 正在進行仿射對位...")
    
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image_sitk,
        moving_image_sitk,
        sitk.AffineTransform(fixed_image_sitk.GetDimension()),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    registration_method_affine = sitk.ImageRegistrationMethod()
    registration_method_affine.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method_affine.SetMetricSamplingStrategy(registration_method_affine.RANDOM)
    registration_method_affine.SetMetricSamplingPercentage(0.01)
    registration_method_affine.SetInterpolator(sitk.sitkLinear)
    
    registration_method_affine.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=500,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10
    )
    registration_method_affine.SetOptimizerScalesFromPhysicalShift()
    registration_method_affine.SetInitialTransform(initial_transform, inPlace=False)

    try:
        affine_transform = registration_method_affine.Execute(
            sitk.Cast(fixed_image_sitk, sitk.sitkFloat32),
            sitk.Cast(moving_image_sitk, sitk.sitkFloat32)
        )
        print(f"  仿射對位完成。最終度量值: {registration_method_affine.GetMetricValue():.4f}")
    except Exception as e:
        print(f"  仿射對位執行時發生錯誤: {e}")
        return None

    # --- 階段二：B-Spline 非剛性對位 ---
    print("\n  [階段 2/2] 正在進行 B-Spline 非剛性對位...")

    transform_domain_mesh_size = [8] * fixed_image_sitk.GetDimension()
    bspline_transform = sitk.BSplineTransformInitializer(
        image1=fixed_image_sitk,
        transformDomainMeshSize=transform_domain_mesh_size,
        order=3
    )
    
    registration_method_bspline = sitk.ImageRegistrationMethod()
    registration_method_bspline.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method_bspline.SetMetricSamplingStrategy(registration_method_bspline.RANDOM)
    registration_method_bspline.SetMetricSamplingPercentage(0.01)
    registration_method_bspline.SetInterpolator(sitk.sitkLinear)

    registration_method_bspline.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-5,
        numberOfIterations=100,
        maximumNumberOfCorrections=5,
        maximumNumberOfFunctionEvaluations=1024,
        costFunctionConvergenceFactor=1e+7
    )

    composite_transform = sitk.CompositeTransform(affine_transform)
    composite_transform.AddTransform(bspline_transform)
    registration_method_bspline.SetInitialTransform(composite_transform)

    try:
        final_transform = registration_method_bspline.Execute(
            sitk.Cast(fixed_image_sitk, sitk.sitkFloat32),
            sitk.Cast(moving_image_sitk, sitk.sitkFloat32)
        )
        print(f"  B-Spline 對位完成。最終度量值: {registration_method_bspline.GetMetricValue():.4f}")
    except Exception as e:
        print(f"  B-Spline 對位執行時發生錯誤: {e}")
        return None

    # --- 使用最終的轉換來重採樣影像 ---
    print("\n--- 對位完成，正在重採樣影像 ---")
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image_sitk)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(float(np.min(sitk.GetArrayViewFromImage(moving_image_sitk))))
    resampler.SetTransform(final_transform)

    moving_image_resampled_sitk = resampler.Execute(moving_image_sitk)
    moving_image_resampled_array = sitk.GetArrayFromImage(moving_image_resampled_sitk)
    
    return moving_image_resampled_array, final_transform
    """
    將移動影像對位到固定影像空間。
    採用多階段對位策略：
    1. 仿射對位 (Affine Registration) 作為粗對位。
    2. B-Spline 對位 (B-Spline Registration) 作為精細的非剛性對位。
    """
    print("--- 開始執行多階段影像對位 ---")

    # --- 階段一：仿射對位 (Affine Registration) ---
    print("  [階段 1/2] 正在進行仿射對位...")
    
    # 初始化轉換
    initial_transform = sitk.CenteredTransformInitializer(
        fixed_image_sitk,
        moving_image_sitk,
        sitk.AffineTransform(fixed_image_sitk.GetDimension()),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )

    # 設定對位方法
    registration_method_affine = sitk.ImageRegistrationMethod()
    registration_method_affine.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method_affine.SetMetricSamplingStrategy(registration_method_affine.RANDOM)
    registration_method_affine.SetMetricSamplingPercentage(0.01)
    registration_method_affine.SetInterpolator(sitk.sitkLinear)
    
    # 設定優化器
    registration_method_affine.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=300,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10
    )
    registration_method_affine.SetOptimizerScalesFromPhysicalShift()
    registration_method_affine.SetInitialTransform(initial_transform, inPlace=False)

    try:
        affine_transform = registration_method_affine.Execute(
            sitk.Cast(fixed_image_sitk, sitk.sitkFloat32),
            sitk.Cast(moving_image_sitk, sitk.sitkFloat32)
        )
        print(f"  仿射對位完成。最終度量值: {registration_method_affine.GetMetricValue():.4f}")
    except Exception as e:
        print(f"  仿射對位執行時發生錯誤: {e}")
        return None

    # --- 階段二：B-Spline 非剛性對位 ---
    print("\n  [階段 2/2] 正在進行 B-Spline 非剛性對位...")

    # B-Spline 轉換的網格大小，數值越小，自由度越高
    transform_domain_mesh_size = [8] * fixed_image_sitk.GetDimension()
    bspline_transform = sitk.BSplineTransformInitializer(
        image1=fixed_image_sitk,
        transformDomainMeshSize=transform_domain_mesh_size,
        order=3
    )
    
    # 設定對位方法
    registration_method_bspline = sitk.ImageRegistrationMethod()
    registration_method_bspline.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method_bspline.SetMetricSamplingStrategy(registration_method_bspline.RANDOM)
    registration_method_bspline.SetMetricSamplingPercentage(0.01)
    registration_method_bspline.SetInterpolator(sitk.sitkLinear)

    # 更換優化器為 LBFGSB，這是 B-Spline 的常用選擇
    registration_method_bspline.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-5,
        numberOfIterations=100,
        maximumNumberOfCorrections=5,
        maximumNumberOfFunctionEvaluations=1024,
        costFunctionConvergenceFactor=1e+7
    )

    # 以第一階段的仿射對位結果作為 B-Spline 對位的初始轉換
    # 這裡我們需要一個 CompositeTransform
    composite_transform = sitk.CompositeTransform(affine_transform)
    composite_transform.AddTransform(bspline_transform)
    registration_method_bspline.SetInitialTransform(composite_transform)

    try:
        final_transform = registration_method_bspline.Execute(
            sitk.Cast(fixed_image_sitk, sitk.sitkFloat32),
            sitk.Cast(moving_image_sitk, sitk.sitkFloat32)
        )
        print(f"  B-Spline 對位完成。最終度量值: {registration_method_bspline.GetMetricValue():.4f}")
    except Exception as e:
        print(f"  B-Spline 對位執行時發生錯誤: {e}")
        return None

    # --- 使用最終的轉換來重採樣影像 ---
    print("\n--- 對位完成，正在重採樣影像 ---")
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(fixed_image_sitk)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(float(np.min(sitk.GetArrayViewFromImage(moving_image_sitk))))
    resampler.SetTransform(final_transform)

    moving_image_resampled_sitk = resampler.Execute(moving_image_sitk)
    moving_image_resampled_array = sitk.GetArrayFromImage(moving_image_resampled_sitk)
    
    return moving_image_resampled_array


# 在 utils/image_processing.py 中
def process_and_save_overlay_series(
    pdff_volume_np, t1c_resampled_volume_np, 
    output_folder, frangi_params, original_pdff_dicom_paths
):
    """
    對整個影像體積進行 Frangi 濾波、疊加，並將結果儲存為 DICOM 序列。
    (已更新為最終簡化版邏輯)
    """
    print(f"  開始處理與儲存疊加影像至: {output_folder}")
    os.makedirs(output_folder, exist_ok=True)
    
    new_series_instance_uid = generate_uid()
    num_slices = min(pdff_volume_np.shape[0], t1c_resampled_volume_np.shape[0])

    # 從參數字典中獲取 vessel_scale
    vessel_scale = frangi_params.get('vessel_scale', 1)

    for i in range(num_slices):
        pdff_slice = pdff_volume_np[i, :, :]
        t1c_slice = t1c_resampled_volume_np[i, :, :]

        # --- 更新後的 Frangi 血管偵測 ---
        vesselness = frangi(t1c_slice, scale_range=(vessel_scale, vessel_scale + 4), scale_step=1, black_ridges=False)
        p2, p98 = np.percentile(vesselness, (2, 98))
        vesselness_rescaled = exposure.rescale_intensity(vesselness, in_range=(p2, p98))
        
        # --- 全自動 Otsu 閾值處理 ---
        if vesselness_rescaled.max() > 0:
            threshold_val = threshold_otsu(vesselness_rescaled[vesselness_rescaled > 0])
        else:
            threshold_val = 0.5
        vessel_mask = (vesselness_rescaled > threshold_val)

        # --- 建立疊加影像 ---
        pdff_uint8 = exposure.rescale_intensity(pdff_slice, out_range=(0, 255)).astype(np.uint8)
        overlay_rgb = cv2.cvtColor(pdff_uint8, cv2.COLOR_GRAY2BGR)
        overlay_rgb[vessel_mask] = [255, 0, 0] # BGR for Blue

        save_slice_as_dicom(
            overlay_rgb, i, output_folder, new_series_instance_uid, 
            original_pdff_dicom_paths
        )
    
    print(f"  成功儲存 {num_slices} 個 DICOM 檔案。")


def save_slice_as_dicom(
    rgb_image, slice_index, output_folder, series_uid, 
    original_dicom_paths
):
    """將單張 RGB 影像儲存為 DICOM 檔案。"""
    
    # 建立檔名
    filename = os.path.join(output_folder, f"slice_{slice_index:04d}.dcm")
    
    # 使用原始 DICOM 作為模板以保留病人資訊
    try:
        ds = pydicom.dcmread(original_dicom_paths[slice_index])
    except Exception:
        # 如果模板讀取失敗，建立一個最小化的 DICOM header
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = '1.2.840.10008.5.1.4.1.1.7' # Secondary Capture Image Storage
        file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
        ds = Dataset()
        ds.file_meta = file_meta
        ds.PatientName = "Unknown"
        ds.PatientID = "Unknown"
        ds.StudyInstanceUID = generate_uid() # 應從模板繼承，但此處作為備用
    
    # 設定影像特有的 DICOM tags
    ds.SOPInstanceUID = generate_uid()
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds.SeriesInstanceUID = series_uid
    ds.ImageType = ['DERIVED', 'SECONDARY', 'OVERLAY']
    ds.Modality = "OT" # Other
    ds.SeriesNumber = 999 # 給予一個高數字以區別於原始序列
    ds.InstanceNumber = slice_index + 1
    ds.SeriesDescription = "PDFF with Frangi Vessel Overlay"
    
    # 設定 RGB 影像屬性
    ds.SamplesPerPixel = 3
    ds.PhotometricInterpretation = "RGB"
    ds.PlanarConfiguration = 0
    ds.Rows, ds.Columns, _ = rgb_image.shape
    ds.BitsAllocated = 8
    ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    
    # 移除與灰階相關的 tag
    for tag_name in ["WindowCenter", "WindowWidth", "RescaleIntercept", "RescaleSlope"]:
        if tag_name in ds:
            delattr(ds, tag_name)
    
    ds.PixelData = rgb_image.tobytes()
    ds.save_as(filename, write_like_original=False)