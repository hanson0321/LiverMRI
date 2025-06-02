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
    """將移動影像對位到固定影像空間。"""
    print("  開始進行影像對位...")

    registration_method = sitk.ImageRegistrationMethod()

    # --- 我們在這裡加入除錯標記 ---
    print("\n---- DEBUG: 正在設定度量方法... ----") # <--- 新增的除錯訊息

    # registration_method.SetMetricAsCorrelation() # 確保這一行已經被註解或刪除
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50) # 這是我們想要的方法

    print("---- DEBUG: 已成功設定為 MattesMutualInformation ----\n") # <--- 新增的除錯訊息
    # -----------------------------
    registration_method.SetOptimizerAsGradientDescent(
        learningRate=1.0,
        numberOfIterations=200,
        convergenceMinimumValue=1e-6,
        convergenceWindowSize=10
    )
    registration_method.SetOptimizerScalesFromPhysicalShift()

    initial_transform = sitk.CenteredTransformInitializer(
    fixed_image_sitk,
    moving_image_sitk,
    sitk.AffineTransform(fixed_image_sitk.GetDimension()), # <--- 修改為 AffineTransform
    sitk.CenteredTransformInitializerFilter.GEOMETRY
)
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    registration_method.SetInterpolator(sitk.sitkLinear)

    try:
        final_transform = registration_method.Execute(
            sitk.Cast(fixed_image_sitk, sitk.sitkFloat32),
            sitk.Cast(moving_image_sitk, sitk.sitkFloat32)
        )
        print(f"  對位完成。最終度量值: {registration_method.GetMetricValue():.4f}")

        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(fixed_image_sitk)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(float(np.min(sitk.GetArrayViewFromImage(moving_image_sitk))))
        resampler.SetTransform(final_transform)

        moving_image_resampled_sitk = resampler.Execute(moving_image_sitk)
        moving_image_resampled_array = sitk.GetArrayFromImage(moving_image_resampled_sitk)
        
        return moving_image_resampled_array

    except Exception as e:
        print(f"  影像對位執行時發生錯誤: {e}")
        return None


def process_and_save_overlay_series(
    pdff_volume_np, t1c_resampled_volume_np, 
    output_folder, frangi_params, original_pdff_dicom_paths
):
    """
    對整個影像體積進行 Frangi 濾波、疊加，並將結果儲存為 DICOM 序列。
    """
    print(f"  開始處理與儲存疊加影像至: {output_folder}")
    os.makedirs(output_folder, exist_ok=True)
    
    # 為這個新的 DICOM 系列產生唯一的 ID
    new_series_instance_uid = generate_uid()
    
    num_slices = min(pdff_volume_np.shape[0], t1c_resampled_volume_np.shape[0])

    for i in range(num_slices):
        # 1. 取得單張切片
        pdff_slice = pdff_volume_np[i, :, :]
        t1c_slice = t1c_resampled_volume_np[i, :, :]

        # 2. 血管分割 (Frangi Filter)
        t1c_slice_normalized = exposure.rescale_intensity(
            t1c_slice.astype(np.float64), in_range='image', out_range=(0, 1)
        )
        
        sigmas = range(frangi_params['min_sigma'], frangi_params['max_sigma'] + 1, 1)
        vesselness = frangi(t1c_slice_normalized, sigmas=sigmas, black_ridges=False)

        # 3. 閾值處理
        if np.any(vesselness > 0):
            thresh = threshold_otsu(vesselness[vesselness > 0])
        else:
            thresh = 0.5
        vessel_mask = (vesselness > thresh)

        # 4. 建立疊加影像 (RGB)
        pdff_uint8 = exposure.rescale_intensity(pdff_slice, out_range=(0, 255)).astype(np.uint8)
        overlay_rgb = cv2.cvtColor(pdff_uint8, cv2.COLOR_GRAY2BGR)
        overlay_rgb[vessel_mask] = [0, 0, 255] # BGR for Red

        # 5. 儲存為 DICOM
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