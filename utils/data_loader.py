import os
import SimpleITK as sitk

def load_dicom_series(base_dir, series_description="影像"):
    """
    從指定資料夾載入 DICOM series。
    返回 SimpleITK Image 物件和 NumPy array。
    """
    print(f"開始載入 {series_description} 從: {base_dir}")
    if not os.path.isdir(base_dir):
        print(f"錯誤：路徑 '{base_dir}' 不存在或不是一個資料夾。")
        return None, None

    reader = sitk.ImageSeriesReader()
    try:
        series_IDs = reader.GetGDCMSeriesIDs(base_dir)
        if not series_IDs:
            print(f"錯誤：在路徑 '{base_dir}' 中找不到 DICOM Series。")
            return None, None

        target_series_id = series_IDs[0]
        series_files = reader.GetGDCMSeriesFileNames(base_dir, target_series_id)
        
        if not series_files:
            print(f"錯誤：無法取得 {series_description} (Series ID: '{target_series_id}') 的檔案列表。")
            return None, None

        reader.SetFileNames(series_files)
        volume_sitk = reader.Execute()
        array_np = sitk.GetArrayFromImage(volume_sitk)

        print(f"  {series_description} 載入成功。 NumPy 陣列形狀: {array_np.shape}")
        
        # 返回 SimpleITK 影像物件, NumPy 陣列, 以及原始檔案路徑列表 (用於後續儲存)
        return volume_sitk, array_np, series_files

    except Exception as e:
        print(f"讀取 {series_description} (路徑: {base_dir}) 時發生錯誤: {e}")
        return None, None, None