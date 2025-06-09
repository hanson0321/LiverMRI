import os
import time
from utils.data_loader import load_dicom_series
from utils.image_processing import register_images, process_and_save_overlay_series

# --- 1. 設定區：所有參數都在這裡調整 ---

# 設定資料路徑
BASE_DATA_PATH = "data/All_Patients"
BASE_OUTPUT_PATH = "output"

# 設定 DICOM 資料夾的固定名稱
T1C_FOLDER_NAME = "Ax T1 FS BH+C"
PDFF_FOLDER_NAME = "FatFrac  3D Ax IDEAL IQ BH"

# --- 更新 Frangi 濾波器的參數 ---
# 請將 vessel_scale 的值設為您在互動模式下找到的最佳值
FRANGI_PARAMS = {
    'vessel_scale': 1, 
}

# --- 2. 主程式邏輯 ---

def main():
    """主執行函式，遍歷所有病人並進行處理。"""
    
    print("="*60)
    print("開始執行 MRI 肝臟影像批次處理...")
    print(f"將使用 Frangi 參數: {FRANGI_PARAMS}")
    print("="*60)
    
    if not os.path.isdir(BASE_DATA_PATH):
        print(f"錯誤：找不到病人資料夾 '{BASE_DATA_PATH}'")
        return

    patient_folders = sorted([d for d in os.listdir(BASE_DATA_PATH) if os.path.isdir(os.path.join(BASE_DATA_PATH, d))])
    
    total_start_time = time.time()

    for patient_id in patient_folders:
        print(f"\n>>>>>> 開始處理病人: {patient_id} <<<<<<")
        patient_start_time = time.time()
        
        try:
            # --- 載入資料 ---
            t1c_dir = os.path.join(BASE_DATA_PATH, patient_id, T1C_FOLDER_NAME)
            pdff_dir = os.path.join(BASE_DATA_PATH, patient_id, PDFF_FOLDER_NAME)
            
            volume_t1c_sitk, _, _ = load_dicom_series(t1c_dir, f"T1+C ({patient_id})")
            volume_pdff_sitk, array_pdff, original_pdff_paths = load_dicom_series(pdff_dir, f"PDFF ({patient_id})")

            if volume_t1c_sitk is None or volume_pdff_sitk is None:
                raise ValueError("影像載入失敗")

            # --- 影像對位 (修正函式呼叫) ---
            # register_images 現在返回兩個值，我們只需要第一個（影像陣列）
            array_t1c_resampled, _ = register_images(volume_pdff_sitk, volume_t1c_sitk)

            if array_t1c_resampled is None:
                raise ValueError("影像對位失敗")
                
            # --- 處理與儲存 ---
            patient_output_folder = os.path.join(BASE_OUTPUT_PATH, f"patient_{patient_id}_overlay")
            process_and_save_overlay_series(
                pdff_volume_np=array_pdff,
                t1c_resampled_volume_np=array_t1c_resampled,
                output_folder=patient_output_folder,
                frangi_params=FRANGI_PARAMS,
                original_pdff_dicom_paths=original_pdff_paths
            )
            
            patient_time = time.time() - patient_start_time
            print(f"--- 病人 {patient_id} 處理完成，耗時: {patient_time:.2f} 秒 ---")

        except Exception as e:
            print(f"!!!!!! 處理病人 {patient_id} 時發生錯誤: {e} !!!!!!")
            print("!!!!!! 跳過此病人，繼續處理下一個。 !!!!!!")
            continue

    total_time = time.time() - total_start_time
    print("\n="*60)
    print(f"所有病人批次處理完成！總耗時: {total_time / 60:.2f} 分鐘。")
    print(f"所有結果已儲存在 '{BASE_OUTPUT_PATH}' 資料夾中。")

if __name__ == "__main__":
    main()