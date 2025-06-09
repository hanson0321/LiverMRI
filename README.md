## 專案結構
利用 T1 加權影像中較清晰的血管標記，透過 registration 對齊至 PDFF 序列，進一步輔助 liver segmentation 與脂肪含量估算，強化 PDFF segmentation 準確度與臨床應用價值。
流程大致如下：

資料載入：從 data/All_Patients/ 中讀取指定病人的 T1 Contrast-Enhanced (Ax T1 FS BH+C) 和 PDFF (FatFrac 3D Ax IDEAL IQ BH) 這兩個 DICOM 影像序列。
影像對位：以 PDFF 影像為固定參考（Fixed Image），將 T1 影像（Moving Image）對位到 PDFF 的空間座標上。
血管強化與疊加：對位完成後的 T1 影像會經過 Frangi 濾波器處理以強化血管特徵，然後將強化後的血管圖疊加到 PDFF 影像上。
儲存結果：最後將這份包含血管標記的疊加影像，儲存成一個新的 DICOM 序列到 output/ 資料夾。
專案中包含了用於批次處理的 main.py 和一個用於互動式開發與除錯的 interactive_debugger.ipynb
```text


LiverMRI/
├── data/All_Patients/             # 病患 DICOM 資料存放處 (示意)
│   └── PATIENT_ID/
│       ├── Ax T1 FS BH+C/         # T1 對比增強 DICOM 影像
│       └── FatFrac  3D Ax IDEAL IQ BH/ # PDFF DICOM 影像
├── output/                        # 處理後 DICOM 影像的輸出資料夾
├── utils/
│   ├── data_loader.py           # DICOM 序列載入工具
│   └── image_processing.py      # 影像對位與 Frangi 濾波工具
├── main.py                        # 批次處理主腳本
├── interactive_debugger.ipynb     # Jupyter Notebook 互動測試環境
├── requirements.txt               # Python 套件依賴列表
└── README.md                      # 本檔案
