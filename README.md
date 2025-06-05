## 專案結構

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
