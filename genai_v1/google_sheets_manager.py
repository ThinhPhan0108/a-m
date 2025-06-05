import gspread
import json
import os

class GoogleSheetsManager:
    def __init__(self, credentials_path=None):
        # Truy cập gg sheet từ biến môi trường
        google_sheet_credentials_str = os.getenv('GOOGLE_SHEET_CREDENTIALS')
        if google_sheet_credentials_str:
            # Ghi nội dung JSON vào một file tạm thời để gspread có thể đọc
            temp_apisheet_path = 'temp_apisheet.json'
            with open(temp_apisheet_path, 'w', encoding='utf-8') as f:
                json.dump(json.loads(google_sheet_credentials_str), f, ensure_ascii=False, indent=2)
            gc = gspread.service_account(filename=temp_apisheet_path)
        else:
            # Fallback nếu không có biến môi trường (chỉ dùng cho dev cục bộ)
            gc = gspread.service_account(filename=credentials_path)
        
        self.wks = gc.open("Finding Alpha").worksheet("Auto_alpha_demo")

    def append_rows(self, result_simulate):
        try:
            self.wks.append_rows(result_simulate)
        except Exception as e:
            print(f'Lỗi khi ghi vào Google Sheet: {e}')
            raise
