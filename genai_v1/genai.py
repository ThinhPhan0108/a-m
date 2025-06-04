'''
Quy trình bắt đầu từ file document --> tạo Group_Hypothesis --> Tạo Sub Hypothesis --> Tạo biểu thức alpha --> simulate --> update sheet
'''
import sys
import os
# Lấy đường dẫn thư mục cha
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# Thêm vào sys.path nếu chưa có
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from google import genai
from google.genai import types
from pydantic import BaseModel
import pathlib
import json
import pandas as pd
import datetime
import gspread
from time import sleep
from worldquant import WorldQuant
import csv
import itertools # Thêm import itertools để tạo hiệu ứng quay

def print_spinner(message, duration=1):
    """In ra một spinner đơn giản để chỉ báo tiến độ."""
    spinner = itertools.cycle(['-', '\\', '|', '/'])
    for _ in range(duration * 10): # Lặp 10 lần mỗi giây
        sys.stdout.write(f'\r{message} {next(spinner)}')
        sys.stdout.flush()
        sleep(0.1)
    sys.stdout.write('\r' + ' ' * (len(message) + 2) + '\r') # Xóa spinner

class genai_group_format(BaseModel):
    Group_Hypothesis: str
    Definition: str
    Examples: str

class genai_sub_format(BaseModel):
    Group_Hypothesis: str
    Sub_Hypothesis: str
    Description: str
    Expression: str

class genai_alpha_format(BaseModel):
    Group_Hypothesis: str
    Sub_Hypothesis: str
    Description: str
    Expression: str
    Expression_alpha: str

class GenAI:
    def __init__(self,index_key=0):
        self.index_key = index_key # Lưu index_key như một thuộc tính
        self.date=datetime.datetime.now().strftime('%d-%m-%Y')
        self.process_name='version_1'
        
        # Đọc API keys từ biến môi trường
        google_api_keys_str = os.getenv('GOOGLE_API_KEYS')
        if google_api_keys_str:
            self.list_key = json.loads(google_api_keys_str)
        else:
            # Fallback nếu không có biến môi trường (chỉ dùng cho dev cục bộ)
            data_key=self.read_json('WorldQuant-master/keyapi.json')
            self.list_key=data_key['list_key']

        self.client=genai.Client(api_key=self.list_key[self.index_key]) # Sử dụng self.index_key
        self.model_name="gemini-2.0-flash"

        self.group_prompt=open("WorldQuant-master/genai_v1/prompt/group_hypothesis_prompt.txt", "r", encoding="utf-8").read()
        self.sub_prompt=open("WorldQuant-master/genai_v1/prompt/sub_hypothesis_prompt.txt", "r", encoding="utf-8").read()
        self.sub_system=open("WorldQuant-master/genai_v1/prompt/sub_hypothesis_system.txt", "r", encoding="utf-8").read()
        self.alpha_prompt=open("WorldQuant-master/genai_v1/prompt/alpha_prompt.txt", "r", encoding="utf-8").read()
        self.alpha_system=open("WorldQuant-master/genai_v1/prompt/alpha_system.txt", "r", encoding="utf-8").read()

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
            gc = gspread.service_account(filename='WorldQuant-master/apisheet.json')
        
        wks = gc.open("Finding Alpha").worksheet("Auto_alpha_demo")
        self.wks=wks

        #dữ liệu lịch sử phản hồi
        response_history=wks.get_all_records()
        #chỉ lấy một số cột nhất định
        desired_columns = ['Group Hypothesis','Sub Hypothesis','Description','Expression']
        response_history = [{col: row[col] for col in desired_columns if col in row} for row in response_history]

        self.response_history=f"response_history:\n{json.dumps(response_history, ensure_ascii=False, indent=2)}"


    # Đọc file JSON
    def read_json(self,file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)  # Tải dữ liệu JSON
        return data
    
    def append_rows(self,result_simulate):
        try:
            self.wks.append_rows(result_simulate)
        except Exception as e:
            with open('WorldQuant-master/results.csv', mode='a', newline='', encoding='utf-8') as file: # Sửa đường dẫn
                writer = csv.writer(file)
                writer.writerows(result_simulate)
            print(f'Lỗi khi ghi vào Google Sheet, đã lưu vào results.csv: {e}') # Thông báo lỗi rõ ràng hơn

    def contents_prompt(self,file_path,df,prompt):
        if file_path:
            #đọc file pdf
            file = pathlib.Path(file_path)
            obj_file=types.Part.from_bytes(data=file.read_bytes(),mime_type='application/pdf',)
        if df is not None:
            text_json=df.to_json(orient="records", force_ascii=False, indent=2)
        
        if file_path and df is not None:
            contents=[obj_file,text_json,prompt]
        elif file_path and df is None:
            contents=[obj_file,prompt]
        elif not file_path and df is not None:
            contents=[text_json,prompt]
        else:
            contents=[prompt]

        return contents

    def genai_group_hypothesis(self,file_path=None,df_sub_hypothesis=None):
        '''
        file_path: đường dẫn đến tài liệu để mô hình lấy group hypothesis
        '''
        #tạo contents
        contents=self.contents_prompt(file_path,df_sub_hypothesis,self.group_prompt)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config={
                "response_mime_type": "application/json",
                "response_schema": list[genai_group_format],
                }
                )
        #xử lý đầu ra
        results=json.loads(response.text)
        results=pd.DataFrame(results)
        return results
    
    
    def genai_sub_hypothesis(self,group_hypothesis,file_path=None):
        '''
        group_hypothesis: là 1 hàng trong df kết quả của genai_group_pdf
        file_path: đường dẫn đến tài liệu muốn mô hình đọc để kiếm sub_hypothesis
        '''
        #tạo đối tượng model
        client=genai.Client(api_key=self.list_key[self.index_key]) # Sửa lỗi IndexError
        contents=self.contents_prompt(file_path,group_hypothesis,self.sub_prompt)+[self.response_history]
        
        response = client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config={
                "response_mime_type": "application/json",
                "response_schema": list[genai_sub_format],
                #"system_instruction": {self.sub_system}
                }
                )
        #xử lý đầu ra
        results=json.loads(response.text)
        results=pd.DataFrame(results)
        return results
    
    def genai_alpha(self,sub_hypothesis):
        contents=self.contents_prompt(None,sub_hypothesis,self.alpha_prompt)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config={
                "response_mime_type": "application/json",
                "response_schema": list[genai_alpha_format],
                "system_instruction": {self.alpha_system}
                }
                )
        #xử lý đầu ra
        results=json.loads(response.text)
        results=pd.DataFrame(results)
        return results
    
    def run(self,file_pdf_path=None,file_sub_hypothesis_path=None):
        '''
        file_pdf_path: đường dẫn đến file tài liệu
        file_sub_hypothesis_path: đường dẫn đến file csv chứa sub hypothesis
        '''
        if file_pdf_path:
            file_pdf_path = file_pdf_path.strip('"') # Loại bỏ dấu ngoặc kép
            file_name = os.path.basename(file_pdf_path)
        else:
            file_name=None

        #Kiểm tra và chuyển file_sub_hypothesis_path thành dataframe
        if file_sub_hypothesis_path:
            file_sub_hypothesis_path = file_sub_hypothesis_path.strip('"') # Loại bỏ dấu ngoặc kép
            df_sub_hypothesis=pd.read_csv(file_sub_hypothesis_path)
            
        else:
            df_sub_hypothesis=None
            
        print("\n✨ Bắt đầu quá trình tạo Alpha ✨")
        print("------------------------------------")
        print("⏳ Bước 1: Tạo Group Hypothesis...")
        #create group hypothesis
        df_group_hypothesis=self.genai_group_hypothesis(file_pdf_path,df_sub_hypothesis)
        #in kết quả
        print(f"✅ Đã tạo {len(df_group_hypothesis)} Group Hypothesis.")
        print("------------------------------------")
        sleep(10)

        print('⏳ Bước 2: Tạo Sub Hypothesis và Alpha...')
        total_groups = len(df_group_hypothesis)
        for i in range(total_groups):
            group_hypothesis = df_group_hypothesis.loc[[i]]
            group_name = group_hypothesis['Group_Hypothesis'].iloc[0]
            print(f"\n  ➡️ Đang xử lý Group Hypothesis [{i+1}/{total_groups}]: {group_name}")
            try:
                df_sub_hypothesis=self.genai_sub_hypothesis(group_hypothesis,file_pdf_path)
                print(f"    ✅ Đã tạo {len(df_sub_hypothesis)} Sub Hypothesis cho nhóm này.")

                total_sub_hypotheses = len(df_sub_hypothesis)
                for j in range(total_sub_hypotheses):
                    sub_hypothesis=df_sub_hypothesis.loc[[j]]
                    auto_alpha=self.genai_alpha(sub_hypothesis)

                    # Kiểm tra auto_alpha có dữ liệu hợp lệ không
                    if auto_alpha is None or auto_alpha.empty:
                        print(f"      ❌ Tạo Alpha thất bại cho Sub Hypothesis [{j+1}/{total_sub_hypotheses}]. Bỏ qua mô phỏng.")
                        continue # Bỏ qua alpha này nếu tạo thất bại

                    expression_alpha=list(auto_alpha['Expression_alpha'])
                    alpha_code = expression_alpha[0] if expression_alpha and expression_alpha[0] else "N/A"
                    print(f"      ➡️ Alpha [{j+1}/{total_sub_hypotheses}] được tạo: {alpha_code}")

                    #simulate alpha
                    if alpha_code != 'invalid':
                        neutralizations = ["NONE", "MARKET", "SECTOR", "INDUSTRY", "SUBINDUSTRY"] # Đảm bảo đã cập nhật
                        truncations = [1, 0.1, 0.01, 0.001] # Cập nhật
                        decays = [0, 100, 200, 300, 400, 512] # Cập nhật
                        pasteurizations = ["ON", "OFF"] # Thêm
                        universes = ["TOP3000", "TOP1000", "TOP200", "TOPSP500"] # Thêm
                        delays = [1] # Loại bỏ 0 vì không hợp lệ
                        
                        total_simulations = len(neutralizations) * len(truncations) * len(decays) * \
                                            len(pasteurizations) * len(universes) * len(delays)
                        sim_count = 0
                        for neut in neutralizations:
                            for trunc in truncations:
                                for decay in decays:
                                    for pasteur in pasteurizations:
                                        for uni in universes:
                                            for delay_val in delays:
                                                sim_count += 1
                                                print(f"        ⏳ Mô phỏng ({sim_count}/{total_simulations}) - Neut: '{neut}', Trunc: {trunc}, Decay: {decay}, Pasteur: {pasteur}, Uni: {uni}, Delay: {delay_val}...")
                                                
                                                result_simulate=wl.single_simulate(
                                                    expression_alpha, 
                                                    decay=decay, 
                                                    neut=neut, 
                                                    truncation=trunc,
                                                    pasteurization=pasteur, # Truyền
                                                    universe=uni, # Truyền
                                                    delay=delay_val # Truyền
                                                )
                                                
                                                # Xử lý trường hợp result_simulate không phải là list hợp lệ
                                                if not isinstance(result_simulate, list) or result_simulate == [None]:
                                                    sim_metrics = [None]*7 # 7 giá trị None cho các chỉ số mô phỏng
                                                    print(f"        ❌ Mô phỏng thất bại cho cấu hình này.")
                                                else:
                                                    sim_metrics = result_simulate
                                                    print(f"        ✅ Mô phỏng hoàn tất và kết quả đã được lưu.")
                                                
                                                alpha_data_list = auto_alpha.values.tolist()[0]
                                                # Xây dựng danh sách kết quả theo đúng thứ tự cột mong muốn
                                                results = [
                                                    self.date,
                                                    self.process_name,
                                                    file_name,
                                                    alpha_data_list[0], # Group Hypothesis
                                                    alpha_data_list[1], # Sub Hypothesis
                                                    alpha_data_list[2], # Description
                                                    alpha_data_list[3], # Expression
                                                    alpha_data_list[4], # Expression_alpha
                                                    sim_metrics[0],     # Sharpe
                                                    sim_metrics[1],     # Turnover
                                                    sim_metrics[2],     # Fitness
                                                    sim_metrics[3],     # Returns
                                                    sim_metrics[4],     # Drawdown
                                                    sim_metrics[5],     # Margin
                                                    sim_metrics[6]      # Settings
                                                ]
                                                
                                                self.append_rows([results]) # append_rows nhận list of lists
                                                sleep(1) # Giảm thời gian chờ giữa các mô phỏng con
                    else:
                        alpha_data_list = auto_alpha.values.tolist()[0]
                        # Xây dựng danh sách kết quả cho alpha không hợp lệ
                        results = [
                            self.date,
                            self.process_name,
                            file_name,
                            alpha_data_list[0], # Group Hypothesis
                            alpha_data_list[1], # Sub Hypothesis
                            alpha_data_list[2], # Description
                            alpha_data_list[3], # Expression
                            alpha_data_list[4], # Expression_alpha
                            None, None, None, None, None, None, None # 7 giá trị None cho các chỉ số mô phỏng
                        ]
                        self.append_rows([results]) # append_rows nhận list of lists
                        print(f"      ❌ Alpha không hợp lệ, đã lưu thông tin.")
                    sleep(5) # Giảm thời gian chờ giữa các alpha
            except Exception as e:
                print(f'  ❌ LỖI trong quá trình tạo Sub Hypothesis/Alpha cho nhóm "{group_name}": {e}')
                sleep(15) # Giảm thời gian chờ khi có lỗi
                
if __name__ == '__main__':
    index_key=int(input('Nhập index key (int): '))
    file_pdf_path=input('Nhập đường dẫn đến file tài liệu (nếu có): ')
    file_sub_hypothesis_path=input('Nhập đường dẫn đến file sub hypothesis csv (nếu có):')
    wl=WorldQuant(credentials_path='WorldQuant-master/credential.json')
    while True:
        try:
            print("\n--- Bắt đầu chạy dự án đào Alpha ---")
            GenAI(index_key).run(file_pdf_path,file_sub_hypothesis_path)
            print("\n--- Dự án đào Alpha đã hoàn tất một chu kỳ ---")
        except Exception as e:
            print(f'\n❌ LỖI KHÁC trong quá trình chạy dự án: {e}')
            sleep(30) # Giảm thời gian chờ khi có lỗi
