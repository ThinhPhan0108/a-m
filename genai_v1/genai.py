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
import json # Re-add json import
import pandas as pd
import datetime
from time import sleep
from worldquant import WorldQuant
import csv
import itertools # Thêm import itertools để tạo hiệu ứng quay
from genai_v1.google_sheets_manager import GoogleSheetsManager
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable # Import các loại lỗi cụ thể

# Removed print_spinner function as requested for cleaner output

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
            # Cố gắng tải JSON. Nếu là dictionary có 'list_key', lấy list_key. Nếu là list trực tiếp, dùng luôn.
            loaded_data = json.loads(google_api_keys_str)
            if isinstance(loaded_data, dict) and 'list_key' in loaded_data:
                self.list_key = loaded_data['list_key']
            elif isinstance(loaded_data, list):
                self.list_key = loaded_data
            else:
                raise ValueError("Định dạng GOOGLE_API_KEYS không hợp lệ. Phải là mảng JSON hoặc đối tượng JSON với khóa 'list_key'.")
        else:
            # Fallback nếu không có biến môi trường (chỉ dùng cho dev cục bộ)
            data_key=self.read_json(os.path.join(parent_dir, 'keyapi.json'))
            self.list_key=data_key.get('list_key', []) # Sử dụng .get với giá trị mặc định là list rỗng

        if not self.list_key:
            raise ValueError("Không tìm thấy GOOGLE_API_KEYS. Vui lòng kiểm tra biến môi trường GOOGLE_API_KEYS hoặc file keyapi.json.")
        
        # Đảm bảo index_key nằm trong phạm vi hợp lệ
        if not (0 <= self.index_key < len(self.list_key)):
            print(f"Cảnh báo: index_key ({self.index_key}) nằm ngoài phạm vi hợp lệ của list_key (0-{len(self.list_key)-1}). Đặt lại index_key về 0.")
            self.index_key = 0 # Đặt lại về 0 nếu không hợp lệ

        self.client=genai.Client(api_key=self.list_key[self.index_key])
        self.model_name="gemini-2.0-flash"

        self.group_prompt=open(os.path.join(parent_dir, 'genai_v1', 'prompt', 'group_hypothesis_prompt.txt'), "r", encoding="utf-8").read()
        self.sub_prompt=open(os.path.join(parent_dir, 'genai_v1', 'prompt', 'sub_hypothesis_prompt.txt'), "r", encoding="utf-8").read()
        self.sub_system=open(os.path.join(parent_dir, 'genai_v1', 'prompt', 'sub_hypothesis_system.txt'), "r", encoding="utf-8").read()
        self.alpha_prompt=open(os.path.join(parent_dir, 'genai_v1', 'prompt', 'alpha_prompt.txt'), "r", encoding="utf-8").read()
        self.alpha_system=open(os.path.join(parent_dir, 'genai_v1', 'prompt', 'alpha_system.txt'), "r", encoding="utf-8").read()

        self.sheets_manager = GoogleSheetsManager(credentials_path=os.path.join(parent_dir, 'apisheet.json'))

        #dữ liệu lịch sử phản hồi
        #response_history=wks.get_all_records()
        #chỉ lấy một số cột nhất định
        #desired_columns = ['Group Hypothesis','Sub Hypothesis','Description','Expression']
        #response_history = [{col: row[col] for col in desired_columns if col in row} for row in response_history]

        self.response_history="response_history:\n{}"


    # Đọc file JSON
    def read_json(self,file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)  # Tải dữ liệu JSON
        return data
    
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

    @retry(
        stop=stop_after_attempt(5), # Thử lại tối đa 5 lần
        wait=wait_exponential(multiplier=1, min=4, max=10), # Thời gian chờ tăng theo cấp số nhân (4s, 8s, 16s, ...)
        retry=retry_if_exception_type((ResourceExhausted, ServiceUnavailable)) # Chỉ thử lại khi gặp lỗi 429 (ResourceExhausted) hoặc 503 (ServiceUnavailable)
    )
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
    
    @retry(
        stop=stop_after_attempt(5), # Thử lại tối đa 5 lần
        wait=wait_exponential(multiplier=1, min=4, max=10), # Thời gian chờ tăng theo cấp số nhân
        retry=retry_if_exception_type((ResourceExhausted, ServiceUnavailable)) # Chỉ thử lại khi gặp lỗi 429 hoặc 503
    )
    def genai_sub_hypothesis(self,group_hypothesis,file_path=None):
        '''
        group_hypothesis: là 1 hàng trong df kết quả của genai_group_pdf
        file_path: đường dẫn đến tài liệu muốn mô hình đọc để kiếm sub_hypothesis
        '''
        #tạo đối tượng model
        # Sử dụng cơ chế xoay vòng API key cho mỗi lần gọi genai_sub_hypothesis
        self.index_key = (self.index_key + 1) % len(self.list_key)
        client=genai.Client(api_key=self.list_key[self.index_key])
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
    
    @retry(
        stop=stop_after_attempt(5), # Thử lại tối đa 5 lần
        wait=wait_exponential(multiplier=1, min=4, max=10), # Thời gian chờ tăng theo cấp số nhân
        retry=retry_if_exception_type((ResourceExhausted, ServiceUnavailable)) # Chỉ thử lại khi gặp lỗi 429 hoặc 503
    )
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
            
        print("\n" + "="*50)
        print("BẮT ĐẦU QUÁ TRÌNH TẠO ALPHA")
        print("="*50 + "\n")

        print("Bước 1: Tạo Group Hypothesis...")
        #create group hypothesis
        df_group_hypothesis=self.genai_group_hypothesis(file_pdf_path,df_sub_hypothesis)
        #in kết quả
        print(f"✅ Hoàn thành tạo {len(df_group_hypothesis)} Group Hypothesis.\n")
        sleep(2)

        print('Bước 2: Tạo Sub Hypothesis và Alpha...')
        total_groups = len(df_group_hypothesis)
        for i in range(total_groups):
            group_hypothesis = df_group_hypothesis.loc[[i]]
            group_name = group_hypothesis['Group_Hypothesis'].iloc[0]
            print(f"\n--- Đang xử lý Group Hypothesis [{i+1}/{total_groups}]: {group_name} ---")
            try:
                df_sub_hypothesis=self.genai_sub_hypothesis(group_hypothesis,file_pdf_path)
                print(f"  ✅ Đã tạo {len(df_sub_hypothesis)} Sub Hypothesis cho nhóm này.")

                total_sub_hypotheses = len(df_sub_hypothesis)
                for j in range(total_sub_hypotheses):
                    sub_hypothesis=df_sub_hypothesis.loc[[j]]
                    auto_alpha=self.genai_alpha(sub_hypothesis)

                    # Kiểm tra auto_alpha có dữ liệu hợp lệ không
                    if auto_alpha is None or auto_alpha.empty:
                        print(f"    ❌ Tạo Alpha thất bại cho Sub Hypothesis [{j+1}/{total_sub_hypotheses}]. Bỏ qua mô phỏng.")
                        continue # Bỏ qua alpha này nếu tạo thất bại

                    expression_alpha=list(auto_alpha['Expression_alpha'])
                    alpha_code = expression_alpha[0] if expression_alpha and expression_alpha[0] else "N/A"
                    print(f"    ➡️ Alpha [{j+1}/{total_sub_hypotheses}] được tạo: {alpha_code}")

                    # Kiểm tra alpha_code trước khi mô phỏng
                    MAX_ALPHA_LENGTH = 128 # Giới hạn độ dài alpha theo yêu cầu
                    if alpha_code == 'invalid' or alpha_code == 'N/A' or len(alpha_code) > MAX_ALPHA_LENGTH:
                        print(f"    ❌ Alpha [{j+1}/{total_sub_hypotheses}] không hợp lệ (invalid/N/A/quá dài). Bỏ qua mô phỏng.")
                        alpha_data_list = auto_alpha.values.tolist()[0]
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
                        self.sheets_manager.append_rows([results]) # Ghi alpha không hợp lệ vào sheet
                        continue # Bỏ qua alpha này và chuyển sang alpha tiếp theo

                    #simulate alpha
                    neutralizations = ["NONE", "INDUSTRY", "MARKET"] # Cập nhật theo yêu cầu
                    truncations = [0.01] # Cập nhật theo yêu cầu
                    decays = [0, 512] # Cập nhật theo yêu cầu
                    pasteurizations = ["ON"] # Giữ nguyên
                    universes = ["TOP3000"] # Giữ nguyên
                    delays = [1] # Giữ nguyên
                    
                    alpha_configs_for_simulation = []
                    for neut in neutralizations:
                        for trunc in truncations:
                            for decay in decays:
                                for pasteur in pasteurizations:
                                    for uni in universes:
                                        for delay_val in delays:
                                            alpha_configs_for_simulation.append({
                                                'alpha_expression': expression_alpha[0],
                                                'decay': decay,
                                                'neut': neut,
                                                'truncation': trunc,
                                                'pasteurization': pasteur,
                                                'universe': uni,
                                                'delay': delay_val,
                                                'region': 'USA' # Assuming USA as default region
                                            })
                    
                    print(f"      ⏳ Bắt đầu mô phỏng {len(alpha_configs_for_simulation)} cấu hình cho Alpha này (chia thành nhóm 3)...")
                    
                    # Chia danh sách cấu hình thành các nhóm nhỏ (mỗi nhóm 3 cấu hình)
                    chunk_size = 3
                    chunks = [alpha_configs_for_simulation[x:x+chunk_size] for x in range(0, len(alpha_configs_for_simulation), chunk_size)]

                    alpha_simulation_failed = False
                    all_sim_results = [] # Để lưu trữ tất cả kết quả mô phỏng

                    for chunk_idx, chunk in enumerate(chunks):
                        print(f"      ➡️ Đang mô phỏng nhóm {chunk_idx + 1}/{len(chunks)} ({len(chunk)} cấu hình)...")
                        try:
                            sim_results_chunk = wl.simulate(chunk)
                            all_sim_results.extend(sim_results_chunk) # Thêm kết quả của nhóm vào danh sách tổng
                        except Exception as e:
                            print(f"      ❌ LỖI khi mô phỏng nhóm {chunk_idx + 1}: {e}. Các cấu hình trong nhóm này sẽ được coi là thất bại.")
                            # Thêm các giá trị None tương ứng với số lượng cấu hình trong chunk
                            all_sim_results.extend([None] * len(chunk))
                            alpha_simulation_failed = True # Đánh dấu alpha này là thất bại

                        # Thêm độ trễ 3 giây giữa các nhóm, trừ nhóm cuối cùng
                        if chunk_idx < len(chunks) - 1:
                            print("      Đang chờ 3 giây trước khi mô phỏng nhóm tiếp theo...")
                            sleep(3)
                    
                    # Xử lý tất cả kết quả mô phỏng sau khi tất cả các nhóm đã chạy
                    for idx, sim_metrics in enumerate(all_sim_results):
                        config = alpha_configs_for_simulation[idx] # Lấy cấu hình gốc
                        if not isinstance(sim_metrics, list) or sim_metrics == [None]:
                            print(f"      ❌ Mô phỏng thất bại cho cấu hình {config}. Bỏ qua.")
                            alpha_simulation_failed = True # Đánh dấu alpha này là thất bại
                        else:
                            print(f"      ✅ Mô phỏng hoàn tất cho cấu hình {config} và kết quả đã được lưu.")
                            alpha_data_list = auto_alpha.values.tolist()[0]
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
                            self.sheets_manager.append_rows([results])
                    
                    if alpha_simulation_failed:
                        # Nếu bất kỳ cấu hình nào thất bại, ghi alpha là không thành công
                        alpha_data_list = auto_alpha.values.tolist()[0]
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
                        self.sheets_manager.append_rows([results]) # Ghi alpha không thành công vào sheet
                        print(f"    ❌ Alpha [{j+1}/{total_sub_hypotheses}] mô phỏng thất bại hoàn toàn, đã lưu thông tin.")
                    sleep(2) # Giữ nguyên thời gian chờ giữa các alpha (sau khi tất cả các nhóm đã chạy)
            except Exception as e:
                print(f'  ❌ LỖI trong quá trình tạo Sub Hypothesis/Alpha cho nhóm "{group_name}": {e}')
                sleep(5) # Giảm thời gian chờ khi có lỗi
                
import traceback # Import traceback module

if __name__ == '__main__':
    # Đọc các giá trị từ biến môi trường
    index_key_str = os.getenv('INDEX_KEY', '0') # Mặc định là '0' nếu không có
    index_key = int(index_key_str) if index_key_str else 0 # Xử lý trường hợp chuỗi rỗng

    file_pdf_path = os.getenv('FILE_PDF_PATH', None) # Mặc định là None nếu không có
    file_sub_hypothesis_path = os.getenv('FILE_SUB_HYPOTHESIS_PATH', os.path.join(parent_dir, 'Finding Alpha - Sub Hypothesis.csv')) # Mặc định là đường dẫn này

    wl=WorldQuant(credentials_path=os.path.join(parent_dir, 'credential.json'))
    
    # Đọc các giá trị cho vòng lặp từ biến môi trường
    max_run_cycles_str = os.getenv('MAX_RUN_CYCLES', '1') # Mặc định 1 chu kỳ nếu không có
    max_run_cycles = int(max_run_cycles_str) if max_run_cycles_str.isdigit() else 1

    sleep_between_cycles_str = os.getenv('SLEEP_BETWEEN_CYCLES_SECONDS', '300') # Mặc định 300 giây (5 phút)
    sleep_between_cycles = int(sleep_between_cycles_str) if sleep_between_cycles_str.isdigit() else 300

    cycle_count = 0
    while cycle_count < max_run_cycles:
        try:
            print("\n" + "="*50)
            print(f"BẮT ĐẦU CHU KỲ ĐÀO ALPHA MỚI [{cycle_count + 1}/{max_run_cycles if max_run_cycles > 0 else 'VÔ HẠN'}]")
            print("="*50 + "\n")
            GenAI(index_key).run(file_pdf_path,file_sub_hypothesis_path)
            print("\n" + "="*50)
            print(f"CHU KỲ ĐÀO ALPHA ĐÃ HOÀN TẤT [{cycle_count + 1}/{max_run_cycles if max_run_cycles > 0 else 'VÔ HẠN'}]")
            print("="*50 + "\n")
            cycle_count += 1
            if cycle_count < max_run_cycles:
                print(f"Đang chờ {sleep_between_cycles} giây trước khi bắt đầu chu kỳ tiếp theo...")
                sleep(sleep_between_cycles)
        except Exception as e:
            print(f'\n❌ LỖI trong quá trình chạy dự án ở chu kỳ {cycle_count + 1}:')
            traceback.print_exc() # Print full traceback
            print("Tiếp tục với chu kỳ tiếp theo sau khi gặp lỗi...")
            cycle_count += 1 # Vẫn tăng số chu kỳ để tránh lặp vô hạn nếu lỗi liên tục
            sleep(sleep_between_cycles) # Vẫn chờ trước khi thử lại
    
    print("\n" + "="*50)
    print("TẤT CẢ CÁC CHU KỲ ĐÀO ALPHA ĐÃ HOÀN TẤT")
    print("="*50 + "\n")
