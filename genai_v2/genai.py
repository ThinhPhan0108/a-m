'''
Quy trình:
data fields --> select Variable --> Sub_hypothesis --> alpha --> simulate --> update sheet
'''

import sys
import os
import pathlib
import json
import pandas as pd
import datetime
import gspread
from time import sleep
import csv
from google import genai
from google.genai import types
from pydantic import BaseModel

# Định nghĩa các class model
class genai_alpha_format(BaseModel):
    Variables_Used: list[str]
    Sub_Hypothesis: str
    Description: str
    Expression: str
    Expression_alpha: str

class genai_sub_format(BaseModel):
    Variables_Used: list[str]
    Sub_Hypothesis: str
    Description: str
    Expression: str

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

try:
    from worldquant import WorldQuant
    wl = WorldQuant(credentials_path=os.path.join(parent_dir, 'credential.json'))
except Exception as e:
    print(f"Lỗi khởi tạo WorldQuant: {e}")
    wl = None

class GenAI:
    def __init__(self, index_key=0):
        self.date = datetime.datetime.now().strftime('%d-%m-%Y')
        self.process_name = 'version_2'

        try:
            data_key = self.read_json(os.path.join(parent_dir, 'keyapi.json'))
            self.list_key = data_key.get('list_key', [])
            if not self.list_key:
                raise ValueError("list_key rỗng trong keyapi.json")
        except Exception as e:
            print(f"Lỗi đọc keyapi.json: {e}")
            self.list_key = []
        
        self.index_key = index_key
        self.client = genai.Client(api_key=self.list_key[self.index_key]) if self.list_key else None

        try:
            self.sub_prompt = open(os.path.join(parent_dir, 'genai_v2', 'prompt', 'sub_hypothesis_prompt.txt'), "r", encoding="utf-8").read()
            self.alpha_prompt = open(os.path.join(parent_dir, 'genai_v1', 'prompt', 'alpha_prompt.txt'), "r", encoding="utf-8").read()
            self.alpha_system = open(os.path.join(parent_dir, 'genai_v1', 'prompt', 'alpha_system.txt'), "r", encoding="utf-8").read()
        except Exception as e:
            print(f"Lỗi đọc file prompt: {e}")
            self.sub_prompt = ""
            self.alpha_prompt = ""
            self.alpha_system = ""

        try:
            gc = gspread.service_account(filename=os.path.join(parent_dir, 'apisheet.json'))
            wks = gc.open("Finding Alpha").worksheet("Auto_alpha_demo")
            self.wks = wks
            response_history = wks.get_all_records()
            desired_columns = ['Sub Hypothesis', 'Description', 'Expression']
            response_history = [{col: row[col] for col in desired_columns if col in row} for row in response_history]
            self.response_history = f"response_history:\n{json.dumps(response_history, ensure_ascii=False, indent=2)}"
        except Exception as e:
            print(f"Lỗi kết nối Google Sheets hoặc lấy dữ liệu: {e}")
            self.wks = None
            self.response_history = "response_history:\n[]"

    def read_json(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    
    def append_rows(self, result_simulate):
        if not self.wks:
            try:
                with open(os.path.join(parent_dir, 'results.csv'), mode='a', newline='', encoding='utf-8') as file:
                    writer = csv.writer(file)
                    writer.writerows(result_simulate)
                print("Ghi kết quả vào results.csv do không kết nối được Google Sheets.")
            except Exception as e:
                print(f'Lỗi ghi file backup results.csv: {e}')
        else:
            try:
                self.wks.append_rows(result_simulate)
            except Exception as e:
                print(f'Lỗi append_rows vào Google Sheets: {e}')
                try:
                    with open(os.path.join(parent_dir, 'results.csv'), mode='a', newline='', encoding='utf-8') as file:
                        writer = csv.writer(file)
                        writer.writerows(result_simulate)
                    print("Ghi kết quả vào results.csv do lỗi append_rows Google Sheets.")
                except Exception as e2:
                    print(f'Lỗi ghi file backup results.csv: {e2}')

    # Bỏ file_path khỏi contents_prompt: chỉ truyền DataFrame + prompt
    def contents_prompt(self, file_path, df, prompt):
        # file_path luôn None ở đây, bỏ luôn đọc file PDF
        text_json = None
        if df is not None:
            try:
                text_json = df.to_json(orient="records", force_ascii=False, indent=2)
            except Exception as e:
                print(f"Lỗi chuyển df thành json: {e}")
                text_json = None

        if text_json:
            contents = [text_json, prompt]
        else:
            contents = [prompt]

        return contents

    def genai_sub_hypothesis(self, group_hypothesis, file_path=None):
        if not self.list_key:
            raise RuntimeError("Danh sách API key rỗng, không thể gọi genai_sub_hypothesis")
        self.index_key = (self.index_key + 1) % len(self.list_key)
        client = genai.Client(api_key=self.list_key[self.index_key])
        
        contents = self.contents_prompt(None, group_hypothesis, self.sub_prompt) + [self.response_history]

        response = client.models.generate_content(
            model=self.client.model_name if hasattr(self.client, 'model_name') else "gemini-2.0-flash",
            contents=contents,
            config={
                "response_mime_type": "application/json",
                "response_schema": list[genai_sub_format],
                "system_instruction": self.alpha_system
            }
        )

        results = json.loads(response.text)
        return pd.DataFrame(results)

    def genai_alpha(self, sub_hypothesis):
        if not self.client:
            raise RuntimeError("Client genai chưa được khởi tạo.")
        contents = self.contents_prompt(None, sub_hypothesis, self.alpha_prompt)

        response = self.client.models.generate_content(
            model=self.client.model_name if hasattr(self.client, 'model_name') else "gemini-2.0-flash",
            contents=contents,
            config={
                "response_mime_type": "application/json",
                "response_schema": list[genai_alpha_format],
                "system_instruction": self.alpha_system
            }
        )
        results = json.loads(response.text)
        return pd.DataFrame(results)

    def get_score(self, alpha_id):
        if not wl:
            return None
        return wl.get_score(alpha_id)
    
    def get_corr(self, alpha_id):
        if not wl:
            return None
        return wl.get_corr(alpha_id)
    
    def get_pl(self, alpha_id):
        if not wl:
            return None
        return wl.get_pl(alpha_id)
    
    def get_turnover(self, alpha_id):
        if not wl:
            return None
        return wl.get_turnover(alpha_id)

    def run(self, file_pdf_path=None, type_category='dataset'):
        if file_pdf_path:
            file_name = os.path.basename(file_pdf_path)
        else:
            file_name = None

        if not wl:
            print("Cảnh báo: wl (WorldQuant) chưa được khởi tạo, chức năng simulate sẽ bị hạn chế.")

        print('Truy cập vào danh sách các biến')
        try:
            datafields = wl.get_datafields() if wl else pd.DataFrame()
        except Exception as e:
            print(f"Lỗi lấy datafields: {e}")
            datafields = pd.DataFrame()

        condition = (datafields.get('type') == 'MATRIX') | (datafields.get('type') == 'VECTOR') if not datafields.empty else False
        if condition is not False:
            datafields = datafields[condition].sort_values(by=['alphaCount', 'userCount'], ascending=False, ignore_index=True)
            datafields = datafields.loc[20:].reset_index(drop=True)  # bỏ 20 biến thông dụng đầu
        else:
            datafields = pd.DataFrame()

        print('Create sub hypothesis')
        list_category = datafields[type_category].value_counts().index if not datafields.empty else []
        for i in list_category:
            try:
                variable = datafields[datafields[type_category] == i]
                df_sub_hypothesis = self.genai_sub_hypothesis(variable, None)  # Không truyền file_pdf_path

                for j in range(len(df_sub_hypothesis)):
                    sub_hypothesis = df_sub_hypothesis.loc[[j]]
                    auto_alpha = self.genai_alpha(sub_hypothesis)
                    auto_alpha['Variables_Used'] = auto_alpha['Variables_Used'].astype(str)
                    json_text = auto_alpha.to_json(orient="records", force_ascii=False, indent=2)
                    print(json_text)

                    expression_alpha = list(auto_alpha['Expression_alpha'])
                    if expression_alpha[0] != 'invalid':
                        neutralizations = ["NONE", "INDUSTRY", "MARKET"]
                        truncations = [0.01]
                        decays = [0, 512]
                        pasteurizations = ["ON"]
                        universes = ["TOP3000"]
                        delays = [1]

                        alpha_configs = []
                        for neut in neutralizations:
                            for trunc in truncations:
                                for decay in decays:
                                    for pasteur in pasteurizations:
                                        for uni in universes:
                                            for delay_val in delays:
                                                alpha_configs.append({
                                                    'alpha_expression': expression_alpha[0],
                                                    'decay': decay,
                                                    'neut': neut,
                                                    'truncation': trunc,
                                                    'pasteurization': pasteur,
                                                    'universe': uni,
                                                    'delay': delay_val,
                                                    'region': 'USA'
                                                })

                        print(f"  Bắt đầu mô phỏng {len(alpha_configs)} cấu hình cho Alpha này (chia nhóm 3)...")
                        chunk_size = 3
                        chunks = [alpha_configs[x:x+chunk_size] for x in range(0, len(alpha_configs), chunk_size)]

                        all_sim_results = []
                        for chunk_idx, chunk in enumerate(chunks):
                            print(f"   Mô phỏng nhóm {chunk_idx + 1}/{len(chunks)} ({len(chunk)} cấu hình)...")
                            if wl:
                                sim_results = wl.simulate(chunk)
                            else:
                                sim_results = [None] * len(chunk)
                                print("Cảnh báo: wl chưa khởi tạo, bỏ qua simulate.")
                            all_sim_results.extend(sim_results)
                            if chunk_idx < len(chunks) - 1:
                                print("   Đợi 3 giây trước khi chạy nhóm tiếp theo...")
                                sleep(3)

                        for idx, sim_metrics in enumerate(all_sim_results):
                            config = alpha_configs[idx]
                            if not isinstance(sim_metrics, list) or sim_metrics == [None]:
                                print(f"   ❌ Mô phỏng thất bại cho cấu hình: {config}")
                                alpha_data_list = auto_alpha.values.tolist()[0]
                                results = [
                                    self.date, self.process_name, file_name,
                                    alpha_data_list[0], alpha_data_list[1], alpha_data_list[2], alpha_data_list[3], alpha_data_list[4],
                                    None, None, None, None, None, None, None,
                                    None, None
                                ]
                            else:
                                print(f"   ✅ Mô phỏng thành công cho cấu hình: {config}")
                                alpha_data_list = auto_alpha.values.tolist()[0]

                                alpha_id = None
                                try:
                                    settings = sim_metrics[6]
                                    if isinstance(settings, str):
                                        import re
                                        match = re.search(r'alpha_id=([a-zA-Z0-9\-]+)', settings)
                                        if match:
                                            alpha_id = match.group(1)
                                    elif isinstance(settings, dict):
                                        alpha_id = settings.get('alpha_id')
                                except Exception:
                                    alpha_id = None

                                score = None
                                min_corr = None
                                max_corr = None
                                if alpha_id and wl:
                                    score_list = self.get_score(alpha_id)
                                    if score_list:
                                        score = score_list[0]
                                    corr_list = self.get_corr(alpha_id)
                                    if corr_list:
                                        min_corr, max_corr = corr_list

                                results = [
                                    self.date, self.process_name, file_name,
                                    alpha_data_list[0], alpha_data_list[1], alpha_data_list[2], alpha_data_list[3], alpha_data_list[4],
                                    sim_metrics[0], sim_metrics[1], sim_metrics[2], sim_metrics[3], sim_metrics[4], sim_metrics[5], sim_metrics[6],
                                    score, min_corr
                                ]
                            self.append_rows([results])

                    else:
                        results = [[self.date, self.process_name, file_name] + auto_alpha.values.tolist()[0]]
                        self.append_rows(results)
                    sleep(10)
            except Exception as e:
                print(f'ERROR RUN {e}')
                sleep(30)

if __name__ == '__main__':
    index_key_str = os.getenv('INDEX_KEY', '0')
    index_key = int(index_key_str) if index_key_str.isdigit() else 0

    file_pdf_path = os.getenv('FILE_PDF_PATH', None)  # Không dùng trong code này, có thể truyền None
    file_sub_hypothesis_path = os.getenv('FILE_SUB_HYPOTHESIS_PATH', None)  # Không dùng

    max_run_cycles_str = os.getenv('MAX_RUN_CYCLES', '1')
    max_run_cycles = int(max_run_cycles_str) if max_run_cycles_str.isdigit() else 1

    sleep_between_cycles_str = os.getenv('SLEEP_BETWEEN_CYCLES_SECONDS', '300')
    sleep_between_cycles = int(sleep_between_cycles_str) if sleep_between_cycles_str.isdigit() else 300

    cycle_count = 0
    while cycle_count < max_run_cycles:
        try:
            print("\n" + "="*50)
            print(f"BẮT ĐẦU CHU KỲ ĐÀO ALPHA MỚI [{cycle_count + 1}/{max_run_cycles}]")
            print("="*50 + "\n")
            GenAI(index_key).run(None)  # Không truyền file_pdf_path
            print("\n" + "="*50)
            print(f"CHU KỲ ĐÀO ALPHA ĐÃ HOÀN TẤT [{cycle_count + 1}/{max_run_cycles}]")
            print("="*50 + "\n")
            cycle_count += 1
            if cycle_count < max_run_cycles:
                print(f"Đang chờ {sleep_between_cycles} giây trước khi bắt đầu chu kỳ tiếp theo...")
                sleep(sleep_between_cycles)
        except Exception as e:
            import traceback
            print(f'\n❌ LỖI trong quá trình chạy dự án ở chu kỳ {cycle_count + 1}:')
            traceback.print_exc()
            print("Tiếp tục với chu kỳ tiếp theo sau khi gặp lỗi...")
            cycle_count += 1
            sleep(sleep_between_cycles)

    print("\n" + "="*50)
    print("TẤT CẢ CÁC CHU KỲ ĐÀO ALPHA ĐÃ HOÀN TẤT")
    print("="*50 + "\n")
