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

class GenAI:
    def __init__(self, index_key=0):
        self.date = datetime.datetime.now().strftime('%d-%m-%Y')
        self.process_name = 'version_2'
        
        data_key = self.read_json('./keyapi.json')
        self.list_key = data_key['list_key']
        self.index_key = index_key
        self.client = genai.Client(api_key=self.list_key[self.index_key])
        self.name_model = "gemini-2.0-flash"

        self.sub_prompt = open("./genai_v2/prompt/sub_hypothesis_prompt.txt", "r", encoding="utf-8").read()
        self.alpha_prompt = open("./genai_v1/prompt/alpha_prompt.txt", "r", encoding="utf-8").read()
        self.alpha_system = open("./genai_v1/prompt/alpha_system.txt", "r", encoding="utf-8").read()

        # Truy cập Google Sheet
        gc = gspread.service_account(filename='./apisheet.json')
        wks = gc.open("Finding Alpha").worksheet("Auto_alpha_demo")
        self.wks = wks

        # Dữ liệu lịch sử phản hồi
        response_history = wks.get_all_records()
        desired_columns = ['Sub Hypothesis', 'Description', 'Expression']
        response_history = [{col: row[col] for col in desired_columns if col in row} for row in response_history]

        self.response_history = f"response_history:\n{json.dumps(response_history, ensure_ascii=False, indent=2)}"
    
    def read_json(self, file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    
    def append_rows(self, result_simulate):
        try:
            self.wks.append_rows(result_simulate)
        except Exception as e:
            with open('./results.csv', mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerows(result_simulate)
            print(f'ERROR {e}')

    def contents_prompt(self, file_path, df, prompt):
        if file_path:
            file = pathlib.Path(file_path)
            obj_file = types.Part.from_bytes(data=file.read_bytes(), mime_type='application/pdf')
        if df is not None:
            text_json = df.to_json(orient="records", force_ascii=False, indent=2)
        
        if file_path and df is not None:
            contents = [obj_file, text_json, prompt]
        elif file_path and df is None:
            contents = [obj_file, prompt]
        elif not file_path and df is not None:
            contents = [text_json, prompt]
        else:
            contents = [prompt]

        return contents

    def genai_sub_hypothesis(self, group_hypothesis, file_path=None):
        self.index_key = (self.index_key + 1) % len(self.list_key)
        client = genai.Client(api_key=self.list_key[self.index_key])
        
        contents = self.contents_prompt(file_path, group_hypothesis, self.sub_prompt) + [self.response_history]

        response = client.models.generate_content(
            model=self.name_model,
            contents=contents,
            config={
                "response_mime_type": "application/json",
                "response_schema": list[genai_sub_format],
                "system_instruction": self.alpha_system
            }
        )

        results = json.loads(response.text)
        results = pd.DataFrame(results)
        return results

    def genai_alpha(self, sub_hypothesis):
        contents = self.contents_prompt(None, sub_hypothesis, self.alpha_prompt)

        response = self.client.models.generate_content(
            model=self.name_model,
            contents=contents,
            config={
                "response_mime_type": "application/json",
                "response_schema": list[genai_alpha_format],
                "system_instruction": self.alpha_system
            }
        )
        results = json.loads(response.text)
        results = pd.DataFrame(results)
        return results

    # Bổ sung hàm lấy score
    def get_score(self, alpha_id):
        return wl.get_score(alpha_id)  # gọi trực tiếp hàm trong WorldQuant
    
    # Bổ sung hàm lấy correlation
    def get_corr(self, alpha_id):
        return wl.get_corr(alpha_id)
    
    # Bổ sung hàm lấy pnl
    def get_pl(self, alpha_id):
        return wl.get_pl(alpha_id)
    
    # Bổ sung hàm lấy turnover
    def get_turnover(self, alpha_id):
        return wl.get_turnover(alpha_id)

    def run(self, file_pdf_path=None, type_category='dataset'):
        if file_pdf_path:
            file_name = os.path.basename(file_pdf_path)
        else:
            file_name = None

        print('Truy cập vào danh sách các biến')
        datafields = wl.get_datafields()
        condition = (datafields['type'] == 'MATRIX') | (datafields['type'] == 'VECTOR')
        datafields = datafields[condition].sort_values(by=['alphaCount', 'userCount'], ascending=False, ignore_index=True)
        datafields = datafields.loc[20:].reset_index(drop=True)  # bỏ 20 biến thông dụng đầu 

        print('Create sub hypothesis')
        list_category = datafields[type_category].value_counts().index
        for i in list_category:
            try:
                variable = datafields[datafields[type_category] == i]
                df_sub_hypothesis = self.genai_sub_hypothesis(variable, file_pdf_path)

                for j in range(len(df_sub_hypothesis)):
                    sub_hypothesis = df_sub_hypothesis.loc[[j]]
                    auto_alpha = self.genai_alpha(sub_hypothesis)
                    auto_alpha['Variables_Used'] = auto_alpha['Variables_Used'].astype(str)
                    json_text = auto_alpha.to_json(orient="records", force_ascii=False, indent=2)
                    print(json_text)

                    expression_alpha = list(auto_alpha['Expression_alpha'])
                    if expression_alpha[0] != 'invalid':
                        # Simulate alpha nhiều config
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
                            sim_results = wl.simulate(chunk)
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
                                    None, None  # Thêm ô cho score và corr là None khi simulate lỗi
                                ]
                            else:
                                print(f"   ✅ Mô phỏng thành công cho cấu hình: {config}")
                                alpha_data_list = auto_alpha.values.tolist()[0]

                                # Lấy alpha_id từ settings string (cần parse hoặc lấy cách khác tùy response)
                                # Giả sử settings cuối cùng trả về dạng dict hoặc string chứa alpha_id, cần parse chính xác
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
                                if alpha_id:
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

