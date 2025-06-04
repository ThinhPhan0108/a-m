'''
Quy trình:
data fields --> select Variable --> Sub_hypothesis --> alpha --> simulate --> update sheet
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
    def __init__(self,index_key=0):
        self.date=datetime.datetime.now().strftime('%d-%m-%Y')
        self.process_name='version_2'
        
        data_key=self.read_json('./keyapi.json')
        self.list_key=data_key['list_key']
        self.client=genai.Client(api_key=self.list_key[index_key])
        self.name_model="gemini-2.0-flash"

        self.sub_prompt=open("./genai_v2/prompt/sub_hypothesis_prompt.txt", "r", encoding="utf-8").read()

        self.alpha_prompt=open("./genai_v1/prompt/alpha_prompt.txt", "r", encoding="utf-8").read()
        self.alpha_system=open("./genai_v1/prompt/alpha_system.txt", "r", encoding="utf-8").read()

        #truy cập gg sheet
        gc = gspread.service_account(filename='./apisheet.json')
        wks = gc.open("Finding Alpha").worksheet("Auto_alpha_demo")
        self.wks=wks

        #dữ liệu lịch sử phản hồi
        response_history=wks.get_all_records()
        #chỉ lấy một số cột nhất định
        desired_columns = ['Sub Hypothesis','Description','Expression']
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
            with open('./results.csv', mode='a', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerows(result_simulate)
            print(f'ERROR {e}')

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

    def genai_sub_hypothesis(self,variable,file_path=None):
        '''
        variable: là một json gồm id và description
        file_path: đường dẫn đến tài liệu muốn mô hình đọc để kiếm sub_hypothesis
        '''
        #tạo đối tượng model
        client=genai.Client(api_key=self.list_key[5])
        contents=self.contents_prompt(file_path,variable,self.sub_prompt)+[self.response_history]
        
        response = client.models.generate_content(
            model=self.name_model,
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
            model=self.name_model,
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


    def run(self,file_pdf_path=None,type_category='dataset'):
        '''
        file_pdf_path: đường dẫn đến file tài liệu
        '''
        if file_pdf_path:
            file_name = os.path.basename(file_pdf_path)
        else:
            file_name=None

        #select variable
        print('Truy cập vào danh sách các biến')
        datafields=wl.get_datafields()
        condition= (datafields['type']=='MATRIX')|(datafields['type']=='VECTOR')
        datafields=datafields[condition].sort_values(by=['alphaCount','userCount'],ascending=False,ignore_index=True)
        datafields=datafields.loc[20:].reset_index(drop=True) # bỏ 20 biến thông dụng đầu 

        print('Create sub hypothesis')
        list_category= datafields[type_category].value_counts().index
        #create sub hypothesis
        for i in list_category:
            try:
                variable=datafields[datafields[type_category]==i]
                df_sub_hypothesis=self.genai_sub_hypothesis(variable,file_pdf_path)

                #create alpha
                for j in range(len(df_sub_hypothesis)):
                    sub_hypothesis=df_sub_hypothesis.loc[[j]]
                    auto_alpha=self.genai_alpha(sub_hypothesis)
                    auto_alpha['Variables_Used']=auto_alpha['Variables_Used'].astype(str)
                    #in kết quả
                    json_text=auto_alpha.to_json(orient="records", force_ascii=False, indent=2)
                    print(json_text)

                    #simulate alpha
                    expression_alpha=list(auto_alpha['Expression_alpha'])
                    if expression_alpha[0] != 'invalid':
                        result_simulate=wl.single_simulate(expression_alpha)
                        #chuyển kết quả lên gg sheet - auto_alpha là df --> [[]]; result_simulate là list --> []
                        results=[[self.date, self.process_name, file_name] +auto_alpha.values.tolist()[0]+result_simulate]
                        
                        self.append_rows(results)
                    else:
                        results=[[self.date, self.process_name, file_name] +auto_alpha.values.tolist()[0]]
                        self.append_rows(results)
                    sleep(10)
            except Exception as e:
                print(f'ERROR RUN {e}')
                sleep(30)

if __name__ == '__main__':
    index_key=int(input('Nhập index key (int): '))

    option = input('Nhập type_category (0: id | 1: dataset | 2: subcategory | 3: category) :')
    category_json={'0':'id','1':'dataset','2':'subcategory','3':'category'}
    type_category=category_json.get(option)

    file_pdf_path=input('Nhập đường dẫn đến file tài liệu (nếu có): ')

    wl=WorldQuant()
    while True:
        try:
            GenAI(index_key).run(file_pdf_path,type_category)
        except Exception as e:
            print(f'ERROR OTHER {e}')
            sleep(60)