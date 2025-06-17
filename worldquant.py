import requests
import json
from requests.auth import HTTPBasicAuth
from typing import List, Dict
import pandas as pd
from time import sleep
import concurrent.futures
import time  # cần để dùng trong get_score và get_corr

'''# Configure logger with more detailed format
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG level for more detailed logs
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    filename='alpha_polisher.log'
)
logger = logging.getLogger(__name__)

# Also log to console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s: %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)'''

class WorldQuant:
    def __init__(self,credentials_path='WorldQuant-master/credential.json'):
        print("Initializing AlphaPolisher...")
        self.sess = requests.Session()
        self.credentials_path=credentials_path
        self.setup_auth(credentials_path)
        #self.operators = self.get_operators()
        #self.data_fields=self.get_datafields()
        #self.inaccessible_ops = ["log_diff", "s_log_1p", "fraction", "quantile"]
        print("AlphaPolisher initialized successfully")
    
    def setup_auth(self, credentials_path: str) -> None:
        """Set up authentication with WorldQuant Brain."""
        print(f"Loading credentials from {credentials_path}")
        try:
            with open(credentials_path) as f:
                credentials = json.load(f)
            
            username, password = credentials['username'],credentials['password']
            self.sess.auth = HTTPBasicAuth(username, password)
            
            print("Authenticating with WorldQuant Brain...")
            response = self.sess.post('https://api.worldquantbrain.com/authentication')
            print(f"Authentication response status: {response.status_code}")
            print(f"Authentication response: {response.text[:500]}...")
            
            if response.status_code != 201:
                raise Exception(f"Authentication failed: {response.text}")
            print("Authentication successful")
        except Exception as e:
            print(f"Authentication failed: {str(e)}")
            raise

    def get_operators(self) -> Dict:
        """Fetch available operators from WorldQuant Brain API."""
        print("Fetching available operators...")
        try:
            response = self.sess.get('https://api.worldquantbrain.com/operators')
            print(f"Operators response status: {response.status_code}")
            
            if response.status_code == 200:
                operators = response.json()
                print(f"Successfully fetched {len(operators)} operators")
                print(f"Operators: {json.dumps(operators, indent=2)}")
                return pd.DataFrame(operators)
            else:
                print(f"Failed to fetch operators: {response.text}")
                return {}
        except Exception as e:
            print(f"Error fetching operators: {str(e)}")
            return {}
        
    def get_datafields(self,
        instrument_type: str = 'EQUITY',
        region: str = 'USA',
        delay: int = 1,
        universe: str = 'TOP3000',
        dataset_id: str = '',
        search: str = ''
    ):
        if len(search) == 0:
            url_template = "https://api.worldquantbrain.com/data-fields?" +\
                f"&instrumentType={instrument_type}" +\
                f"&region={region}&delay={str(delay)}&universe={universe}&dataset.id={dataset_id}&limit=50" +\
                "&offset={x}"
            count = self.sess.get(url_template.format(x=0)).json()['count'] 
            
        else:
            url_template = "https://api.worldquantbrain.com/data-fields?" +\
                f"&instrumentType={instrument_type}" +\
                f"&region={region}&delay={str(delay)}&universe={universe}&limit=50" +\
                f"&search={search}" +\
                "&offset={x}"
            count = 100
        
        datafields_list = []
        for x in range(0, count, 50):
            datafields = self.sess.get(url_template.format(x=x))
            datafields_list.append(datafields.json()['results'])
     
        datafields_list_flat = [item for sublist in datafields_list for item in sublist]
     
        datafields_df = pd.DataFrame(datafields_list_flat)
        return datafields_df
    
    def get_vec_fields(self, fields):

        #vec_ops = ["vec_avg", "vec_sum", "vec_ir", "vec_max", "vec_count","vec_skewness","vec_stddev", "vec_choose"]
        vec_ops=["vec_avg", "vec_sum"]
        vec_fields = []
     
        for field in fields:
            for vec_op in vec_ops:
                if vec_op == "vec_choose":
                    vec_fields.append("%s(%s, nth=-1)"%(vec_op, field))
                    vec_fields.append("%s(%s, nth=0)"%(vec_op, field))
                else:
                    vec_fields.append("%s(%s)"%(vec_op, field))
     
        return(vec_fields)
    
    def process_datafields(self, df, data_type):

        if data_type == "matrix":
            datafields = df[df['type'] == "MATRIX"]["id"].tolist()
        elif data_type == "vector":
            datafields = self.get_vec_fields(df[df['type'] == "VECTOR"]["id"].tolist())

        tb_fields = []
        for field in datafields:
            tb_fields.append("winsorize(ts_backfill(%s, 120), std=4)"%field)
        return tb_fields
    
    def process_datafields_v2(self, df):
        mask_matrix = df['type'] == 'MATRIX'
        mask_vector = df['type'] == 'VECTOR'

        df.loc[mask_matrix, 'id'] = df.loc[mask_matrix, 'id'].apply(
            lambda var: f"winsorize(ts_backfill({var}, 120), std=4)"
        )
        df.loc[mask_vector, 'id'] = df.loc[mask_vector, 'id'].apply(
            lambda var: f"winsorize(ts_backfill(vec_avg({var}), 120), std=4)"
        )
        return df

    def generate_sim_data(self, alpha_list, decay, region, uni, neut, truncation, pasteurization, delay):
        sim_data_list = []
        for alpha in alpha_list:
            simulation_data = {
                'type': 'REGULAR',
                'settings': {
                    'instrumentType': 'EQUITY',
                    'region': region,
                    'universe': uni,
                    'delay': delay,
                    'decay': decay,
                    'neutralization': neut,
                    'truncation': truncation,
                    'pasteurization': pasteurization,
                    'unitHandling': 'VERIFY',
                    'nanHandling': 'OFF',
                    'language': 'FASTEXPR',
                    'visualization': False,
                },
                'regular': alpha}

            sim_data_list.append(simulation_data)
        return sim_data_list
    
    def simulate(self, alpha_configs: List[Dict]) -> List[List]:
        print(f"Starting parallel simulation for {len(alpha_configs)} alphas")
        
        results = []
        
        def _run_single_simulation(config):
            alpha_expression = config['alpha_expression']
            decay = config.get('decay', 0, 512)
            neut = config.get('neut', "MARKET", "NONE", "INDUSTRY")
            region = config.get('region', 'USA')
            universe = config.get('universe', 'TOP3000')
            truncation = config.get('truncation', 0.01)
            pasteurization = config.get('pasteurization', "ON")
            delay = config.get('delay', 1)

            sim_data_list = self.generate_sim_data([alpha_expression], decay, region, universe, neut, truncation, pasteurization, delay)
            sim_data = sim_data_list[0]

            try:
                simulation_response = self.sess.post('https://api.worldquantbrain.com/simulations', json=sim_data)
                if simulation_response.status_code == 401:
                    print(f"Session expired for alpha {alpha_expression[:50]}..., re-authenticating...")
                    self.setup_auth(self.credentials_path)
                    simulation_response = self.sess.post('https://api.worldquantbrain.com/simulations', json=sim_data)
                
                if simulation_response.status_code != 201:
                    error_detail = simulation_response.json().get("detail", "")
                    if "SIMULATION_LIMIT_EXCEEDED" in error_detail:
                        print(f"Rate limit exceeded for alpha {alpha_expression[:50]}... Retrying after backoff.")
                        sleep(10)
                        return _run_single_simulation(config)
                    else:
                        print(f"Simulation API error for alpha {alpha_expression[:50]}...: {simulation_response.text}")
                        return [None]
                
                simulation_progress_url = simulation_response.headers.get('Location')
                if simulation_progress_url:
                    retries = 0
                    max_retries = 5
                    while retries < max_retries:
                        try:
                            simulation_progress = self.sess.get(simulation_progress_url)
                            simulation_progress.raise_for_status()
                            simulation_progress = simulation_progress.json()
                            
                            if simulation_progress.get("detail") == "Incorrect authentication credentials.":
                                print(f'Incorrect authentication credentials for alpha {alpha_expression[:50]}...')
                                self.setup_auth(self.credentials_path)
                                return _run_single_simulation(config)
                            
                            elif simulation_progress.get("status") in ['COMPLETE', 'WARNING']:
                                alpha_id = simulation_progress.get("alpha")
                                result = self.locate_alpha(alpha_id)
                                print(f"Simulation complete for alpha {alpha_expression[:50]}...")
                                return result
                            
                            elif simulation_progress.get("status") in ["FAILED", "ERROR"]:
                                print(f'ERROR ALPHA {alpha_expression[:50]}...')
                                return [None]
                            
                            sleep(5)
                        except requests.exceptions.HTTPError as http_err:
                            if http_err.response.status_code == 429:
                                print(f"Rate limit exceeded while polling for alpha {alpha_expression[:50]}... Retrying after backoff.")
                                sleep_time = 2 ** retries
                                sleep(sleep_time)
                                retries += 1
                            else:
                                print(f"HTTP error while polling for alpha {alpha_expression[:50]}...: {http_err}")
                                return [None]
                        except requests.exceptions.RequestException as e:
                            print(f"Network error while polling for alpha {alpha_expression[:50]}...: {str(e)}")
                            return [None]
                        except json.JSONDecodeError as e:
                            print(f"JSON decode error while polling for alpha {alpha_expression[:50]}...: {str(e)}")
                            return [None]
                        except Exception as e:
                            print(f"Unexpected error while polling for alpha {alpha_expression[:50]}...: {str(e)}")
                            return [None]
                    print(f"Max retries exceeded for alpha {alpha_expression[:50]}...")
                    return [None]
                else:
                    print(f"No Location header in response for alpha {alpha_expression[:50]}...")
                    return [None]
            except requests.exceptions.RequestException as e:
                print(f"Network error during simulation request for alpha {alpha_expression[:50]}...: {str(e)}")
                return [None]
            except json.JSONDecodeError as e:
                print(f"JSON decode error during simulation request for alpha {alpha_expression[:50]}...: {str(e)}")
                return [None]
            except Exception as e:
                print(f"Unexpected error during simulation request for alpha {alpha_expression[:50]}...: {str(e)}")
                return [None]

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_alpha = {executor.submit(_run_single_simulation, config): config for config in alpha_configs}
            
            for future in concurrent.futures.as_completed(future_to_alpha):
                config = future_to_alpha[future]
                try:
                    sim_result = future.result()
                    results.append(sim_result)
                except Exception as exc:
                    print(f'Alpha {config["alpha_expression"][:50]}... generated an exception: {exc}')
                    results.append([None])
        
        return results
    
    def locate_alpha(self, alpha_id):
        alpha = self.sess.get("https://api.worldquantbrain.com/alphas/" + alpha_id)
        string = alpha.content.decode('utf-8')
        metrics = json.loads(string)
        
        sharpe = metrics["is"]["sharpe"]
        turnover = metrics["is"]["turnover"]
        fitness = metrics["is"]["fitness"]
        returns=metrics["is"]["returns"]
        drawdown=metrics["is"]["drawdown"]
        margin = metrics["is"]["margin"]
        settings=str(metrics['settings'])
        
        triple = [sharpe, turnover,fitness,returns,drawdown,margin,settings]
        triple = [ i if i != 'None' else None for i in triple]
        return triple

    def get_corr(self, alpha_id):
        start_time = time.time()
        timeout = 30

        while True:
            corr_respond = self.sess.get(f"https://api.worldquantbrain.com/alphas/{alpha_id}/correlations/self")
            corr = corr_respond.content.decode('utf-8')
            if corr:
                corr = json.loads(corr)
                if corr.get('min'):
                    min_corr = corr['min']
                    max_corr = corr['max']
                    return [min_corr, max_corr]

            if time.time() - start_time > timeout:
                return [None, None]

            sleep(5)

    def get_score(self, alpha_id):
        start_time = time.time()
        timeout = 30

        while True:
            performance_response = self.sess.get(f'https://api.worldquantbrain.com/competitions/IQC2025S2/alphas/{alpha_id}/before-and-after-performance')
            performance = performance_response.content.decode('utf-8')
            if performance:
                performance = json.loads(performance)
                if performance.get('score'):
                    before_score = performance['score']['before']
                    after_score = performance['score']['after']
                    score = after_score - before_score
                    return [score]

            if time.time() - start_time > timeout:
                return [None]

            sleep(5)

    def get_pl(self, alpha_id):
        while True:
            pl_obj = self.sess.get(f'https://api.worldquantbrain.com/alphas/{alpha_id}/recordsets/pnl')
            if pl_obj.content:
                pl = pl_obj.json()
                pl = pl.get('records')
                pl_df = pd.DataFrame(pl, columns=['date', 'returns'])
                pl_df['returns'] = pl_df['returns'] - pl_df['returns'].shift(1)
                pl_df.dropna(inplace=True)
                return pl_df

    def get_turnover(self, alpha_id):
        while True:
            turnover_obj = self.sess.get(f'https://api.worldquantbrain.com/alphas/{alpha_id}/recordsets/turnover')
            if turnover_obj.content:
                turnover = turnover_obj.json()
                turnover = turnover.get('records')
                turnover_df = pd.DataFrame(turnover, columns=['date', 'turnover'])
                turnover_df.dropna(inplace=True)
                return turnover_df
