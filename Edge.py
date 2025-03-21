from TCP_code import TCP_COM
import time
import json
from network_control import network_control
from utils import make_dataset, generate_avro_schema, remove_all_avro_files
import pandas as pd
import zipfile
import numpy as np
import AVRO
import os
import queue
import shutil
import IoT_model
import matplotlib.pyplot as plt
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", module="sklearn")
class edge_device(TCP_COM):
    def __init__(self, REC_FILE_PATH):
        self.use_PDR=True
        self.total_sent_data=0
        self.device_type="edge"
        self.model_path="models"
        with open("configs.json", "r") as file:
            configs = json.load(file)
        self.local_IP=configs['edgeip']
        self.edgePORT_TCP=configs['edgePORT_TCP']
        self.edgePORT_UDP=configs['edgePORT_UDP']
        self.basePORT=configs['basePORT']
        self.rec_ip=configs['baseip']
        self.nc=network_control(self.device_type)
        if configs['use_config_network_control']==True:
            rate_kbps=configs['bandwidth_limit_kbps']
            burst_kbps=configs['burst_limit_kbps']
            latency_ms=configs['buffering_latency_ms']
            packet_loss_pct=configs['packet_loss_pct']
            #delay_ms=configs['base_delay_ms']
            #jitter_ms=configs['jitter_ms']
            #packet_loss_pct=None
            delay_ms=None
            jitter_ms=None
            #self.nc.set_network_conditions(rate_kbps, burst_kbps, latency_ms, packet_loss_pct, delay_ms, jitter_ms)
        edgePORT=(self.edgePORT_TCP, self.edgePORT_UDP)
        self.file_Q=queue.Queue()
        super().__init__(self.local_IP, edgePORT, self.rec_ip, self.basePORT, REC_FILE_PATH, self.device_type, self.file_Q)
        self.fault_index=0
        self.filename, self.start_offset=make_dataset(fault_index=self.fault_index, num=1)
        df=pd.read_csv(self.filename)
        self.timestamps=df['timestamp']
        self.data=df.drop(columns=['timestamp'])
        self.index=0
        sensors=np.shape(self.data)[1]
        self.len_of_dataset=np.shape(self.data)[0]
        self.schema_path="test_files/avro_"+str(sensors)+'.avsc'
        generate_avro_schema(sensors, self.schema_path)
        self.model = IoT_model.IoT_model("test_files/initial_data.csv")
        self.model.load_model()

    def analyze_samples(self):
        s, t=self.get_sample()    
        for_mse=np.array(s.drop('machine_status')).reshape(1,-1)
        mse=self.model.calc_mse(for_mse)
        self.mse_buff.append(mse)
        return mse, s, t

    def determine_batch_num(self):
        match self.PDR:
            case _ if self.PDR>0.95:
                return 1
            case _ if self.PDR>0.9:
                return 2
            case _ if self.PDR>0.85:
                return 3
            case _ if self.PDR>0.8:
                return 4
            case _:
                return 5
            

    def get_important_important_batch(self):
        batch_not_found=True
        if self.use_PDR:
            important_batches_tar = self.determine_batch_num()
        else:
            important_batches_tar=0
        print("Analyzing samples")
        #NUM_BUF_SAMPLES=100
        NUM_BUF_SAMPLES=100*(1-self.PDR) if self.use_PDR else 100
        print("PDR is", self.PDR, "So Number of samples is: ", NUM_BUF_SAMPLES)
        while batch_not_found:
            mse, s, t = self.analyze_samples()
            if mse>2:
                print("Found sample")
                samples, timestamps= self.get_previous_X_samples(NUM_BUF_SAMPLES)
                self.sample_buffer.extend(samples)
                self.timestamp_buffer.extend(timestamps)
                self.sample_buffer.append(s)
                self.timestamp_buffer.append(t)
                for i in range(NUM_BUF_SAMPLES):
                    s, t=self.get_sample()
                    self.sample_buffer.append(s)
                    self.timestamp_buffer.append(t)
                    self.mse_buff.append(self.mse_buff[-1])
                important_batches+=1
                if important_batches==important_batches_tar: #network parameter
                    batch_not_found=False
                    self.sample_buffer=np.array(self.sample_buffer)
                    filename=os.path.join(
                        'test_files',
                        str(self.timestamp_buffer[0]).replace(" ", "-").replace(":", "-")+'.avro'
                        )
                    #comment
                    AVRO.save_AVRO_default(self.sample_buffer, self.timestamp_buffer,self.schema_path, accuracy=10,path=filename, original_size=important_batches, codec='deflate')
                    self.total_sent_data+=os.path.getsize(filename)
                    self.send_file(self.TAR_IP, self.TAR_PORT_TCP,filename)
                    self.sample_buffer=[]
                    self.timestamp_buffer=[]
        return True


    def run(self, waittime=10):
        start=time.time()
        self.sample_buffer=[]
        self.timestamp_buffer=[]
        self.mse_buff=[]
        done_sending=False
        #self.get_important_important_batch()
        try:
            self.Ready_to_start()
            file, transmission_time = self.file_Q.get(timeout=3)
        except queue.Empty:
            time.sleep(1)
        while True:
            try:
                file, transmission_time= self.file_Q.get(timeout=2)
                print(file)
                if ".tflite" in file or '.zip' in file:
                    self.received_model(file)
                self.file_Q.task_done()
                self.get_important_important_batch()
            except queue.Empty:
                print("waiting for model")
            except Exception as e:
                print(e)
            if self.index>=self.len_of_dataset:
                pd.DataFrame(self.mse_buff).to_csv('test_files/mse_data.csv')
                self.send_file(self.TAR_IP, self.TAR_PORT_TCP,"test_files/mse_data.csv")
                self.send_done_sending()
                print("done")
                print("Time elapsed: ", time.time()-start)
                print("Transmitting time: ", self.time_transmitting)
                print("Total data sent(KB): ", self.total_sent_data/1024)
                self.make_end_plot(self.mse_buff)
                remove_all_avro_files('test_files')
                exit()
        
    def received_model(self, path):
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if '.zip' in path:
            with zipfile.ZipFile(path, 'r') as zipf:
                output_folder=str(path).split('/')[0]
                zipf.extractall(output_folder)
        destination_path=os.path.join(self.model_path, 'autoencoder.tflite')
        shutil.move(os.path.join(output_folder, 'autoencoder.tflite'), destination_path)
        self.model.load_model()
    
    def get_previous_X_samples(self, X):
        sample=self.data.iloc[self.index-X:self.index].values.tolist()
        timestamp=self.timestamps.iloc[self.index-X:self.index].tolist()
        return sample, timestamp

    def get_sample(self):
        #should fetch the next sample in the dataset
        sample=self.data.iloc[self.index]
        timestamp=self.timestamps.iloc[self.index]
        self.index+=1
        return sample, timestamp
    
    def calculate_fault_detection_score(self, mse_buf, adjusted_broken_indices, threshold=2, window=100):
        FP=0
        TP=0
        score = 0
        mse_buf = np.array(mse_buf)
        fault_mask = np.zeros_like(mse_buf, dtype=bool)
        
        # Mark indices within 100 points of any fault index
        for fault_index in adjusted_broken_indices:
            start = max(0, fault_index - window)
            end = min(len(mse_buf), fault_index + window + 1)
            fault_mask[start:end] = True
        
        # Iterate through MSE values and calculate score
        for i, mse in enumerate(mse_buf):
            if fault_mask[i]:
                # Inside fault zone
                if mse>threshold:
                    TP+=1
                score += (mse - threshold)  # Positive if above 2, negative if below 2
            else:
                # Outside fault zone
                if mse > threshold:
                    FP+=1
                    score -= (mse - threshold)  # Penalty for high MSE outside fault zone
        return score, TP, FP

    def make_end_plot(self, mse):
        file_path = "datasets/sensor.csv"
        df = pd.read_csv(file_path)

        # Ensure 'machine_status' column exists
        if "machine_status" not in df.columns:
            raise ValueError("Column 'machine_status' not found in dataset")

        # Find indices where machine_status is 'BROKEN'
        broken_indices = df.index[df["machine_status"] == "BROKEN"].tolist()
        last_index = df.index[-1]
        adjusted_broken_indices = [idx - self.start_offset for idx in broken_indices if idx >= self.start_offset]

        mse_buf = mse  

        try:
            score, TP, FP=self.calculate_fault_detection_score(mse_buf=mse_buf, adjusted_broken_indices=adjusted_broken_indices)
            print("Success SCORE: ", score)
            print("TP: ", TP)
            print("FP: ", FP)
        except Exception as e:
            print(e)
        
        # Plot the mse_buf values
        plt.figure(figsize=(10, 5))
        plt.plot(mse_buf, label="Mean Squared Error")

        # Plot vertical lines where 'machine_status' is 'BROKEN'
        for idx in adjusted_broken_indices:
            plt.axvline(x=idx, color='r', linestyle='--', alpha=0.7, label="BROKEN" if idx == adjusted_broken_indices[0] else "")
        plt.axvline(x=(last_index-self.start_offset), color='g', linestyle='--', alpha=0.7, label="Last Entry")


        # Labels and legend
        plt.xlabel("Index")
        plt.ylabel("MSE Values")
        plt.title("Singular Value Plot with BROKEN Machine Status")
        plt.legend()
        plt.show()

#fd
bs=edge_device("received")
bs.run()
#bs.send_file("test_files/PEPE.jpeg")
#bs.receive_file()