import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="numpy.core.fromnumeric")
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
from alternative_iot_models import mlp_classifier
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
import threading
import IoT_energy
import subprocess
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", module="sklearn")

class edge_device(TCP_COM):
    def __init__(self, REC_FILE_PATH, input, eQ, bQ):
        """
        Initializes an IoT device object.

        Parameters:
        ----------
        REC_FILE_PATH : string acting as path to folder where received files are to be stored.           
        input : test parameter.
        --------
        """
        self.inference_batch=0
        self.use_PDR=False
        self.throughputs=[]
        self.total_sent_data=0
        self.total_received_data=0
        self.num_inferences=0
        self.device_type="edge"
        self.model_path="edge_models"
        with open("configs.json", "r") as file:
            configs = json.load(file)
        self.file_Q=queue.Queue()
        super().__init__(eQ, bQ, REC_FILE_PATH, self.device_type, self.file_Q, throughput=input)
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
        self.model = IoT_model.IoT_model("test_files/initial_data.csv", 0.2)
        self.energy_model=IoT_energy.energy()
        self.configs=configs
        #self.model = mlp_classifier("test_files/initial_data.csv", input)
        #self.model.load_model()

    


    def analyze_samples(self):
        """
        Fetch a sample and check for probability of being a fault.

        Returns
        rare: binary value for whether rare or not.
        mse: mse score/ model output of sample
        s: sample
        t: timestamp
        """
        s, t=self.get_sample()    
        if self.inference_batch==0:
            for_mse=np.array(s.drop('machine_status')).reshape(1,-1)
        else:
            for_mse=s.drop(columns='machine_status')
        rare, mse=self.model.check_sample(for_mse)
        self.num_inferences+=1
        self.mse_buff.append(mse)
        return rare, mse, s, t


            

    def get_important_important_batch(self, input):
        """
        Iterates through samples until it finds a rare one,keeps a buffer of length N 
        of previous samples and collects future context of length N.
        Decides on N as well as amount of buffering.
        When it is evaluated that all data necessary for a package has been collected,
        it serializes the data with avro and transmits it to the server.

        Parameters:
        ----------         
        input : test parameter.
        --------
        """
        batch_not_found=True
        if self.throughput:
            self.throughputs.append(self.throughput)
            if self.throughput<250:
                important_batches_tar=3
            elif self.throughput<400:
                important_batches_tar=2
            else:
                important_batches_tar=1
        else:
            important_batches_tar=1
        important_batches=0
        #print("Analyzing samples")
        #NUM_BUF_SAMPLES=input
        NUM_BUF_SAMPLES=int(max(max(1.74*(self.throughput/8 - 8),0),60))
        print("Throughput ", self.throughput, "NUMSAMPLES: ", NUM_BUF_SAMPLES, "Buffering: ", important_batches_tar)
        time.sleep(0.01)
        while batch_not_found:
            
            rare, mse, s, t = self.analyze_samples()
            self.energy_buff.append(self.energy_model.inference_energy(self.model_quantization))
            self.measured_throughput_buf.append(self.throughput)
            self.throughput_buf.append(self.throughput)
            self.samples_since_last_batch+=1
            if rare:
                self.samples_since_last_batch-=1
                print("Getting last :", min(NUM_BUF_SAMPLES, self.samples_since_last_batch), "samples")
                samples, timestamps= self.get_previous_X_samples(min(NUM_BUF_SAMPLES, self.samples_since_last_batch))
                #samples, timestamps= self.get_previous_X_samples(NUM_BUF_SAMPLES)
                self.samples_since_last_batch=0
                try:
                    self.sample_buffer.extend(samples)
                    self.timestamp_buffer.extend(timestamps)
                except:
                    pass
                self.sample_buffer.append(s)
                self.timestamp_buffer.append(t)
                for i in range(int(NUM_BUF_SAMPLES)):
                    s, t=self.get_sample()
                    self.sample_buffer.append(s)
                    self.timestamp_buffer.append(t)
                    self.mse_buff.append(self.mse_buff[-1])
                    self.measured_throughput_buf.append(self.throughput)
                    self.throughput_buf.append(self.throughput)
                    self.energy_buff.append(0)
                important_batches+=1
                if important_batches==important_batches_tar: #network parameter
                    batch_not_found=False
                    self.sample_buffer=np.array(self.sample_buffer)
                    filename=os.path.join(
                        'test_files',
                        str(self.timestamp_buffer[0]).replace(" ", "-").replace(":", "-")+'.avro'
                        )
                    print(self.timestamp_buffer[0])
                    filename="test_files/avro_51.avsc"
                    #comment
                    #AVRO.save_AVRO_default(self.sample_buffer, self.timestamp_buffer,self.schema_path, accuracy=10,path=filename, original_size=important_batches, codec='deflate')
                    #AVRO.save_AVRO_default(self.sample_buffer, self.timestamp_buffer,self.schema_path, accuracy=10,path=filename, original_size=important_batches)
                    #self.total_sent_data+=os.path.getsize(filename)+20
                    tx_time=self.send_file((self.sample_buffer,self.timestamp_buffer, important_batches),filename)
                    self.energy_buff[-1]+=self.energy_model.transmission_energy(tx_time)
                    self.sample_buffer=[]
                    self.timestamp_buffer=[]
        return True


    def run(self, input, resultQ):
        """
        Runs the basic IoT routine of checking samples, collecting context, then transmitting 
        the samples and receiving an updated model afterwards.

        Parameters:
        ----------
        input: test parameter
        --------
        Returns:
        self.time_transmitting: total measured time spent transmitting.
        self.time_receiving: total measured time spent receiving.
        self.total_sent_data: total bytes transmitted
        self.total_received_data: total bytes received.
        self.num_inferences: total number of inferences
        self.throughputs: average throughput
        """
        Running=True
        start=time.time()
        self.sample_buffer=[]
        self.timestamp_buffer=[]
        self.mse_buff=[]
        self.energy_buff=[]
        self.throughput_buf=[]
        self.measured_throughput_buf=[]
        self.energy_buff.append(0)
        done_sending=False
        self.samples_since_last_batch=0
        files_received=0
        try:
            self.Ready_to_start()
            file, rec_time = self.file_Q.get(timeout=3)
        except queue.Empty:
            time.sleep(1)
        while Running:
            try:
                file, rec_time= self.file_Q.get(timeout=2)
                files_received+=1
                if ".tflite" in file or '.zip' in file:
                    self.received_model(file)
                    #pass
                self.file_Q.task_done()
                #self.rate_kbps-=10
                #self.nc.set_network_conditions(self.rate_kbps, self.burst_kbps, self.latency_ms, self.packet_loss_pct, self.delay_ms, self.jitter_ms)
                self.get_important_important_batch(input)
                #self.send_ACK()
                #self.index+=50000
            except queue.Empty:
                #print("waiting for model")
                pass
            except Exception as e:
                print(e)
            if self.index>=self.len_of_dataset:
                #pd.DataFrame(self.mse_buff).to_csv('test_files/mse_data.csv')
                #pd.DataFrame(self.energy_buff).to_csv('test_files/energy_data.csv')
                
                #self.send_file(self.TAR_IP, self.TAR_PORT_TCP,"test_files/mse_data.csv")
                try:
                    self.send_done_sending()
                except:
                    print("Failed to send done")
                self.stop_TCP()
                Running=False
                #subprocess.run(f"sudo tc qdisc del dev {self.configs['edgeNET_INTERFACE']} root", shell=True)
        resultQ.put({
            "t_T":self.time_transmitting, 
            "t_R":self.time_receiving,
            "datasent":self.total_sent_data,
            "data_recevied":self.total_received_data,
            "inferences":self.num_inferences,
            "avg_throughput":np.mean(self.throughputs)})
        return None

        
    def received_model(self, path):
        """
        Decompresses a received model and loads it into memory.

        Parameters:
        ----------
        path: string model path
        --------
        """
        if "Q" in path:
            path=path[1:]
        path="models/"+path
        model_name=str(path).split('/')[-1].split('.')[0]
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if '.zip' in path:
            with zipfile.ZipFile(path, 'r') as zipf:
                output_folder=str(path).split('/')[0]
                zipf.extractall(output_folder)
        destination_path=os.path.join(self.model_path, self.model.model_name+'.tflite')
        shutil.move(os.path.join(output_folder, model_name+'.tflite'), destination_path)
        self.total_received_data += os.path.getsize(destination_path)+20
        self.model.load_model(destination_path)
    
    def get_previous_X_samples(self, X):
        """
        Fetches N samples from buffer

        Parameters:
        ----------
        X: int number of samples to be fetched
        --------
        """
        sample=self.data.iloc[self.index-X:self.index].values.tolist()
        timestamp=self.timestamps.iloc[self.index-X:self.index].tolist()
        return sample, timestamp

    def get_sample(self):
        #should fetch the next sample in the dataset
        sample=self.data.iloc[self.index]
        timestamp=self.timestamps.iloc[self.index]
        self.index+=1+self.inference_batch
        return sample, timestamp
    


#bs=edge_device("received", 1000)
#bs.run(1000)
