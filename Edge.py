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
import warnings
from sklearn.exceptions import ConvergenceWarning
import psutil
import threading
import IoT_energy
import subprocess
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", module="sklearn")

class edge_device(TCP_COM):
    def __init__(self, REC_FILE_PATH, input):
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
            
            #rate_kbps=configs['bandwidth_limit_kbps']
            #burst_kbps=configs['burst_limit_kbps']
            #self.rate_kbps=input
            self.rate_kbps=1000
            self.burst_kbps=16#input
            self.latency_ms=configs['buffering_latency_ms']
            self.packet_loss_pct=configs['packet_loss_pct']
            #delay_ms=configs['base_delay_ms']
            #jitter_ms=configs['jitter_ms']
            #packet_loss_pct=None
            self.delay_ms=None
            self.jitter_ms=None
            self.nc.set_network_conditions(self.rate_kbps, self.burst_kbps, self.latency_ms, self.packet_loss_pct, self.delay_ms, self.jitter_ms)
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
        important_batches_tar=1
        important_batches=0
        #print("Analyzing samples")
        NUM_BUF_SAMPLES=input
        #NUM_BUF_SAMPLES=int(max(max(1.74*(self.throughput/8 - 8),0),60))
        print("Throughput ", self.throughput, "NUMSAMPLES: ", NUM_BUF_SAMPLES, "Buffering: ", important_batches_tar)
        time.sleep(0.01)
        while batch_not_found:
            
            rare, mse, s, t = self.analyze_samples()
            self.energy_buff.append(self.energy_model.inference_energy(self.model_quantization))
            self.measured_throughput_buf.append(self.throughput)
            self.throughput_buf.append(self.rate_kbps)
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
                    self.throughput_buf.append(self.rate_kbps)
                    self.energy_buff.append(0)
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
                    #AVRO.save_AVRO_default(self.sample_buffer, self.timestamp_buffer,self.schema_path, accuracy=10,path=filename, original_size=important_batches)
                    self.total_sent_data+=os.path.getsize(filename)+20
                    tx_time=self.send_file(self.TAR_IP, self.TAR_PORT_TCP,filename)
                    self.energy_buff[-1]+=self.energy_model.transmission_energy(tx_time)
                    self.sample_buffer=[]
                    self.timestamp_buffer=[]
        return True


    def run(self, input):
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
                #self.energy_buff[-1]+=self.energy_model.receiving_energy(rec_time)
                files_received+=1
                print(file)
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
                print("mse",np.shape(self.mse_buff))
                print("energy",np.shape(self.energy_buff[1:]))
                print("th",np.shape(self.throughput_buf))
                print("Mth",np.shape(self.measured_throughput_buf))
                data = {
                    'mse': self.mse_buff,
                    'energy': self.energy_buff[1:],
                    'throughput': self.throughput_buf,
                    'measured_throughput': self.measured_throughput_buf
                }

                df = pd.DataFrame(data)

                # Save to CSV
                df.to_csv("test_files/full_IoTresult.csv", index=False)
                #self.send_file(self.TAR_IP, self.TAR_PORT_TCP,"test_files/mse_data.csv")
                try:
                    self.send_done_sending()
                except:
                    print("Failed to send done")
                print("done")
                print("Received, ", files_received, "files")
                print("Time elapsed: ", time.time()-start)
                print("Transmitting time: ", self.time_transmitting)
                print("Total data sent(KB): ", self.total_sent_data/1024)
                #self.make_end_plot(self.mse_buff)
                remove_all_avro_files('test_files')
                self.stop_TCP()
                Running=False
                subprocess.run(f"sudo tc qdisc del dev {self.configs['edgeNET_INTERFACE']} root", shell=True)
        return self.time_transmitting, self.time_receiving, self.total_sent_data, self.total_received_data, self.num_inferences, np.mean(self.throughputs)

        
    def received_model(self, path):
        """
        Decompresses a received model and loads it into memory.

        Parameters:
        ----------
        path: string model path
        --------
        """
        model_name=str(path).split('/')[-1].split('.')[0]
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)
        if '.zip' in path:
            with zipfile.ZipFile(path, 'r') as zipf:
                output_folder=str(path).split('/')[0]
                zipf.extractall(output_folder)
        if "Q" in model_name:
            model_name=model_name[1:]
        destination_path=os.path.join(self.model_path, self.model.model_name+'.tflite')
        shutil.move(os.path.join(output_folder, model_name+'.tflite'), destination_path)
        self.total_received_data += os.path.getsize(destination_path)+20
        self.model.load_model()
    
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

        
        # Plot the mse_buf values
        plt.figure(figsize=(10, 5))
        plt.plot(mse_buf, label="Autoencoder MSE Output")
        plt.ylim(-0.1, min(10, max(mse_buf)))
        # Plot vertical lines where 'machine_status' is 'BROKEN'
        for idx in adjusted_broken_indices:
            plt.axvline(x=idx, color='r', linestyle='--', alpha=0.7, label="Fault" if idx == adjusted_broken_indices[0] else "")
        #plt.axvline(x=(last_index-self.start_offset), color='g', linestyle='--', alpha=0.7, label="Last Entry")


        # Labels and legend
        plt.xlabel("Index")
        plt.ylabel("Autoencoder MSE Output")
        plt.title("Autoencoder fault detector")
        plt.legend()
        #plt.show()

#bs=edge_device("received", 1000)
#bs.run(1000)
